export UncertaintyMethod, MCDropout, DeepEnsemble, Bootstrap
export UncertaintyResult, estimate_uncertainty

# =============================================================================
# Uncertainty quantification for hybrid models
# =============================================================================
#
# A single generic entry point, `estimate_uncertainty`, selects between several
# uncertainty quantification (UQ) methods via the `method` argument:
#
#   * `MCDropout`     — Monte-Carlo dropout: repeated stochastic forward passes
#                       with dropout kept active at inference time.
#   * `DeepEnsemble`  — train several models from different random seeds (and,
#                       optionally, on bootstrap resamples) and use the spread
#                       across members as the uncertainty estimate.
#   * `Bootstrap`     — train several models on bootstrap resamples of the data
#                       (with a fixed initialization seed) so the spread is driven
#                       purely by data resampling. Well suited to small datasets.
#
# What is being quantified?
# -------------------------
# All three methods characterise how much the *model prediction* varies under a
# source of randomness (dropout masks, weight initialization, or the data
# sample). That spread is predominantly *epistemic* (model/parameter)
# uncertainty. To additionally capture *aleatoric* (irreducible data-noise)
# uncertainty, pair these with a mechanistic/neural output that predicts a noise
# scale and a Gaussian negative-log-likelihood training loss; the spread returned
# here then reflects epistemic uncertainty on top of that learned noise model.
# The `combine` helper on the result also reports the total predictive spread.

"""
    UncertaintyMethod

Abstract supertype for the uncertainty quantification methods accepted by
[`estimate_uncertainty`](@ref): [`MCDropout`](@ref), [`DeepEnsemble`](@ref) and
[`Bootstrap`](@ref).
"""
abstract type UncertaintyMethod end

"""
    MCDropout(; n_samples = 50)

Monte-Carlo dropout. Runs `n_samples` stochastic forward passes of an already
trained model with dropout kept active, and uses the spread across passes as the
uncertainty estimate.

Requirements checked by [`estimate_uncertainty`](@ref):

  * the network must contain `Dropout` layer(s) — otherwise every pass is
    identical and MC dropout is meaningless;
  * the model must **not** estimate any global (physical) parameters. Global
    parameters are single point estimates that dropout does not perturb, so MC
    dropout would report zero uncertainty for them and misrepresent the
    prediction. Use [`DeepEnsemble`](@ref) or [`Bootstrap`](@ref) instead.
"""
struct MCDropout <: UncertaintyMethod
    n_samples::Int
end
MCDropout(; n_samples::Int = 50) = MCDropout(n_samples)

"""
    DeepEnsemble(; n_models = 5, bootstrap = false, seeds = nothing)

Deep ensemble. Trains `n_models` independent models from different random seeds
and uses the spread across members as the uncertainty estimate.

  * `bootstrap = false` (default): every member sees the full training data and
    differs only through its random initialization seed.
  * `bootstrap = true`: every member is trained on a bootstrap resample **and**
    a different seed. This combines the two randomness sources and is generally
    the most rigorous option — the seeds explore different optima while the
    resampling propagates sampling variability of the (finite) dataset. It is
    especially recommended for small datasets (see [`Bootstrap`](@ref)).
  * `seeds`: optionally supply the exact per-member seeds (length must match
    `n_models`); otherwise deterministic seeds are derived automatically.
"""
struct DeepEnsemble <: UncertaintyMethod
    n_models::Int
    bootstrap::Bool
    seeds::Union{Nothing, Vector{Int}}
end
function DeepEnsemble(; n_models::Int = 5, bootstrap::Bool = false, seeds = nothing)
    return DeepEnsemble(n_models, bootstrap, seeds === nothing ? nothing : collect(Int, seeds))
end

"""
    Bootstrap(; n_models = 20, seed = 161803)

Bootstrap ensemble. Trains `n_models` models on bootstrap resamples (sampling
observations with replacement) of the training data, all from the **same**
initialization `seed`, so the spread across members is driven purely by data
resampling.

Why bootstrap for small datasets? With few observations the dominant uncertainty
is *which* data points were sampled. A plain deep ensemble (varying only the
seed) trains every member on the exact same small dataset and therefore tends to
*underestimate* this sampling variability — the members converge to very similar
fits. Bootstrapping resamples the data for each member, so the ensemble spread
directly reflects the sampling distribution of the estimator, giving
better-calibrated intervals in the small-`n` regime. Combining both sources
(`DeepEnsemble(bootstrap = true)`) is the most rigorous choice.
"""
struct Bootstrap <: UncertaintyMethod
    n_models::Int
    seed::Int
end
Bootstrap(; n_models::Int = 20, seed::Int = 161803) = Bootstrap(n_models, seed)

"""
    UncertaintyResult

Result of [`estimate_uncertainty`](@ref).

# Fields
- `method`: the [`UncertaintyMethod`](@ref) used.
- `targets`: the target variable names.
- `mean`: `NamedTuple` of per-target mean prediction vectors.
- `std`: `NamedTuple` of per-target standard-deviation (uncertainty) vectors.
- `lower` / `upper`: `NamedTuple`s of per-target lower/upper predictive-interval
  bounds at the requested `quantiles`.
- `samples`: `NamedTuple` of per-target `n_obs × n` matrices with the raw
  per-sample (MC pass or ensemble member) predictions.
- `params`: `NamedTuple` keyed by the model's estimated global parameter names;
  each entry is a `(; mean, std, samples)` `NamedTuple` giving the spread of that
  physical parameter across ensemble members. Empty for MC dropout (which is not
  applicable when global parameters are estimated).
- `n`: number of MC passes / ensemble members that contributed.
- `quantiles`: the `(lower, upper)` probability levels used for the interval.
- `metadata`: method-specific extra information (seeds, dropout rates, …).
"""
struct UncertaintyResult
    method::UncertaintyMethod
    targets::Vector{Symbol}
    mean::NamedTuple
    std::NamedTuple
    lower::NamedTuple
    upper::NamedTuple
    samples::NamedTuple
    params::NamedTuple
    n::Int
    quantiles::Tuple{Float64, Float64}
    metadata::NamedTuple
end

function Base.show(io::IO, ::MIME"text/plain", r::UncertaintyResult)
    println(io, "UncertaintyResult ($(nameof(typeof(r.method))), n = $(r.n))")
    println(io, "  quantiles: $(r.quantiles)")
    for t in r.targets
        μ = r.mean[t]
        σ = r.std[t]
        println(io, "  • target $(t): mean std = $(round(mean(σ); digits = 4)) over $(length(μ)) obs ",
            "(σ range $(round(minimum(σ); digits = 4))–$(round(maximum(σ); digits = 4)))")
    end
    for g in keys(r.params)
        p = r.params[g]
        println(io, "  • global param $(g): $(round(p.mean; digits = 4)) ± $(round(p.std; digits = 4))")
    end
    if !isempty(r.metadata)
        println(io, "  metadata: $(r.metadata)")
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Dropout detection
# -----------------------------------------------------------------------------

# Collect dropout probabilities from an arbitrary Lux layer tree. Returns the
# `p` of every `Dropout` layer found (recursing into `Chain`s and wrapper layers).
_dropout_rates(d::Dropout) = Float64[Float64(d.p)]
function _dropout_rates(c::Chain)
    rates = Float64[]
    for l in values(c.layers)
        append!(rates, _dropout_rates(l))
    end
    return rates
end
function _dropout_rates(x)
    # Generic fallback: recurse into common wrapper fields (`layers`, `layer`).
    if hasproperty(x, :layers)
        rates = Float64[]
        for l in values(getproperty(x, :layers))
            append!(rates, _dropout_rates(l))
        end
        return rates
    elseif hasproperty(x, :layer)
        return _dropout_rates(getproperty(x, :layer))
    end
    return Float64[]
end

"""
    dropout_rates(model::HybridModel) -> Vector{Float64}

Return the dropout probabilities of every `Dropout` layer in the model's neural
network(s). An empty vector means the network has no dropout, so MC dropout is
not applicable.
"""
function dropout_rates(model::HybridModel)
    nns = model.NNs
    if nns isa NamedTuple
        rates = Float64[]
        for nn in values(nns)
            append!(rates, _dropout_rates(nn))
        end
        return rates
    else
        return _dropout_rates(nns)
    end
end

# -----------------------------------------------------------------------------
# Prediction / aggregation helpers
# -----------------------------------------------------------------------------

# Forward pass returning a NamedTuple of per-target prediction vectors.
function _predict_targets(model::HybridModel, x, ps, st)
    out, _ = model(x, ps, st)
    return NamedTuple(t => vec(out[t]) for t in model.targets)
end

# Forward pass returning both per-target predictions and the scalar value of each
# estimated global parameter (constant across observations).
function _predict_targets_and_params(model::HybridModel, x, ps, st)
    out, _ = model(x, ps, st)
    tgt = NamedTuple(t => vec(out[t]) for t in model.targets)
    prm = NamedTuple(g => Float64(first(vec(out.parameters[g]))) for g in model.global_param_names)
    return tgt, prm
end

# Aggregate a Vector (over members) of per-parameter NamedTuples into
# (; mean, std, samples) summaries per global parameter.
function _aggregate_params(param_preds::Vector{<:NamedTuple}, global_names)
    isempty(global_names) && return NamedTuple()
    out = NamedTuple()
    for g in global_names
        vals = Float64[p[g] for p in param_preds]
        entry = (; mean = mean(vals), std = Statistics.std(vals), samples = vals)
        out = merge(out, NamedTuple{(g,)}((entry,)))
    end
    return out
end

# Aggregate a Vector (over samples/members) of per-target NamedTuples into
# mean/std/quantile summaries plus the raw sample matrices.
function _aggregate(preds::Vector{<:NamedTuple}, targets, quantiles)
    n = length(preds)
    n >= 2 || throw(ArgumentError("Need at least 2 samples/members to quantify uncertainty, got $n."))
    ql, qu = quantiles

    means = NamedTuple()
    stds = NamedTuple()
    lowers = NamedTuple()
    uppers = NamedTuple()
    samples = NamedTuple()

    for t in targets
        M = reduce(hcat, (p[t] for p in preds))   # n_obs × n
        μ = vec(mean(M; dims = 2))
        σ = vec(Statistics.std(M; dims = 2))       # corrected (sample) std
        lo = [quantile(view(M, i, :), ql) for i in 1:size(M, 1)]
        hi = [quantile(view(M, i, :), qu) for i in 1:size(M, 1)]
        means = merge(means, NamedTuple{(t,)}((μ,)))
        stds = merge(stds, NamedTuple{(t,)}((σ,)))
        lowers = merge(lowers, NamedTuple{(t,)}((lo,)))
        uppers = merge(uppers, NamedTuple{(t,)}((hi,)))
        samples = merge(samples, NamedTuple{(t,)}((M,)))
    end
    return means, stds, lowers, uppers, samples
end

# -----------------------------------------------------------------------------
# Bootstrap resampling of the raw data (variable × observation layout)
# -----------------------------------------------------------------------------

"""
    bootstrap_resample(data, rng) -> data′

Resample observations (with replacement) of a 2-D dataset laid out as
`variable × observation` (the layout produced by [`to_keyedArray`](@ref) /
[`to_dimArray`](@ref)), or a `DataFrame` (row = observation). Sequence/3-D data
is rejected because resampling individual timesteps would destroy the temporal
structure.
"""
function bootstrap_resample(data::KeyedArray, rng::AbstractRNG)
    ndims(data) == 2 || throw(ArgumentError("Bootstrap resampling supports 2-D (variable × observation) data only; got $(ndims(data))-D. Bootstrapping sequence data would break temporal structure."))
    n = size(data, 2)
    idx = rand(rng, 1:n, n)
    raw = _raw_array(data)[:, idx]
    vars = collect(Symbol.(axiskeys(data, :variable)))
    return KeyedArray(raw; variable = vars, batch_size = 1:n)
end

function bootstrap_resample(data::AbstractDimArray, rng::AbstractRNG)
    ndims(data) == 2 || throw(ArgumentError("Bootstrap resampling supports 2-D (variable × observation) data only; got $(ndims(data))-D."))
    n = size(data, 2)
    idx = rand(rng, 1:n, n)
    raw = Array(parent(data))[:, idx]
    vars = collect(Symbol.(lookup(data, :variable)))
    return DimArray(raw, (Dim{:variable}(vars), Dim{:batch_size}(1:n)))
end

function bootstrap_resample(data::DataFrame, rng::AbstractRNG)
    n = size(data, 1)
    idx = rand(rng, 1:n, n)
    return data[idx, :]
end

function bootstrap_resample(::Tuple, ::AbstractRNG)
    throw(ArgumentError("Bootstrap resampling needs the raw data (DataFrame/KeyedArray/DimArray), not an already-prepared `(x, y)` tuple."))
end

# -----------------------------------------------------------------------------
# Generic entry point
# -----------------------------------------------------------------------------

"""
    estimate_uncertainty(method, model, data, [train_output]; kwargs...) -> UncertaintyResult

Generic uncertainty quantification for a hybrid `model`. The `method` selects the
algorithm — [`MCDropout`](@ref), [`DeepEnsemble`](@ref) or [`Bootstrap`](@ref).

# Methods

  * `estimate_uncertainty(m::MCDropout, model, data, train_output; …)` — uses the
    trained parameters/state in `train_output` (a [`TrainResults`](@ref)); no
    retraining. Errors if the network has no dropout, or if the model estimates
    global parameters.
  * `estimate_uncertainty(m::DeepEnsemble, model, data; train_kwargs...)` and
    `estimate_uncertainty(m::Bootstrap, model, data; train_kwargs...)` — (re)train
    `n_models` members; all `train_kwargs` are forwarded to [`train`](@ref).

# Common keyword arguments
- `eval_data`: data on which predictions/uncertainty are computed. Defaults to
  `data` (the training data). Pass a held-out set for out-of-sample uncertainty.
- `quantiles = (0.025, 0.975)`: probability levels of the predictive interval.
- `verbose = true`: (ensemble/bootstrap) print per-member progress.
"""
function estimate_uncertainty end

# --- MC dropout --------------------------------------------------------------
function estimate_uncertainty(
        method::MCDropout, model::HybridModel, data, train_output::TrainResults;
        eval_data = data, quantiles::Tuple{<:Real, <:Real} = (0.025, 0.975),
    )
    # Guard 1: MC dropout cannot represent uncertainty of estimated global params.
    if !isempty(model.global_param_names)
        throw(ArgumentError(
            "MC dropout is not applicable: the model estimates global parameter(s) " *
            "$(model.global_param_names). Global (physical) parameters are single point " *
            "estimates that dropout does not perturb, so MC dropout would report no " *
            "uncertainty for them. Use DeepEnsemble or Bootstrap instead."
        ))
    end

    # Guard 2: dropout must be present (and was therefore active during training,
    # since training always runs in trainmode in EasyHybrid).
    rates = dropout_rates(model)
    if isempty(rates)
        throw(ArgumentError(
            "MC dropout requires Dropout layer(s) in the neural network, but none were " *
            "found. Rebuild the model with dropout, e.g. " *
            "hidden_layers = Chain(Dense(n, h, relu), Dropout(0.2), Dense(h, h, relu), Dropout(0.2))."
        ))
    end
    if all(iszero, rates)
        @warn "All Dropout layers have probability 0; MC dropout will produce zero uncertainty."
    end

    ps, st = train_output.ps, train_output.st
    x = prepare_data(model, eval_data)[1]

    # Keep dropout active at inference. Threading the returned state advances the
    # dropout RNG so each pass draws a fresh mask. (BatchNorm, if present, uses
    # batch statistics of the evaluation batch in this mode.)
    st_mc = LuxCore.trainmode(st)
    preds = Vector{NamedTuple}(undef, method.n_samples)
    for i in 1:method.n_samples
        out, st_mc = model(x, ps, st_mc)
        preds[i] = NamedTuple(t => vec(out[t]) for t in model.targets)
    end

    q = (Float64(quantiles[1]), Float64(quantiles[2]))
    means, stds, lowers, uppers, samples = _aggregate(preds, model.targets, q)
    meta = (; dropout_rates = rates)
    return UncertaintyResult(method, copy(model.targets), means, stds, lowers, uppers, samples, NamedTuple(), method.n_samples, q, meta)
end

# --- Deep ensemble & bootstrap (shared runner) -------------------------------
function _ensemble_uncertainty(
        method::UncertaintyMethod, model::HybridModel, data;
        seeds::Vector{Int}, use_bootstrap::Bool, eval_data, quantiles, verbose, train_kwargs,
    )
    x_eval = prepare_data(model, eval_data)[1]
    n_models = length(seeds)

    # Sensible defaults for UQ retraining; user train_kwargs win on conflict.
    base_kwargs = (; plotting = false, show_progress = false, save_training = false)
    merged_kwargs = merge(base_kwargs, train_kwargs)

    preds = Vector{NamedTuple}(undef, n_models)
    param_preds = Vector{NamedTuple}(undef, n_models)
    for (i, seed) in enumerate(seeds)
        # A separate RNG per member keeps bootstrap draws reproducible and
        # independent of the training seed.
        member_data = use_bootstrap ? bootstrap_resample(data, Random.MersenneTwister(seed)) : data
        member_kwargs = merge(merged_kwargs, (; random_seed = seed))
        verbose && @info "Uncertainty: training ensemble member $i/$n_models (seed = $seed, bootstrap = $use_bootstrap)"
        res = train(model, member_data; member_kwargs...)
        res === nothing && throw(ErrorException("Training member $i returned nothing (data preparation failed)."))
        preds[i], param_preds[i] = _predict_targets_and_params(model, x_eval, res.ps, LuxCore.testmode(res.st))
    end

    q = (Float64(quantiles[1]), Float64(quantiles[2]))
    means, stds, lowers, uppers, samples = _aggregate(preds, model.targets, q)
    params = _aggregate_params(param_preds, model.global_param_names)
    meta = (; seeds = seeds, bootstrap = use_bootstrap)
    return UncertaintyResult(method, copy(model.targets), means, stds, lowers, uppers, samples, params, n_models, q, meta)
end

function estimate_uncertainty(
        method::DeepEnsemble, model::HybridModel, data;
        eval_data = data, quantiles::Tuple{<:Real, <:Real} = (0.025, 0.975),
        verbose::Bool = true, train_kwargs...,
    )
    seeds = if method.seeds === nothing
        collect(1:method.n_models) .* 101 .+ 17   # deterministic, well-spread
    else
        length(method.seeds) == method.n_models ||
            throw(ArgumentError("Provided $(length(method.seeds)) seeds but n_models = $(method.n_models)."))
        method.seeds
    end
    return _ensemble_uncertainty(
        method, model, data;
        seeds = seeds, use_bootstrap = method.bootstrap,
        eval_data = eval_data, quantiles = quantiles, verbose = verbose,
        train_kwargs = NamedTuple(train_kwargs),
    )
end

function estimate_uncertainty(
        method::Bootstrap, model::HybridModel, data;
        eval_data = data, quantiles::Tuple{<:Real, <:Real} = (0.025, 0.975),
        verbose::Bool = true, train_kwargs...,
    )
    # Same initialization seed for every member: the spread comes purely from the
    # bootstrap resampling. Distinct resampling seeds are derived from the base.
    seeds = fill(method.seed, method.n_models)
    # Give each member an independent resample while sharing the init seed by
    # offsetting the RNG seed used for resampling only.
    resample_seeds = collect(1:method.n_models) .+ method.seed
    x_eval = prepare_data(model, eval_data)[1]
    base_kwargs = merge((; plotting = false, show_progress = false, save_training = false), NamedTuple(train_kwargs))

    preds = Vector{NamedTuple}(undef, method.n_models)
    param_preds = Vector{NamedTuple}(undef, method.n_models)
    for i in 1:method.n_models
        member_data = bootstrap_resample(data, Random.MersenneTwister(resample_seeds[i]))
        member_kwargs = merge(base_kwargs, (; random_seed = method.seed))
        verbose && @info "Uncertainty: training bootstrap member $i/$(method.n_models) (init seed = $(method.seed), resample seed = $(resample_seeds[i]))"
        res = train(model, member_data; member_kwargs...)
        res === nothing && throw(ErrorException("Training bootstrap member $i returned nothing (data preparation failed)."))
        preds[i], param_preds[i] = _predict_targets_and_params(model, x_eval, res.ps, LuxCore.testmode(res.st))
    end

    q = (Float64(quantiles[1]), Float64(quantiles[2]))
    means, stds, lowers, uppers, samples = _aggregate(preds, model.targets, q)
    params = _aggregate_params(param_preds, model.global_param_names)
    meta = (; init_seed = method.seed, resample_seeds = resample_seeds, bootstrap = true)
    return UncertaintyResult(method, copy(model.targets), means, stds, lowers, uppers, samples, params, method.n_models, q, meta)
end
