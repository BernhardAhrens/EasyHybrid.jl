export UncertaintyMethod, MCDropout, DeepEnsemble, Bootstrap, SGLD
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
#   * `SGLD`          — Stochastic Gradient Langevin Dynamics: continue from a
#                       trained (MAP) point, injecting Gaussian noise into
#                       gradient steps so the trajectory samples an approximate
#                       posterior over the selected parameters.
#
# Parameter taxonomy (hybrid models)
# ----------------------------------
#   * Global physical parameters (`global_param_names`, e.g. Q10) are free
#     scalars in `ps`. Dropout does not perturb them; SGLD with `scope = :global`
#     (or `:all`) and ensembles do.
#   * Latent / process parameters (`neural_param_names`, e.g. rb) are *outputs*
#     of the neural network h(x; θ), not free variables. They cannot be sampled
#     independently. A posterior over latents is induced by sampling the NN
#     weights θ (MC dropout, SGLD `scope = :nn` / `:all`, or ensembles).
#   * NN weights live in `ps.ps` (and named NN branches).
#
# What is being quantified?
# -------------------------
# These methods characterise how much the *model prediction* (and, where
# applicable, global / latent parameters) varies under a source of randomness
# (dropout masks, weight initialization, data resampling, or Langevin noise).
# That spread is predominantly *epistemic* (model/parameter) uncertainty. To
# additionally capture *aleatoric* (irreducible data-noise) uncertainty, pair
# these with a mechanistic/neural output that predicts a noise scale and a
# Gaussian negative-log-likelihood training loss; the spread returned here then
# reflects epistemic uncertainty on top of that learned noise model.

"""
    UncertaintyMethod

Abstract supertype for the uncertainty quantification methods accepted by
[`estimate_uncertainty`](@ref): [`MCDropout`](@ref), [`DeepEnsemble`](@ref),
[`Bootstrap`](@ref) and [`SGLD`](@ref).
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
    prediction. Use [`SGLD`](@ref) (`scope = :global` or `:all`),
    [`DeepEnsemble`](@ref) or [`Bootstrap`](@ref) instead.
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
    SGLD(; n_samples = 50, n_burnin = 100, n_thin = 10, lr = 1f-3,
         temperature = 1f-2, batchsize = 64, scope = :all, seed = nothing)

Stochastic Gradient Langevin Dynamics (Welling & Teh, 2011). Starts from a MAP
estimate (`train_output`) and injects Gaussian noise into minibatch gradient
steps so the trajectory samples an approximate posterior:

```
θ ← θ − lr ∇L(θ) + √(2 lr T) ε,    ε ∼ 𝒩(0, I)
```

`L` is the training loss (MSE by default). That is not a true negative log
likelihood, so `temperature` is a practical knob rather than a calibrated
thermodynamic temperature. Prefer a Gaussian NLL (`training_loss` / `extra_loss`)
when a well-defined posterior is required.

# Scope (which parameters are sampled)

  * `:all` (default) — NN weights **and** global physical parameters.
  * `:global` — only global physical parameters (cheap; fills the gap that
    [`MCDropout`](@ref) cannot). Latent/process parameters stay at their MAP
    network output.
  * `:nn` — only NN weights. Global parameters stay at the MAP point; latents
    `h(x; θ)` then vary because θ does.

Latent / process parameters (`neural_param_names`) are NN outputs, not free
variables. SGLD does not sample them independently: their posterior is induced
by sampling the NN weights (`scope = :nn` or `:all`).
"""
struct SGLD <: UncertaintyMethod
    n_samples::Int
    n_burnin::Int
    n_thin::Int
    lr::Float32
    temperature::Float32
    batchsize::Int
    scope::Symbol
    seed::Union{Nothing, Int}
end
function SGLD(;
        n_samples::Int = 50,
        n_burnin::Int = 100,
        n_thin::Int = 10,
        lr::Real = 1.0f-3,
        temperature::Real = 1.0f-2,
        batchsize::Int = 64,
        scope::Symbol = :all,
        seed::Union{Nothing, Int} = nothing,
    )
    scope in (:all, :global, :nn) ||
        throw(ArgumentError("SGLD scope must be :all, :global or :nn; got $(scope)."))
    n_samples >= 2 || throw(ArgumentError("SGLD n_samples must be ≥ 2, got $n_samples."))
    n_thin >= 1 || throw(ArgumentError("SGLD n_thin must be ≥ 1, got $n_thin."))
    n_burnin >= 0 || throw(ArgumentError("SGLD n_burnin must be ≥ 0, got $n_burnin."))
    return SGLD(n_samples, n_burnin, n_thin, Float32(lr), Float32(temperature), batchsize, scope, seed)
end

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
  physical parameter across samples/members. Empty for MC dropout (which is not
  applicable when global parameters are estimated) and for SGLD `scope = :nn`.
- `latents`: `NamedTuple` keyed by the model's neural / process parameter names
  (`neural_param_names`); each entry is a `(; mean, std, samples)` `NamedTuple`
  of *per-observation* vectors / `n_obs × n` matrices. These are NN outputs
  `h(x; θ)`, so they vary when θ is perturbed (dropout, SGLD `:nn`/`:all`,
  ensembles) and stay put when only globals are sampled (SGLD `:global`).
- `n`: number of MC passes / ensemble members / Langevin samples that contributed.
- `quantiles`: the `(lower, upper)` probability levels used for the interval.
- `metadata`: method-specific extra information (seeds, dropout rates, SGLD
  hyperparameters, …).
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
    latents::NamedTuple
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
    for n in keys(r.latents)
        lat = r.latents[n]
        println(io, "  • latent $(n): mean std = $(round(mean(lat.std); digits = 4)) over $(length(lat.mean)) obs")
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
    dropout_rates(model) -> Vector{Float64}

Return the dropout probabilities of every `Dropout` layer in the model's neural
network(s). Works for `HybridModel`, `SingleNNModel` and `MultiNNModel`. An empty
vector means the network has no dropout, so MC dropout is not applicable.
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
dropout_rates(model::SingleNNModel) = _dropout_rates(model.NN)
function dropout_rates(model::MultiNNModel)
    rates = Float64[]
    for nn in values(model.NNs)
        append!(rates, _dropout_rates(nn))
    end
    return rates
end

# Model types supported by uncertainty quantification.
const UQModel = Union{HybridModel, SingleNNModel, MultiNNModel}

# Estimated global (physical) parameters, if any. Pure NN models have none.
_global_param_names(m::HybridModel) = m.global_param_names
_global_param_names(::UQModel) = Symbol[]

# Neural / process ("latent") parameters: per-observation NN outputs. Pure NN
# models expose targets directly and have no separate latent parameter block.
_neural_param_names(m::HybridModel) = m.neural_param_names
_neural_param_names(::UQModel) = Symbol[]

# -----------------------------------------------------------------------------
# Prediction / aggregation helpers
# -----------------------------------------------------------------------------

# Forward pass returning a NamedTuple of per-target prediction vectors.
function _predict_targets(model::UQModel, x, ps, st)
    out, _ = model(x, ps, st)
    return NamedTuple(t => vec(out[t]) for t in model.targets)
end

# Forward pass returning per-target predictions, scalar global parameters, and
# per-observation latent / process parameters.
function _predict_full(model::UQModel, x, ps, st)
    out, _ = model(x, ps, st)
    tgt = NamedTuple(t => vec(out[t]) for t in model.targets)
    gnames = _global_param_names(model)
    prm = isempty(gnames) ? NamedTuple() :
        NamedTuple(g => Float64(first(vec(out.parameters[g]))) for g in gnames)
    return tgt, prm, _extract_latents(model, out)
end

function _extract_latents(model::UQModel, out)
    names = _neural_param_names(model)
    (isempty(names) || !hasproperty(out, :parameters)) && return NamedTuple()
    return NamedTuple(n => Float64.(vec(out.parameters[n])) for n in names)
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

# Aggregate per-observation latent / process parameters into the same nested
# (; mean, std, samples) layout used for globals, but with vector-valued mean/std
# and an n_obs × n sample matrix.
function _aggregate_latents(latent_preds::Vector{<:NamedTuple}, names, quantiles)
    (isempty(names) || isempty(latent_preds) || isempty(keys(first(latent_preds)))) &&
        return NamedTuple()
    means, stds, _, _, samples = _aggregate(latent_preds, names, quantiles)
    return NamedTuple(n => (; mean = means[n], std = stds[n], samples = samples[n]) for n in names)
end

# Backward-compatible wrapper used by older call sites.
function _predict_targets_and_params(model::UQModel, x, ps, st)
    tgt, prm, _ = _predict_full(model, x, ps, st)
    return tgt, prm
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
algorithm — [`MCDropout`](@ref), [`DeepEnsemble`](@ref), [`Bootstrap`](@ref)
or [`SGLD`](@ref).

# Methods

  * `estimate_uncertainty(m::MCDropout, model, data, train_output; …)` — uses the
    trained parameters/state in `train_output` (a [`TrainResults`](@ref)); no
    retraining. Errors if the network has no dropout, or if the model estimates
    global parameters.
  * `estimate_uncertainty(m::SGLD, model, data, train_output; …)` — continues
    from `train_output` with Langevin noise; no retraining from scratch. See
    [`SGLD`](@ref) for `scope` (`:all` / `:global` / `:nn`).
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
        method::MCDropout, model::UQModel, data, train_output::TrainResults;
        eval_data = data, quantiles::Tuple{<:Real, <:Real} = (0.025, 0.975),
    )
    # Guard 1: MC dropout cannot represent uncertainty of estimated global params.
    if !isempty(_global_param_names(model))
        throw(ArgumentError(
            "MC dropout is not applicable: the model estimates global parameter(s) " *
            "$(_global_param_names(model)). Global (physical) parameters are single point " *
            "estimates that dropout does not perturb, so MC dropout would report no " *
            "uncertainty for them. Use SGLD (scope = :global or :all), DeepEnsemble, or Bootstrap instead."
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
    latent_preds = Vector{NamedTuple}(undef, method.n_samples)
    for i in 1:method.n_samples
        out, st_mc = model(x, ps, st_mc)
        preds[i] = NamedTuple(t => vec(out[t]) for t in model.targets)
        latent_preds[i] = _extract_latents(model, out)
    end

    q = (Float64(quantiles[1]), Float64(quantiles[2]))
    means, stds, lowers, uppers, samples = _aggregate(preds, model.targets, q)
    latents = _aggregate_latents(latent_preds, _neural_param_names(model), q)
    meta = (; dropout_rates = rates)
    return UncertaintyResult(method, copy(model.targets), means, stds, lowers, uppers, samples, NamedTuple(), latents, method.n_samples, q, meta)
end

# --- Deep ensemble & bootstrap (shared runner) -------------------------------
function _ensemble_uncertainty(
        method::UncertaintyMethod, model::UQModel, data;
        seeds::Vector{Int}, use_bootstrap::Bool, eval_data, quantiles, verbose, train_kwargs,
    )
    x_eval = prepare_data(model, eval_data)[1]
    n_models = length(seeds)

    # Sensible defaults for UQ retraining; user train_kwargs win on conflict.
    base_kwargs = (; plotting = false, show_progress = false, save_training = false)
    merged_kwargs = merge(base_kwargs, train_kwargs)

    preds = Vector{NamedTuple}(undef, n_models)
    param_preds = Vector{NamedTuple}(undef, n_models)
    latent_preds = Vector{NamedTuple}(undef, n_models)
    for (i, seed) in enumerate(seeds)
        # A separate RNG per member keeps bootstrap draws reproducible and
        # independent of the training seed.
        member_data = use_bootstrap ? bootstrap_resample(data, Random.MersenneTwister(seed)) : data
        member_kwargs = merge(merged_kwargs, (; random_seed = seed))
        verbose && @info "Uncertainty: training ensemble member $i/$n_models (seed = $seed, bootstrap = $use_bootstrap)"
        res = train(model, member_data; member_kwargs...)
        res === nothing && throw(ErrorException("Training member $i returned nothing (data preparation failed)."))
        preds[i], param_preds[i], latent_preds[i] = _predict_full(model, x_eval, res.ps, LuxCore.testmode(res.st))
    end

    q = (Float64(quantiles[1]), Float64(quantiles[2]))
    means, stds, lowers, uppers, samples = _aggregate(preds, model.targets, q)
    params = _aggregate_params(param_preds, _global_param_names(model))
    latents = _aggregate_latents(latent_preds, _neural_param_names(model), q)
    meta = (; seeds = seeds, bootstrap = use_bootstrap)
    return UncertaintyResult(method, copy(model.targets), means, stds, lowers, uppers, samples, params, latents, n_models, q, meta)
end

function estimate_uncertainty(
        method::DeepEnsemble, model::UQModel, data;
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
        method::Bootstrap, model::UQModel, data;
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
    latent_preds = Vector{NamedTuple}(undef, method.n_models)
    for i in 1:method.n_models
        member_data = bootstrap_resample(data, Random.MersenneTwister(resample_seeds[i]))
        member_kwargs = merge(base_kwargs, (; random_seed = method.seed))
        verbose && @info "Uncertainty: training bootstrap member $i/$(method.n_models) (init seed = $(method.seed), resample seed = $(resample_seeds[i]))"
        res = train(model, member_data; member_kwargs...)
        res === nothing && throw(ErrorException("Training bootstrap member $i returned nothing (data preparation failed)."))
        preds[i], param_preds[i], latent_preds[i] = _predict_full(model, x_eval, res.ps, LuxCore.testmode(res.st))
    end

    q = (Float64(quantiles[1]), Float64(quantiles[2]))
    means, stds, lowers, uppers, samples = _aggregate(preds, model.targets, q)
    params = _aggregate_params(param_preds, _global_param_names(model))
    latents = _aggregate_latents(latent_preds, _neural_param_names(model), q)
    meta = (; init_seed = method.seed, resample_seeds = resample_seeds, bootstrap = true)
    return UncertaintyResult(method, copy(model.targets), means, stds, lowers, uppers, samples, params, latents, method.n_models, q, meta)
end

# --- Stochastic Gradient Langevin Dynamics -----------------------------------

function estimate_uncertainty(
        method::SGLD, model::UQModel, data, train_output::TrainResults;
        eval_data = data, quantiles::Tuple{<:Real, <:Real} = (0.025, 0.975),
        training_loss = :mse, extra_loss = nothing, agg = sum,
        verbose::Bool = true,
    )
    if method.scope == :global && isempty(_global_param_names(model))
        throw(ArgumentError(
            "SGLD scope = :global requires estimated global parameter(s), but this model " *
            "has none. Use scope = :nn or :all to sample neural-network weights (which " *
            "induce a posterior over latent/process parameters)."
        ))
    end

    rng = method.seed === nothing ? Random.default_rng() : Random.MersenneTwister(method.seed)
    cfg = TrainConfig(;
        batchsize = method.batchsize,
        plotting = false,
        show_progress = false,
        save_training = false,
        training_loss = training_loss,
        extra_loss = extra_loss,
        agg = agg,
    )
    ((x_train, forcings_train), y_train) = prepare_data(model, data)
    mask_train, empty_mask = valid_mask(y_train)
    empty_mask && throw(ArgumentError("SGLD: training data has no valid (non-NaN) targets."))
    loader = build_loader(x_train, forcings_train, y_train, mask_train, cfg)
    loss_fn = build_loss_fn(model, cfg)
    x_eval = prepare_data(model, eval_data)[1]

    ps = deepcopy(train_output.ps)
    st = LuxCore.trainmode(deepcopy(train_output.st))
    η = convert(Float32, method.lr)
    σ = sqrt(2 * η * convert(Float32, method.temperature))

    n_keep = method.n_samples
    preds = Vector{NamedTuple}(undef, n_keep)
    param_preds = Vector{NamedTuple}(undef, n_keep)
    latent_preds = Vector{NamedTuple}(undef, n_keep)
    kept = 0
    step = 0
    last_loss = NaN32

    verbose && @info "Uncertainty: SGLD sampling (scope = $(method.scope), burn-in = $(method.n_burnin), n_samples = $n_keep, thin = $(method.n_thin))"
    while kept < n_keep
        progressed = false
        for batch in loader
            progressed = true
            x_batch, y_batch = batch
            (x_col, y_col) = collect_dim_data(x_batch, y_batch, cfg)
            isemptybatch(y_col[2]) && continue

            ps, st, last_loss = _sgld_step(model, ps, st, (x_col, y_col), loss_fn, η, σ, rng, method.scope)
            step += 1

            if step > method.n_burnin && ((step - method.n_burnin) % method.n_thin == 0)
                kept += 1
                preds[kept], param_preds[kept], latent_preds[kept] =
                    _predict_full(model, x_eval, ps, LuxCore.testmode(st))
                kept >= n_keep && break
            end
        end
        progressed || throw(ErrorException("SGLD data loader is empty; cannot sample."))
    end

    q = (Float64(quantiles[1]), Float64(quantiles[2]))
    means, stds, lowers, uppers, samples = _aggregate(preds, model.targets, q)
    gnames = method.scope == :nn ? Symbol[] : _global_param_names(model)
    params = _aggregate_params(param_preds, gnames)
    latents = _aggregate_latents(latent_preds, _neural_param_names(model), q)
    meta = (;
        scope = method.scope,
        n_burnin = method.n_burnin,
        n_thin = method.n_thin,
        lr = method.lr,
        temperature = method.temperature,
        batchsize = method.batchsize,
        n_steps = step,
        last_loss = Float64(last_loss),
        seed = method.seed,
    )
    return UncertaintyResult(method, copy(model.targets), means, stds, lowers, uppers, samples, params, latents, n_keep, q, meta)
end

# One Langevin minibatch: ∇L via Zygote, then θ ← θ − η ∇L + σ ε on the selected keys.
function _sgld_step(model, ps, st, data, loss_fn, η, σ, rng, scope)
    (loss, st_new), back = Zygote.pullback(ps) do p
        l, st2, _ = loss_fn(model, p, st, data)
        (l, st2)
    end
    isfinite(loss) || throw(ErrorException(
        "SGLD produced a non-finite loss (loss = $loss). Try a smaller `lr` or `temperature`."
    ))
    gs = first(back((one(loss), nothing)))
    return _sgld_update(ps, gs, η, σ, rng, scope, model), st_new, loss
end

# Langevin update: θ ← θ − η ∇L + σ ε, applied to the selected top-level keys
# of `ps` (`:all` / `:global` / `:nn`).
function _sgld_update(ps, gs, η, σ, rng, scope::Symbol, model)
    gnames = Set(_global_param_names(model))
    keep = function (k)
        scope === :all && return true
        scope === :global && return k in gnames
        return k ∉ gnames   # :nn
    end
    return _sgld_update_keys(ps, gs, η, σ, rng, keep)
end

function _sgld_update_keys(ps::NamedTuple, gs, η, σ, rng, keep)
    return NamedTuple{keys(ps)}(
        map(keys(ps)) do k
            xk = getproperty(ps, k)
            keep(k) ? _langevin_leaf(xk, getproperty(gs, k), η, σ, rng) : xk
        end,
    )
end

function _sgld_update_keys(ps::ComponentArray, gs, η, σ, rng, keep)
    ps_new = copy(ps)
    for k in keys(ps)
        keep(k) || continue
        newv = _langevin_leaf(ps[k], gs[k], η, σ, rng)
        getproperty(ps_new, k) .= newv
    end
    return ps_new
end

_langevin_leaf(x, ::Nothing, η, σ, rng) = x

function _langevin_leaf(x::NamedTuple, g, η, σ, rng)
    return NamedTuple{keys(x)}(
        map(k -> _langevin_leaf(getproperty(x, k), getproperty(g, k), η, σ, rng), keys(x)),
    )
end

function _langevin_leaf(x::AbstractArray, g, η, σ, rng)
    noise = randn(rng, eltype(x), size(x))
    return @. x - η * g + σ * noise
end

function _langevin_leaf(x::Number, g, η, σ, rng)
    return x - η * g + σ * randn(rng, typeof(x))
end
