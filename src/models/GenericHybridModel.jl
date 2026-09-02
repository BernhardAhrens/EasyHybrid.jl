export HybridModel, ParameterContainer, constructHybridModel

"""
    ParameterContainer{NT <: NamedTuple, T}

A container for holding the parameter definitions of a model, including their default values, lower bounds, and upper bounds.

$(TYPEDFIELDS)
"""
mutable struct ParameterContainer{NT <: NamedTuple, T}
    "The raw parameter definitions. A `NamedTuple` where each entry is a tuple of `(default, lower, upper)` bounds for a parameter."
    values::NT

    "A `ComponentArray` matrix representation of the parameter bounds, organized for efficient access by name and bound type."
    table::T

    function ParameterContainer(values::NT) where {NT <: NamedTuple}
        table = build_parameter_matrix(values)
        return new{NT, typeof(table)}(values, table)
    end
end

"""
    HybridModel{T, P} <: LuxCore.AbstractLuxContainerLayer{(:NNs,)}

A unified hybrid model struct that handles both single and multi neural network architectures.
It combines predictive neural networks (`NNs`) with a `mechanistic_model` to form a differentiable hybrid model.

$(TYPEDFIELDS)
"""
struct HybridModel{T, P, MM, NP, GP, FP, KW, EX, TG, PC, SN} <: LuxCore.AbstractLuxContainerLayer{(:NNs,)}
    "Neural network(s) used to predict parameters. Can be a single `Chain` or a `NamedTuple` of `Chain`s."
    NNs::T

    "Predictor variables for the neural networks. Can be a `Vector{Symbol}` or a `NamedTuple`."
    predictors::P

    "Forcing variables passed directly to the mechanistic model."
    forcing::Vector{Symbol}

    "Target variables the model will output/be trained against."
    targets::Vector{Symbol}

    "The core process-based or mechanistic model function."
    mechanistic_model::MM

    "Base parameters of the model (encapsulated in a `ParameterContainer`)."
    parameters::PC

    "Names of the parameters predicted by the neural network(s)."
    neural_param_names::Vector{Symbol}

    "Names of the globally optimized (constant) parameters."
    global_param_names::Vector{Symbol}

    "Names of the fixed (non-optimized) parameters."
    fixed_param_names::Vector{Symbol}

    "Whether to scale neural network outputs to the parameter bounds."
    scale_nn_outputs::Bool

    "Whether to initialize global parameters from their default values."
    start_from_default::Bool

    "Configuration named tuple capturing the hyperparameters used for initialization."
    config::NamedTuple
end

# Ordered, de-duplicated tuple of names across the given symbol groups. Used to
# encode parameter-name groups in the type domain so the forward pass can build
# concrete (type-stable) `NamedTuple`s instead of ones keyed by runtime vectors.
function _merge_names(groups...)
    out = Symbol[]
    for grp in groups, n in grp
        n in out || push!(out, n)
    end
    return Tuple(out)
end

# Outer constructor: lift the parameter-name groups and the mechanistic-model
# kwarg plan into type parameters. The reflection (`methods(f)`) that used to run
# on *every* forward pass now runs once, here, at construction time.
#
# - `NP`/`GP`/`FP`: neural / global / fixed parameter names.
# - `KW`: names to forward to the mechanistic model, or `nothing` when it slurps
#         `kwargs...` (forward everything).
# - `EX`: parameter names the mechanistic model does not consume, surfaced at the
#         top level of the output (e.g. loss-only parameters).
# - `TG`: target names, used to keep the training-loss assembly type-stable.
# - `PC`: concrete `ParameterContainer` type, so `values` (the bounds NamedTuple)
#         is visible to the compiler and parameter scaling stays type-stable.
# - `SN`: `scale_nn_outputs` as a type-domain `Bool`, so the scale/no-scale
#         branch of `_run_nn` is resolved at compile time (no Union).
function HybridModel(
        NNs, predictors, forcing, targets, mechanistic_model, parameters,
        neural_param_names, global_param_names, fixed_param_names,
        scale_nn_outputs, start_from_default, config,
    )
    NP = Tuple(neural_param_names)
    GP = Tuple(global_param_names)
    FP = Tuple(fixed_param_names)

    all_kwarg_names = _merge_names(forcing, NP, GP, FP)
    accepted = _accepted_kwarg_names(mechanistic_model, all_kwarg_names)
    KW = accepted === nothing ? nothing : accepted
    param_names = _merge_names(NP, GP, FP)
    EX = accepted === nothing ? () : Tuple(k for k in param_names if !(k in accepted))
    TG = Tuple(targets)

    return HybridModel{typeof(NNs), typeof(predictors), typeof(mechanistic_model), NP, GP, FP, KW, EX, TG, typeof(parameters), scale_nn_outputs}(
        NNs, predictors, forcing, targets, mechanistic_model, parameters,
        neural_param_names, global_param_names, fixed_param_names,
        scale_nn_outputs, start_from_default, config,
    )
end

"""
    constructHybridModel(predictors::Vector{Symbol}, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names; kwargs...)

Construct a `HybridModel` with a single neural network architecture predicting all `neural_param_names` from the `predictors`.

# Arguments:
- `predictors::Vector{Symbol}`: Variables used as inputs to the neural network.
- `forcing`: Variables passed directly to the mechanistic model.
- `targets`: The target variables to predict.
- `mechanistic_model`: A function implementing the process-based model.
- `parameters`: A parameter container defining defaults, lowers, and uppers.
- `neural_param_names`: Names of the parameters to be predicted by the neural network.
- `global_param_names`: Names of the parameters to be globally optimized.
- `kwargs`: Additional configuration like `hidden_layers`, `activation`, `scale_nn_outputs`, etc.
"""
function constructHybridModel(
        predictors::Vector{Symbol},
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names,
        global_param_names;
        hidden_layers::Union{Vector{Int}, Chain} = [32, 32],
        activation = tanh,
        scale_nn_outputs = false,
        input_batchnorm = false,
        start_from_default = true,
        kwargs...
    )

    if !isa(parameters, ParameterContainer)
        parameters = ParameterContainer(parameters)
    end

    all_names = pnames(parameters)
    @assert all(n in all_names for n in neural_param_names) "neural_param_names ⊆ param_names"

    # if empty predictors do not construct NN
    if length(predictors) > 0 && length(neural_param_names) > 0

        in_dim = length(predictors)
        out_dim = length(neural_param_names)

        NN = prepare_hidden_chain(
            hidden_layers, in_dim, out_dim;
            activation = activation,
            input_batchnorm = input_batchnorm
        )
    else
        NN = Chain()
    end

    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]

    # capture the configuration used for construction
    config = (;
        hidden_layers,
        activation,
        scale_nn_outputs,
        input_batchnorm,
        start_from_default,
        kwargs...,
    )

    return HybridModel(
        NN,
        predictors,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names,
        global_param_names,
        fixed_param_names,
        scale_nn_outputs,
        start_from_default,
        config
    )
end

"""
    constructHybridModel(predictors::NamedTuple, forcing, targets, mechanistic_model, parameters, global_param_names; kwargs...)

Construct a `HybridModel` with multiple neural network architectures. A separate neural network is built for each key in the `predictors` NamedTuple.

# Arguments:
- `predictors::NamedTuple`: A NamedTuple where keys are network names, and values are vectors of predictor variables for that network.
- `forcing`: Variables passed directly to the mechanistic model.
- `targets`: The target variables to predict.
- `mechanistic_model`: A function implementing the process-based model.
- `parameters`: A parameter container defining defaults, lowers, and uppers.
- `global_param_names`: Names of the parameters to be globally optimized.
- `kwargs`: Additional configuration. `hidden_layers` and `activation` can also be NamedTuples to configure each network independently.
"""
function constructHybridModel(
        predictors::NamedTuple,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        global_param_names;
        hidden_layers::Union{Vector{Int}, Chain, NamedTuple} = [32, 32],
        activation::Union{Function, NamedTuple} = tanh,
        scale_nn_outputs = false,
        input_batchnorm = false,
        start_from_default = true,
        kwargs...
    )

    if !isa(parameters, ParameterContainer)
        parameters = ParameterContainer(parameters)
    end

    all_names = pnames(parameters)
    neural_param_names = collect(keys(predictors))
    # Create neural networks based on predictors
    NNs = NamedTuple()
    for (nn_name, preds) in pairs(predictors)
        # Create a simple NN for each predictor set
        in_dim = length(preds)
        out_dim = 1
        if hidden_layers isa NamedTuple
            if activation isa NamedTuple
                nn = prepare_hidden_chain(
                    hidden_layers[nn_name], in_dim, out_dim;
                    activation = activation[nn_name],
                    input_batchnorm = input_batchnorm
                )
            else
                nn = prepare_hidden_chain(
                    hidden_layers[nn_name], in_dim, out_dim;
                    activation = activation,
                    input_batchnorm = input_batchnorm
                )
            end
        else
            nn = prepare_hidden_chain(
                hidden_layers, in_dim, out_dim;
                activation = activation,
                input_batchnorm = input_batchnorm
            )
        end
        NNs = merge(NNs, NamedTuple{(nn_name,), Tuple{typeof(nn)}}((nn,)))
    end

    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]

    # capture the configuration used for construction
    config = (;
        hidden_layers,
        activation,
        scale_nn_outputs,
        input_batchnorm,
        start_from_default,
        kwargs...,
    )

    return HybridModel(
        NNs,
        predictors,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names,
        global_param_names,
        fixed_param_names,
        scale_nn_outputs,
        start_from_default,
        config
    )
end

function constructHybridModel(
        ; predictors,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names = nothing,
        global_param_names,
        kwargs...
    )
    if predictors isa Vector{Symbol}
        @assert neural_param_names !== nothing "Provide neural_param_names for Vector predictors"
        return constructHybridModel(
            predictors, forcing, targets, mechanistic_model, parameters,
            neural_param_names, global_param_names; kwargs...
        )
    elseif predictors isa NamedTuple
        return constructHybridModel(
            predictors, forcing, targets, mechanistic_model, parameters,
            global_param_names; kwargs...
        )
    else
        throw(ArgumentError("predictors must be Vector{Symbol} or NamedTuple, got $(typeof(predictors))"))
    end
end

"""
    _init_nn_params(rng, m::HybridModel{<:Any, <:NamedTuple})

Initialize parameters for a multi-neural network architecture.
Returns a `NamedTuple` containing the initialized parameters for each sub-network.
"""
function _init_nn_params(rng::AbstractRNG, m::HybridModel{<:Any, <:NamedTuple})
    return map(nn -> LuxCore.setup(rng, nn)[1], m.NNs)
end

"""
    _init_nn_params(rng, m::HybridModel{<:Any, <:Vector})

Initialize parameters for a single-neural network architecture.
Returns a `NamedTuple` containing a single `ps` field with the network's parameters.
"""
function _init_nn_params(rng::AbstractRNG, m::HybridModel{<:Any, <:Vector})
    ps_nn, _ = LuxCore.setup(rng, m.NNs)
    return (; ps = ps_nn)
end

# Initial parameters for HybridModel
function LuxCore.initialparameters(rng::AbstractRNG, m::HybridModel)
    nt = _init_nn_params(rng, m)

    # Then append each global parameter as a 1-vector of Float32
    if !isempty(m.global_param_names)
        if m.start_from_default
            for g in m.global_param_names
                default_val = scale_single_param_minmax(g, m.parameters)
                nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
            end
        else
            for g in m.global_param_names
                random_val = rand(rng, Float32)
                nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([random_val],)))
            end
        end
    end

    return nt
end

"""
    _init_nn_states(rng, m::HybridModel{<:Any, <:NamedTuple})

Initialize states for a multi-neural network architecture.
Returns a `NamedTuple` containing the initialized states for each sub-network.
"""
function _init_nn_states(rng::AbstractRNG, m::HybridModel{<:Any, <:NamedTuple})
    return map(nn -> LuxCore.setup(rng, nn)[2], m.NNs)
end

"""
    _init_nn_states(rng, m::HybridModel{<:Any, <:Vector})

Initialize states for a single-neural network architecture.
Returns a `NamedTuple` containing a single `st_nn` field with the network's states.
"""
function _init_nn_states(rng::AbstractRNG, m::HybridModel{<:Any, <:Vector})
    _, st_nn = LuxCore.setup(rng, m.NNs)
    return (; st_nn = st_nn)
end

# Initial states for HybridModel
function LuxCore.initialstates(rng::AbstractRNG, m::HybridModel)
    nn_states_nt = _init_nn_states(rng, m)
    nt = (;)

    # Then append each fixed parameter as a 1-vector of Float32
    if !isempty(m.fixed_param_names)
        for f in m.fixed_param_names
            default_val = default(m.parameters)[f]
            nt = merge(nt, NamedTuple{(f,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end

    return merge(nn_states_nt, (; fixed = nt))
end

"""
    _run_nn(m::HybridModel{<:Any, <:NamedTuple}, ds_k::Tuple, ps, st)

Execute the forward pass for a multi-neural network architecture.
Applies each sub-network to its specific predictors, and applies scaling to the outputs if required.
Returns scaled parameter values, updated states, and raw network outputs.
"""
@inline _scale_nn_outputs(::HybridModel{T, P, MM, NP, GP, FP, KW, EX, TG, PC, SN}) where {T, P, MM, NP, GP, FP, KW, EX, TG, PC, SN} = SN

function _run_nn(m::HybridModel{<:Any, <:NamedTuple, <:Any, NP}, ds_k::Tuple, ps, st) where {NP}
    nn_names = keys(m.NNs)
    applied = map(nn_names) do nn_name
        LuxCore.apply(m.NNs[nn_name], ds_k[1][nn_name], ps[nn_name], st[nn_name])
    end
    nn_outputs = NamedTuple{nn_names}(map(first, applied))
    nn_states = NamedTuple{nn_names}(map(last, applied))

    scaled_vals = map(nn_names, NP) do nn_name, param_name
        val = eachslice(nn_outputs[nn_name]; dims = 1)[1]
        return _scale_nn_outputs(m) ? scale_single_param(Val(param_name), val, m.parameters) : val
    end
    scaled_nn_params = NamedTuple{NP}(scaled_vals)

    return scaled_nn_params, nn_states, (; nn_outputs = nn_outputs)
end

"""
    _run_nn(m::HybridModel{<:Any, <:Vector}, ds_k::Tuple, ps, st)

Execute the forward pass for a single-neural network architecture.
Applies the neural network to the given predictors, slices the output for multiple predicted parameters, and scales them if required.
Returns scaled parameter values, updated states, and raw network outputs.
"""
function _run_nn(m::HybridModel{<:Any, <:Vector, <:Any, NP}, ds_k::Tuple, ps, st) where {NP}
    if !isempty(NP)
        nn_out, st_nn = LuxCore.apply(m.NNs, ds_k[1], ps.ps, st.st_nn)
        slices = eachslice(nn_out, dims = 1)
        nn_cols = ntuple(i -> slices[i], Val(length(NP)))
        if _scale_nn_outputs(m)
            scaled_nn_vals = map((name, col) -> scale_single_param(Val(name), col, m.parameters), NP, nn_cols)
        else
            scaled_nn_vals = nn_cols
        end
        scaled_nn_params = NamedTuple{NP}(scaled_nn_vals)
    else
        scaled_nn_params = NamedTuple()
        st_nn = st.st_nn
    end
    return scaled_nn_params, (; st_nn = st_nn), (;)
end

"""
    (m::HybridModel)(ds_k::Tuple, ps, st)

Forward pass of the hybrid model.
Evaluates the neural networks to predict parameters, merges them with scaled global parameters and fixed parameters, and executes the mechanistic model.
Returns a tuple `(out, st_new)`.
"""
function (m::HybridModel{T, P, MM, NP, GP, FP, KW, EX})(ds_k::Tuple, ps, st) where {T, P, MM, NP, GP, FP, KW, EX}
    parameters = m.parameters

    # 1) Scale global parameters. Keys come from the `GP` type parameter, so the
    #    resulting `NamedTuple` is concrete (type-stable).
    global_params = NamedTuple{GP}(map(g -> scale_single_param(Val(g), ps[g], parameters), GP))

    # 2) Run neural network(s)
    scaled_nn_params, st_new_nns, out_extra = _run_nn(m, ds_k, ps, st)

    # 3) Pick fixed parameters (keys from the `FP` type parameter).
    fixed_params = NamedTuple{FP}(map(f -> st.fixed[f], FP))

    # 4) merge all parameters
    all_params = merge(scaled_nn_params, global_params, fixed_params)

    # 5) unpack forcing data
    forcing_data = ds_k[2]
    all_kwargs = merge(forcing_data, all_params)

    # 6) Apply mechanistic model. Only forward the kwargs it actually declares, so
    #    "loss-only" parameters (e.g. a learned noise scale used only in the loss)
    #    can be defined without the mechanistic model having to accept them. The set
    #    of accepted kwargs (`KW`) is precomputed once at construction, so no
    #    reflection runs here on the hot path.
    y_pred = _apply_mechanistic(m.mechanistic_model, Val(KW), all_kwargs)

    # Parameters the mechanistic model does not consume (`EX`, e.g. loss-only ones
    # such as a learned noise scale) are surfaced at the top level so they can be
    # monitored and plotted, in addition to always being available under `parameters`.
    extra_params = NamedTuple{EX}(all_params)
    out = (; y_pred..., extra_params..., parameters = all_params, out_extra...)
    st_new = (; st_new_nns..., fixed = st.fixed)

    return out, st_new
end

# Forward only the mechanistic model's declared kwargs. `Val{nothing}` means the
# model slurps `kwargs...`, so everything is forwarded; otherwise `KW` is the
# precomputed tuple of accepted names, keeping the call type-stable.
_apply_mechanistic(f, ::Val{nothing}, all_kwargs::NamedTuple) = f(; all_kwargs...)
function _apply_mechanistic(f, ::Val{KW}, all_kwargs::NamedTuple) where {KW}
    return f(; NamedTuple{KW}(all_kwargs)...)
end

function (m::HybridModel)(ds_k, ps, st)
    # Forward pass fallback when ds_k is not explicitly typed as Tuple
    return m(Tuple(ds_k), ps, st)
end

# Returns the tuple of `available` names accepted by `f`, or `nothing` to signal
# "pass everything" (the model slurps `kwargs...`, or has no introspectable kwargs).
# Called once per model at construction time (see the `HybridModel` outer
# constructor), not on the forward-pass hot path.
function _accepted_kwarg_names(f, available::Tuple)
    names = Symbol[]
    for mth in methods(f)
        for d in Base.kwarg_decl(mth)
            endswith(string(d), "...") && return nothing  # slurps kwargs → keep all
            push!(names, d)
        end
    end
    isempty(names) && return nothing
    return Tuple(k for k in available if k in names)
end

function (m::HybridModel)(df::DataFrame, ps, st)
    @warn "Only makes sense in test mode, not training!"

    # Process numeric or missing-containing columns
    for col in names(df)
        what_type = eltype(df[!, col])
        if what_type <: Union{Missing, Real} || what_type <: Real
            df[!, col] = Float32.(coalesce.(df[!, col], NaN))
        end
    end

    all_data = to_keyedArray(df)
    x, _ = prepare_data(m, all_data)
    out, _ = m(x, ps, LuxCore.testmode(st))
    dfnew = copy(df)
    n_samples = x[1] isa NamedTuple ? size(first(values(x[1])), 2) : size(x[1], 2)
    for k in keys(out)
        if length(out[k]) == n_samples
            dfnew[!, String(k) * "_pred"] = out[k]
        end
    end
    return dfnew
end
