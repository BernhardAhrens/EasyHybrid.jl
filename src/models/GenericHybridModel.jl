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

function HybridModel(
        NNs, predictors, forcing, targets, mechanistic_model, parameters,
        neural_param_names, global_param_names, fixed_param_names,
        scale_nn_outputs, start_from_default, config,
    )
    NP, GP, FP = Tuple(neural_param_names), Tuple(global_param_names), Tuple(fixed_param_names)
    accepted = _accepted_kwarg_names(
        mechanistic_model,
        Tuple(unique([forcing; neural_param_names; global_param_names; fixed_param_names])),
    )
    EX = accepted === nothing ? () : Tuple(k for k in (NP..., GP..., FP...) if !(k in accepted))
    return HybridModel{typeof(NNs), typeof(predictors), typeof(mechanistic_model), NP, GP, FP, accepted, EX, Tuple(targets), typeof(parameters), scale_nn_outputs}(
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
function _run_nn(m::HybridModel{T, <:NamedTuple, MM, NP, GP, FP, KW, EX, TG, PC, SN}, ds_k::Tuple, ps, st) where {T, MM, NP, GP, FP, KW, EX, TG, PC, SN}
    nn_names = keys(m.NNs)
    applied = map(nn_names) do nn_name
        LuxCore.apply(m.NNs[nn_name], ds_k[1][nn_name], ps[nn_name], st[nn_name])
    end
    nn_outputs = NamedTuple{nn_names}(map(first, applied))
    nn_states = NamedTuple{nn_names}(map(last, applied))
    scaled_nn_params = _scale_named(Val{NP}(), Val{SN}(), nn_outputs, m.parameters)
    return scaled_nn_params, nn_states, (; nn_outputs = nn_outputs)
end

# Vector forward: bind NN / global / fixed params and call the mechanistic model.
# `Val(true)` also returns `all_params` so the public call can pack `parameters`.
@generated function _hybrid_vector(
        m::HybridModel{T, P, MM, NP, GP, FP, KW, EX, TG, PC, SN},
        ds_k::Tuple, ps, st, ::Val{need_params}
    ) where {T, P, MM, NP, GP, FP, KW, EX, TG, PC, SN, need_params}
    stmts = Expr[]
    all_names = (NP..., GP..., FP...)

    if NP === ()
        push!(stmts, :(st_nn = st.st_nn))
    else
        push!(stmts, :((nn_out, st_nn) = LuxCore.apply(m.NNs, ds_k[1], ps.ps, st.st_nn)))
        push!(stmts, :(slices = eachslice(nn_out, dims = 1)))
    end

    for (i, n) in enumerate(NP)
        rhs = SN ?
            :(scale_single_param(Val($(QuoteNode(n))), slices[$i], m.parameters)) :
            :(slices[$i])
        push!(stmts, Expr(:(=), n, rhs))
    end
    for n in GP
        push!(stmts, Expr(:(=), n, :(scale_single_param(Val($(QuoteNode(n))), getproperty(ps, $(QuoteNode(n))), m.parameters))))
    end
    for n in FP
        push!(stmts, Expr(:(=), n, :(getproperty(st.fixed, $(QuoteNode(n))))))
    end

    # Zygote drops unused generated-function args from the pullback.
    NP === () && GP === () && push!(stmts, :(ChainRulesCore.ignore_derivatives(ps)))
    NP === () && KW === () && push!(stmts, :(ChainRulesCore.ignore_derivatives(ds_k)))
    !SN && GP === () && push!(stmts, :(ChainRulesCore.ignore_derivatives(m.parameters)))

    if need_params || KW === nothing
        push!(stmts, :(all_params = (; $(all_names...))))
    end

    if KW === nothing
        push!(stmts, :(y_pred = m.mechanistic_model(; merge(ds_k[2], all_params)...)))
    else
        args = Expr[]
        for n in KW
            src = n in all_names ? n : :(getfield(ds_k[2], $(QuoteNode(n))))
            push!(args, Expr(:kw, n, src))
        end
        push!(stmts, :(y_pred = m.mechanistic_model(; $(args...))))
    end

    push!(stmts, need_params ? :(return y_pred, all_params, st_nn) : :(return y_pred, st_nn))
    return Expr(:block, stmts...)
end

function (m::HybridModel{T, <:Vector, MM, NP, GP, FP, KW, EX, TG, PC, SN})(ds_k::Tuple, ps, st) where {T, MM, NP, GP, FP, KW, EX, TG, PC, SN}
    y_pred, all_params, st_nn = _hybrid_vector(m, ds_k, ps, st, Val(true))
    extra = NamedTuple{EX}(all_params)
    out = (; y_pred..., extra..., parameters = all_params)
    return out, ChainRulesCore.ignore_derivatives((; st_nn, fixed = st.fixed))
end

"""
    (m::HybridModel)(ds_k::Tuple, ps, st)

Forward pass of the hybrid model.
Evaluates the neural networks to predict parameters, merges them with scaled global parameters and fixed parameters, and executes the mechanistic model.
Returns a tuple `(out, st_new)`.
"""
function (m::HybridModel{T, P, MM, NP, GP, FP, KW, EX, TG, PC, SN})(ds_k::Tuple, ps, st) where {T, P, MM, NP, GP, FP, KW, EX, TG, PC, SN}
    scaled_nn_params, st_new_nns, out_extra = _run_nn(m, ds_k, ps, st)
    all_params = _all_params(Val{NP}(), Val{GP}(), Val{FP}(), scaled_nn_params, ps, st, m.parameters)
    y_pred = _call_mech(m.mechanistic_model, ds_k[2], all_params, Val{KW}())
    extra = NamedTuple{EX}(all_params)
    out = (; y_pred..., extra..., parameters = all_params, out_extra...)
    return out, ChainRulesCore.ignore_derivatives((; st_new_nns..., fixed = st.fixed))
end

@generated function _scale_named(::Val{NP}, ::Val{SN}, nn_outputs, parameters) where {NP, SN}
    scaled = ntuple(length(NP)) do i
        n = NP[i]
        slice = :(eachslice(getfield(nn_outputs, $(QuoteNode(n))); dims = 1)[1])
        SN ? :(scale_single_param(Val($(QuoteNode(n))), $slice, parameters)) : slice
    end
    return quote
        $(SN ? nothing : :(ChainRulesCore.ignore_derivatives(parameters)))
        NamedTuple{NP}(($(scaled...),))
    end
end

@generated function _all_params(::Val{NP}, ::Val{GP}, ::Val{FP}, nn, ps, st, parameters) where {NP, GP, FP}
    parts = Expr[]
    for n in NP
        push!(parts, Expr(:kw, n, :(getfield(nn, $(QuoteNode(n))))))
    end
    for n in GP
        push!(parts, Expr(:kw, n, :(scale_single_param(Val($(QuoteNode(n))), getproperty(ps, $(QuoteNode(n))), parameters))))
    end
    for n in FP
        push!(parts, Expr(:kw, n, :(getproperty(getproperty(st, :fixed), $(QuoteNode(n))))))
    end
    unused = Expr[]
    NP === () && push!(unused, :(ChainRulesCore.ignore_derivatives(nn)))
    GP === () && push!(unused, :(ChainRulesCore.ignore_derivatives(ps)))
    FP === () && push!(unused, :(ChainRulesCore.ignore_derivatives(st)))
    GP === () && push!(unused, :(ChainRulesCore.ignore_derivatives(parameters)))
    return quote
        $(unused...)
        (; $(parts...))
    end
end

_call_mech(f, forcing, params, ::Val{nothing}) = f(; merge(forcing, params)...)

@generated function _call_mech(f, forcing::NamedTuple{FK}, params::NamedTuple{PK}, ::Val{KW}) where {FK, PK, KW}
    args = Expr[]
    for n in KW
        src = n in PK ? :(getfield(params, $(QuoteNode(n)))) :
            n in FK ? :(getfield(forcing, $(QuoteNode(n)))) :
            error("mechanistic kwarg $(n) not found in forcing or parameters")
        push!(args, Expr(:kw, n, src))
    end
    return :(f(; $(args...)))
end

function (m::HybridModel)(ds_k, ps, st)
    # Forward pass fallback when ds_k is not explicitly typed as Tuple
    return m(Tuple(ds_k), ps, st)
end

# Returns the tuple of `all_kwargs` names accepted by `f`, or `nothing` to signal
# "pass everything" (the model slurps `kwargs...`, or has no introspectable kwargs).
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
