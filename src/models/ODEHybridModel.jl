export ODEHybridModel, constructHybridODE

using Lux: LSTMCell

"""
    ODEHybridModel

Hybrid model that couples an LSTM with a process-based ODE step function,
optionally augmented by static (per-window) neural networks for parameters
like the initial ODE state.

**Two kinds of NN-predicted parameters:**

| Kind | Architecture | Runs | Example |
|------|-------------|------|---------|
| Dynamic params | LSTM → Dense, or feedforward Chain | per-timestep, inside the loop | `rb` |
| Static NN params | same groups, all inputs time-constant | once per window, before the loop | `C` (initial carbon pool) |

Dynamic NNs receive `predictors` at each step.
Put `state` names (or other process carry) in `predictors` to feed them into the NN;
names not found in the data are taken from the ODE loop.
Columns that do not vary over time in a window are treated as static: a group
whose predictors are all static is run once (e.g. `sm_max`). A group whose neural
params include a `state` name is also run once and used as C₀.

`predictors` / `neural_param_names` may be a `NamedTuple` of groups (several LSTMs);
`state` / `deriv` may be a `Vector` of names.

# Fields
- `lstm_cells`, `projs`: LSTM + projection per group; feedforward groups are a `Chain` (no proj)
- `mechanistic_model`: user function `f(; C, rb, Q10, ...) → (; dC, reco, ...)`
- `parameters`: `ParameterContainer` with bounds for scaling
- `predictors`: LSTM input feature names
- `forcing`, `targets`: same role as in `SingleNNHybridModel`
- `neural_param_names`: params predicted by each dynamic (or static) NN group
- `global_param_names`, `fixed_param_names`: non-neural params
"""
struct ODEHybridModel{LC, P, F, PM <: AbstractHybridModel} <: LuxCore.AbstractLuxContainerLayer{(:lstm_cells, :projs)}
    lstm_cells::LC
    projs::P
    lstm_predictors::NamedTuple
    neural_param_names::NamedTuple
    mechanistic_model::F
    parameters::PM
    predictors::Union{Vector{Symbol}, NamedTuple}
    forcing::Vector{Symbol}
    targets::Vector{Symbol}
    all_neural_param_names::Vector{Symbol}
    global_param_names::Vector{Symbol}
    fixed_param_names::Vector{Symbol}
    scale_nn_outputs::Bool
    start_from_default::Bool
    state_names::Vector{Symbol}
    deriv_names::Vector{Symbol}
    n_state::Int
    config::NamedTuple
end

"""
    constructHybridODE(predictors, forcing, targets, mechanistic_model, parameters,
                       neural_param_names, global_param_names; kwargs...)

Construct an `ODEHybridModel` — the ODE counterpart of `constructHybridModel`.

The user writes the mechanistic model as a plain Julia function with keyword arguments,
exactly like `RbQ10`, but returning a derivative field (e.g. `dC`) in addition to
observable outputs.

# Example — LSTM only (all neural params are time-varying)
```julia
model = constructHybridODE(
    [:sw_pot, :dsw_pot, :C],      # predictors (include :C to feed state into the LSTM)
    [:SW_IN, :TA],                # forcing
    [:NEE],                       # targets
    mOnePool_step,
    (rb = (3f0, 0f0, 13f0), Q10 = (2f0, 1f0, 4f0), C = (100f0, 10f0, 500f0)),
    [:rb],                        # neural_param_names
    [:Q10, :C];                   # global_param_names (C₀ trainable scalar)
    hidden_dims = 16,
    state = :C, deriv = :dC,
)
```

# Example — LSTM + static NN for initial C
```julia
model = constructHybridODE(
    (rb = [:sw_pot, :dsw_pot, :C], C = [:soil_moisture, :clay_fraction]),
    [:SW_IN, :TA],
    [:NEE],
    mOnePool_step,
    (rb = (3f0, 0f0, 13f0), Q10 = (2f0, 1f0, 4f0), C = (100f0, 10f0, 500f0)),
    (rb = [:rb], C = [:C]),
    [:Q10];
    hidden_dims = 16,
    recurrent = (C = false,),
    state = :C, deriv = :dC,
)
```

# Keyword Arguments
- `hidden_dims::Int = 16`: LSTM hidden size, or feedforward hidden width (`Int` or `NamedTuple`)
- `recurrent=true`: LSTM. `false` → feedforward inside the loop. NamedTuple per NN; omitted → LSTM
- `n_state::Int = 1`: dimensionality of ODE state
- `state::Symbol = :C`: name of the ODE state variable (`Symbol` or `Vector{Symbol}`).
  If this name appears in `parameters`, the initial condition is taken from there
  (trainable if in `global_param_names`, fixed otherwise).
  If a static NN group predicts it, that output is used as C₀.
- `deriv::Symbol = :dC`: name of the derivative in the step function output
- `scale_nn_outputs::Bool = true`: apply sigmoid scaling to NN outputs
- `start_from_default::Bool = true`: initialize global params at their default values
"""
function constructHybridODE(
        predictors::Union{Vector{Symbol}, NamedTuple},
        forcing::Vector{Symbol},
        targets::Vector{Symbol},
        mechanistic_model,
        parameters,
        neural_param_names,
        global_param_names::Vector{Symbol};
        hidden_dims::Union{Int, NamedTuple} = 16,
        recurrent::Union{Bool, NamedTuple} = true,
        n_state::Int = 1,
        state::Union{Symbol, Vector{Symbol}} = :C,
        deriv::Union{Symbol, Vector{Symbol}} = :dC,
        scale_nn_outputs::Bool = true,
        start_from_default::Bool = true,
        kwargs...
    )

    if !isa(parameters, AbstractHybridModel)
        parameters = build_parameters(parameters, mechanistic_model)
    end

    lstm_predictors, lstm_pnames = _wrap_lstm_groups(predictors, neural_param_names)
    state_names = state isa Symbol ? Symbol[state] : collect(state)
    deriv_names = deriv isa Symbol ? Symbol[deriv] : collect(deriv)
    length(state_names) == length(deriv_names) ||
        throw(ArgumentError("`state` and `deriv` must have the same length"))
    n_state = length(state_names) == 1 ? n_state : 1

    all_names = pnames(parameters)
    all_neural_param_names = unique(reduce(vcat, collect(values(lstm_pnames)); init = Symbol[]))
    @assert all(n in all_names for n in all_neural_param_names) "all neural param names must be in parameters"

    fixed_param_names = [n for n in all_names if !(n in [all_neural_param_names..., global_param_names...])]

    # ---- LSTM + projection ----
    lstm_cells = (;)
    projs = (;)
    for name in keys(lstm_predictors)
        n_out = length(lstm_pnames[name])
        n_out == 0 && continue
        n_in = _input_rows(lstm_predictors[name], state_names, n_state)
        n_in > 0 || throw(ArgumentError("NN :$name needs predictors"))
        hd = hidden_dims isa NamedTuple ? hidden_dims[name] : hidden_dims
        if _is_recurrent(recurrent, name)
            lstm_cells = merge(lstm_cells, NamedTuple{(name,)}((LSTMCell(n_in => hd),)))
            projs = merge(projs, NamedTuple{(name,)}((Dense(hd => n_out),)))
        else
            nn = prepare_hidden_chain([hd, hd], n_in, n_out)
            lstm_cells = merge(lstm_cells, NamedTuple{(name,)}((nn,)))
        end
    end

    config = (;
        hidden_dims, recurrent, n_state, state, deriv,
        scale_nn_outputs, start_from_default, kwargs...,
    )

    return ODEHybridModel(
        lstm_cells, projs,
        lstm_predictors, lstm_pnames,
        mechanistic_model, parameters,
        predictors, forcing, targets,
        all_neural_param_names,
        global_param_names, fixed_param_names,
        scale_nn_outputs, start_from_default,
        state_names, deriv_names, n_state, config
    )
end

# Keyword-argument overload
function constructHybridODE(;
        predictors, forcing, targets, mechanistic_model, parameters,
        neural_param_names, global_param_names, kwargs...
    )
    return constructHybridODE(
        predictors, forcing, targets, mechanistic_model, parameters,
        neural_param_names, global_param_names; kwargs...
    )
end

function _wrap_lstm_groups(predictors::Vector{Symbol}, neural_param_names::Vector{Symbol})
    return (; lstm = predictors), (; lstm = neural_param_names)
end
_wrap_lstm_groups(predictors::NamedTuple, neural_param_names::NamedTuple) = predictors, neural_param_names

_is_recurrent(recurrent::Bool, _) = recurrent
_is_recurrent(recurrent::NamedTuple, name) = get(recurrent, name, true)

_input_rows(preds, state_names, n_state) =
    sum(n -> (length(state_names) == 1 && n === state_names[1]) ? n_state : 1, preds; init = 0)

_nrow(m::ODEHybridModel, name) =
    (length(m.state_names) == 1 && name === m.state_names[1]) ? m.n_state : 1

# ───────────────────────────────────────────────────────────────────────────
# Lux parameter / state initialization

function LuxCore.initialparameters(rng::AbstractRNG, m::ODEHybridModel)
    ps_cells = map(c -> first(LuxCore.setup(rng, c)), m.lstm_cells)
    ps_projs = map(p -> first(LuxCore.setup(rng, p)), m.projs)
    nt = (; lstm_cells = ps_cells, projs = ps_projs)

    # Global scalars
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

function LuxCore.initialstates(rng::AbstractRNG, m::ODEHybridModel)
    st_cells = map(c -> last(LuxCore.setup(rng, c)), m.lstm_cells)
    st_projs = map(p -> last(LuxCore.setup(rng, p)), m.projs)

    # Fixed params
    fixed = (;)
    if !isempty(m.fixed_param_names)
        for f in m.fixed_param_names
            default_val = default(m.parameters)[f]
            fixed = merge(fixed, NamedTuple{(f,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end

    return (; lstm_cells = st_cells, projs = st_projs, fixed = fixed)
end

# ───────────────────────────────────────────────────────────────────────────
# Forward pass — explicit time loop (SpiralClassifier pattern)

function (m::ODEHybridModel)(ds_k::Union{KeyedArray, AbstractDimArray}, ps, st)
    data_vars, T_len, B, ET, pred_cache, forc_3d, carry_names, static_groups, dyn_groups = ChainRulesCore.ignore_derivatives() do
        dv = _data_vars(ds_k)
        T_len, B, ET = _ode_time_batch(ds_k, m, dv)
        pred_cache = _pred_cache(ds_k, m, dv)
        forc_3d = isempty(m.forcing) ? nothing : toArray(ds_k, m.forcing)
        static_vars = Set{Symbol}(p for p in keys(pred_cache) if _is_static_series(pred_cache[p]))
        static_groups = [
            n for n in keys(m.lstm_cells) if
                any(p -> p in m.state_names, m.neural_param_names[n]) ||
                _is_static_group(m.lstm_predictors[n], static_vars)
        ]
        dyn_groups = [n for n in keys(m.lstm_cells) if n ∉ static_groups]
        return dv, T_len, B, ET, pred_cache, forc_3d, _carry_names(m, dv), static_groups, dyn_groups
    end

    # ── static groups: run once per window, before the time loop ──
    st_cells, st_projs = st.lstm_cells, st.projs
    static_kw, _, st_cells, st_projs = _run_lstms(
        m, pred_cache, 1, (;), nothing, ps, st_cells, st_projs, static_groups
    )

    # ── initialize ODE state ──
    # Priority: static NN > global param > fixed param > zeros
    phys_carry = (;)
    for name in carry_names
        nrow = _nrow(m, name)
        val = if haskey(static_kw, name) && name in m.state_names
            static_kw[name]
        elseif name in m.global_param_names
            scale_single_param(name, ps[name], m.parameters) .+ zeros(ET, nrow, B)
        elseif name in m.fixed_param_names
            st.fixed[name] .+ zeros(ET, nrow, B)
        else
            zeros(ET, nrow, B)
        end
        phys_carry = merge(phys_carry, NamedTuple{(name,)}((val,)))
    end

    sns = m.state_names
    # ── global params (excluding state — state is managed by the ODE loop) ──
    global_names = [g for g in m.global_param_names if g ∉ sns]
    global_kw = isempty(global_names) ? (;) :
        (; zip(global_names, Tuple(scale_single_param(g, ps[g], m.parameters) for g in global_names))...)

    # ── fixed params (excluding state) ──
    fixed_names = [f for f in m.fixed_param_names if f ∉ sns]
    fixed_kw = isempty(fixed_names) ? (;) : (; zip(fixed_names, Tuple(st.fixed[f] for f in fixed_names))...)

    # ── static NN params that are NOT the ODE state (constant through the loop) ──
    static_non_state_names = [n for n in keys(static_kw) if n ∉ sns]
    static_non_state_kw = isempty(static_non_state_names) ? (;) :
        (; zip(static_non_state_names, [static_kw[n] for n in static_non_state_names])...)

    dyn_param_names = unique(reduce(vcat, (collect(m.neural_param_names[n]) for n in dyn_groups); init = Symbol[]))

    # ── first timestep (no carry) ──
    nn_kw_1, lstm_carry, st_cells, st_projs = _run_lstms(
        m, pred_cache, 1, phys_carry, nothing, ps, st_cells, st_projs, dyn_groups
    )
    result_1, phys_carry = _ode_inner_step(
        m, nn_kw_1, phys_carry, forc_3d, 1, global_kw, fixed_kw, static_non_state_kw
    )

    # Accumulate *all* mechanistic outputs (not just targets) via vcat (mutation-free for AD)
    result_names = collect(keys(result_1))
    result_trajs = NamedTuple{Tuple(result_names)}(Tuple(result_1[k] for k in result_names))
    nn_trajs = NamedTuple{Tuple(dyn_param_names)}(Tuple(nn_kw_1[n] for n in dyn_param_names))

    # ── remaining timesteps ──
    for t in 2:T_len
        nn_kw_t, lstm_carry, st_cells, st_projs = _run_lstms(
            m, pred_cache, t, phys_carry, lstm_carry, ps, st_cells, st_projs, dyn_groups
        )
        result_t, phys_carry = _ode_inner_step(
            m, nn_kw_t, phys_carry, forc_3d, t, global_kw, fixed_kw, static_non_state_kw
        )
        result_trajs = NamedTuple{Tuple(result_names)}(
            Tuple(vcat(result_trajs[k], result_t[k]) for k in result_names)
        )
        nn_trajs = NamedTuple{Tuple(dyn_param_names)}(
            Tuple(vcat(nn_trajs[n], nn_kw_t[n]) for n in dyn_param_names)
        )
    end

    # ── output as plain NamedTuple (time subsetting handled by compute_loss) ──
    output = merge(result_trajs, (; parameters = merge(nn_trajs, global_kw, fixed_kw, static_kw)))
    st_new = (; lstm_cells = st_cells, projs = st_projs, fixed = st.fixed)
    return output, st_new
end

function _data_vars(ds_k)
    return Set{Symbol}(Symbol.(axiskeys(ds_k, 1)))
end

function _is_static_series(a)
    T = size(a, 2)
    T <= 1 && return true
    return all(a .== view(a, :, 1:1, :))
end

_is_static_group(preds, static_vars) = !isempty(preds) && all(p -> p in static_vars, preds)

function _carry_names(m, data_vars)
    extra = [p for preds in values(m.lstm_predictors) for p in preds if p in m.state_names || p ∉ data_vars]
    return unique(vcat(m.state_names, extra))
end

function _pred_cache(ds_k, m, data_vars)
    cache = (;)
    for preds in values(m.lstm_predictors)
        for p in preds
            (p in data_vars && p ∉ m.state_names && !haskey(cache, p)) || continue
            cache = merge(cache, NamedTuple{(p,)}((toArray(ds_k, [p]),)))
        end
    end
    return cache
end

function _ode_time_batch(ds_k, m, data_vars = _data_vars(ds_k))
    for preds in values(m.lstm_predictors)
        data_p = [p for p in preds if p in data_vars && p ∉ m.state_names]
        isempty(data_p) && continue
        a = toArray(ds_k, data_p)
        return size(a, 2), size(a, 3), eltype(a)
    end
    a = toArray(ds_k, m.forcing)
    return size(a, 2), size(a, 3), eltype(a)
end

function _run_lstms(m, pred_cache, t, phys_carry, lstm_carry, ps, st_cells, st_projs, group_names)
    nn_kw = (;)
    new_carry = (;)
    new_st_cells = st_cells
    new_st_projs = st_projs
    for name in group_names
        preds = m.lstm_predictors[name]
        x = vcat(
            ntuple(
                i -> begin
                    p = preds[i]
                    haskey(phys_carry, p) ? phys_carry[p] : pred_cache[p][:, t, :]
                end, length(preds)
            )...
        )
        if haskey(m.projs, name)
            cell_in = lstm_carry isa NamedTuple && haskey(lstm_carry, name) ? (x, lstm_carry[name]) : x
            (h, c), st_c = Lux.apply(m.lstm_cells[name], cell_in, ps.lstm_cells[name], st_cells[name])
            raw, st_p = Lux.apply(m.projs[name], h, ps.projs[name], st_projs[name])
            new_carry = merge(new_carry, NamedTuple{(name,)}((c,)))
            new_st_projs = merge(new_st_projs, NamedTuple{(name,)}((st_p,)))
        else
            raw, st_c = Lux.apply(m.lstm_cells[name], x, ps.lstm_cells[name], st_cells[name])
        end
        pnames = m.neural_param_names[name]
        n_nn = length(pnames)
        nn_scaled = if m.scale_nn_outputs
            ntuple(i -> scale_single_param(pnames[i], raw[i:i, :], m.parameters), n_nn)
        else
            ntuple(i -> raw[i:i, :], n_nn)
        end
        nn_kw = merge(nn_kw, (; zip(pnames, nn_scaled)...))
        new_st_cells = merge(new_st_cells, NamedTuple{(name,)}((st_c,)))
    end
    return nn_kw, new_carry, new_st_cells, new_st_projs
end

"""
Inner step: merge NN params with static/global/fixed, call mechanistic model, Euler-update state.
"""
function _ode_inner_step(m, nn_kw, phys_carry, forc_3d, t, global_kw, fixed_kw, static_non_state_kw)
    if forc_3d !== nothing
        forc_t = forc_3d[:, t, :]
        forc_kw = (; zip(m.forcing, [forc_t[i:i, :] for i in 1:length(m.forcing)])...)
    else
        forc_kw = (;)
    end
    state_kw = (; zip(m.state_names, Tuple(phys_carry[n] for n in m.state_names))...)
    result = m.mechanistic_model(; merge(nn_kw, global_kw, fixed_kw, static_non_state_kw, forc_kw, state_kw)...)

    new_states = NamedTuple{Tuple(m.state_names)}(
        Tuple(phys_carry[s] .+ result[d] for (s, d) in zip(m.state_names, m.deriv_names))
    )
    src = merge(phys_carry, nn_kw, result, new_states)
    new_carry = (; zip(keys(phys_carry), Tuple(src[n] for n in keys(phys_carry)))...)
    return result, new_carry
end
