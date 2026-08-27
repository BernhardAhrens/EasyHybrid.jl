export ODEHybridModel, constructHybridODE

using Lux: LSTMCell

"""
    ODEHybridModel

Hybrid model that couples one or more LSTMs with a process-based step function.
The step runs *inside* the time loop so physical state can feed back into the LSTMs
(H2CM-style), unlike `constructHybridModel` + `Recurrence` which applies physics after
the full unroll.

Parameter kinds:

| Kind | Architecture | Runs | Example |
|------|-------------|------|---------|
| LSTM params | LSTMCell → Dense, optionally several named groups | per-timestep | `rb`, H2CM `alpha_*` |
| Static NN params | independent feedforward NNs | once per window | `C₀`, H2CM `sm_max` |
| Global scalars | learned, shared | constant | `Q10` |
| Fixed | frozen at default | constant | unused params |

LSTM input at each step is `[predictors_t; feedback_t]`. By default `feedback` is
the ODE state(s). Extra process outputs (`npp`, `fAPAR`, `rel_SM`, …) can be listed
in `feedback` so they are carried into the next LSTM call without being Euler-integrated.

The mechanistic model is a keyword function
`f(; state..., params..., forcing...) → (; dState..., observables...)`.
States are updated with Euler: `u ← u + du` (`Δt = 1`). Intra-step operator splitting
belongs inside `f`; return the net increment.
"""
struct ODEHybridModel{LC, P, SNN, F, PM <: AbstractHybridModel} <: LuxCore.AbstractLuxContainerLayer{(:lstm_cells, :projs)}
    lstm_cells::LC
    projs::P
    static_NNs::SNN
    static_predictors::NamedTuple
    lstm_predictors::NamedTuple
    lstm_param_names::NamedTuple
    lstm_feedback::NamedTuple
    mechanistic_model::F
    parameters::PM
    predictors::Union{Vector{Symbol}, NamedTuple}
    forcing::Vector{Symbol}
    targets::Vector{Symbol}
    all_lstm_param_names::Vector{Symbol}
    static_nn_param_names::Vector{Symbol}
    global_param_names::Vector{Symbol}
    fixed_param_names::Vector{Symbol}
    state_names::Vector{Symbol}
    deriv_names::Vector{Symbol}
    feedback_names::Vector{Symbol}
    scale_nn_outputs::Bool
    start_from_default::Bool
    config::NamedTuple
end

"""
    constructHybridODE(predictors, forcing, targets, mechanistic_model, parameters,
                       lstm_param_names, global_param_names; kwargs...)

ODE counterpart of `constructHybridModel`.

# Single LSTM (Vector predictors)
```julia
model = constructHybridODE(
    [:sw_pot, :dsw_pot], [:ta], [:reco],
    mOnePool_step,
    (rb = (3f0, 0f0, 13f0), Q10 = (2f0, 1f0, 4f0), C = (100f0, 10f0, 500f0)),
    [:rb], [:Q10];
    hidden_dims = 16, state = :C, deriv = :dC,
)
```

# Several LSTMs / several states (H2CM-shaped)
```julia
model = constructHybridODE(
    (water = [:rn, :prec], carbon = [:tair, :vpd, :CO2], rb = [:rn, :prec]),
    [:rn, :prec, :tair, :vpd, :CO2],
    [:ET, :gpp, :nee],
    h2cm_step,
    parameters,
    (water = [:alpha_r_soil, :alpha_r_gw, :alpha_snow],
     carbon = [:cue, :fAPAR],
     rb = [:rb, :alpha_Es]),
    [:Q10, :beta_snow];
    hidden_dims = 16,
    state = [:swe, :SM, :GW],
    deriv = [:dswe, :dSM, :dGW],
    feedback = (water = [:SM, :swe, :GW, :fAPAR],
                carbon = [:SM, :fAPAR, :npp],
                rb = [:npp, :fAPAR]),
    static_predictors = (; sm_max = static_cols, alpha_Ei = static_cols),
)
```

# Keyword arguments
- `hidden_dims=16`: LSTM hidden size (`Int` or `NamedTuple` per LSTM)
- `state=:C` / `deriv=:dC`: ODE state and derivative names (`Symbol` or `Vector{Symbol}`)
- `feedback=nothing`: extra carry into each LSTM. `nothing` → all states; `Vector` shared
  across LSTMs; `NamedTuple` per LSTM (same keys as `predictors`)
- `static_predictors=(;)`: per-param features for per-window NNs (initial conditions, `sm_max`, …)
- `static_hidden_layers=[8,8]`, `static_activation=tanh`
- `scale_nn_outputs=true`, `start_from_default=true`
"""
function constructHybridODE(
        predictors::Union{Vector{Symbol}, NamedTuple},
        forcing::Vector{Symbol},
        targets::Vector{Symbol},
        mechanistic_model,
        parameters,
        lstm_param_names,
        global_param_names::Vector{Symbol};
        hidden_dims::Union{Int, NamedTuple} = 16,
        n_state::Union{Int, Nothing} = nothing,
        state::Union{Symbol, Vector{Symbol}} = :C,
        deriv::Union{Symbol, Vector{Symbol}} = :dC,
        feedback = nothing,
        scale_nn_outputs::Bool = true,
        start_from_default::Bool = true,
        static_predictors::NamedTuple = (;),
        static_hidden_layers::Union{NamedTuple, Vector{Int}} = [8, 8],
        static_activation::Union{NamedTuple, Function} = tanh,
        kwargs...
    )

    if !isa(parameters, AbstractHybridModel)
        parameters = build_parameters(parameters, mechanistic_model)
    end

    lstm_predictors, lstm_pnames = _wrap_lstm_groups(predictors, lstm_param_names)
    state_names = _as_symbols(state)
    deriv_names = _as_symbols(deriv)
    length(state_names) == length(deriv_names) ||
        throw(ArgumentError("`state` and `deriv` must have the same length"))
    n_state === nothing || n_state == length(state_names) ||
        throw(ArgumentError("`n_state` must match `state` (got $n_state vs $(length(state_names)))"))

    lstm_feedback, feedback_names = _resolve_feedback(feedback, keys(lstm_predictors), state_names)

    all_names = pnames(parameters)
    all_lstm_param_names = unique(reduce(vcat, collect(values(lstm_pnames)); init = Symbol[]))
    static_nn_param_names = Symbol[k for k in keys(static_predictors)]
    all_neural = unique([all_lstm_param_names; static_nn_param_names])
    @assert all(n in all_names for n in all_neural) "all neural param names must be in parameters"

    fixed_param_names = [n for n in all_names if !(n in [all_neural; global_param_names])]

    lstm_cells = (;)
    projs = (;)
    for name in keys(lstm_predictors)
        n_out = length(lstm_pnames[name])
        n_out == 0 && continue
        n_in = length(lstm_predictors[name]) + length(lstm_feedback[name])
        n_in > 0 || throw(ArgumentError("LSTM :$name needs predictors or feedback"))
        hd = hidden_dims isa NamedTuple ? hidden_dims[name] : hidden_dims
        lstm_cells = merge(lstm_cells, NamedTuple{(name,)}((LSTMCell(n_in => hd),)))
        projs = merge(projs, NamedTuple{(name,)}((Dense(hd => n_out),)))
    end

    static_NNs = (;)
    for (nn_name, preds) in pairs(static_predictors)
        hl = static_hidden_layers isa NamedTuple ? static_hidden_layers[nn_name] : static_hidden_layers
        act = static_activation isa NamedTuple ? static_activation[nn_name] : static_activation
        nn = prepare_hidden_chain(hl, length(preds), 1; activation = act)
        static_NNs = merge(static_NNs, NamedTuple{(nn_name,)}((nn,)))
    end

    config = (;
        hidden_dims, state, deriv, feedback, scale_nn_outputs, start_from_default,
        static_hidden_layers, static_activation, kwargs...
    )

    return ODEHybridModel(
        lstm_cells, projs, static_NNs, static_predictors,
        lstm_predictors, lstm_pnames, lstm_feedback,
        mechanistic_model, parameters,
        predictors, forcing, targets,
        all_lstm_param_names, static_nn_param_names,
        global_param_names, fixed_param_names,
        state_names, deriv_names, feedback_names,
        scale_nn_outputs, start_from_default, config
    )
end

function constructHybridODE(;
        predictors, forcing, targets, mechanistic_model, parameters,
        lstm_param_names, global_param_names, kwargs...
    )
    return constructHybridODE(
        predictors, forcing, targets, mechanistic_model, parameters,
        lstm_param_names, global_param_names; kwargs...
    )
end

_as_symbols(x::Symbol) = Symbol[x]
_as_symbols(x::AbstractVector{Symbol}) = collect(x)

function _wrap_lstm_groups(predictors::Vector{Symbol}, lstm_param_names::Vector{Symbol})
    return (; lstm = predictors), (; lstm = lstm_param_names)
end

function _wrap_lstm_groups(predictors::NamedTuple, lstm_param_names::NamedTuple)
    issetequal(keys(predictors), keys(lstm_param_names)) ||
        throw(ArgumentError("predictors and lstm_param_names must have the same keys"))
    return predictors, lstm_param_names
end

function _resolve_feedback(feedback::Nothing, lstm_names, state_names)
    fb_each = NamedTuple{Tuple(lstm_names)}(ntuple(_ -> copy(state_names), length(lstm_names)))
    return fb_each, copy(state_names)
end

function _resolve_feedback(feedback::AbstractVector{Symbol}, lstm_names, state_names)
    fb = collect(feedback)
    fb_each = NamedTuple{Tuple(lstm_names)}(ntuple(_ -> copy(fb), length(lstm_names)))
    return fb_each, unique(vcat(state_names, fb))
end

function _resolve_feedback(feedback::NamedTuple, lstm_names, state_names)
    issetequal(keys(feedback), lstm_names) ||
        throw(ArgumentError("feedback keys must match LSTM names $(lstm_names)"))
    fb_each = map(collect, feedback)
    extra = reduce(vcat, collect.(values(fb_each)); init = Symbol[])
    return fb_each, unique(vcat(state_names, extra))
end

function LuxCore.initialparameters(rng::AbstractRNG, m::ODEHybridModel)
    nt = (;)
    if !isempty(m.lstm_cells)
        ps_cells = map(c -> first(LuxCore.setup(rng, c)), m.lstm_cells)
        ps_projs = map(p -> first(LuxCore.setup(rng, p)), m.projs)
        nt = (; lstm_cells = ps_cells, projs = ps_projs)
    end

    if !isempty(m.static_nn_param_names)
        snn_ps = map(nn -> first(LuxCore.setup(rng, nn)), m.static_NNs)
        nt = merge(nt, (; static_NNs = snn_ps))
    end

    if !isempty(m.global_param_names)
        for g in m.global_param_names
            val = m.start_from_default ? Float32(scale_single_param_minmax(g, m.parameters)) : rand(rng, Float32)
            nt = merge(nt, NamedTuple{(g,)}(([val],)))
        end
    end
    return nt
end

function LuxCore.initialstates(rng::AbstractRNG, m::ODEHybridModel)
    st_cells = map(c -> last(LuxCore.setup(rng, c)), m.lstm_cells)
    st_projs = map(p -> last(LuxCore.setup(rng, p)), m.projs)

    snn_st = (;)
    if !isempty(m.static_nn_param_names)
        snn_st = map(nn -> last(LuxCore.setup(rng, nn)), m.static_NNs)
    end

    fixed = (;)
    for f in m.fixed_param_names
        fixed = merge(fixed, NamedTuple{(f,)}(([Float32(default(m.parameters)[f])],)))
    end
    return (; lstm_cells = st_cells, projs = st_projs, static_NNs = snn_st, fixed = fixed)
end

function (m::ODEHybridModel)(ds_k::Union{KeyedArray, AbstractDimArray}, ps, st)
    T_len, B, ET = _ode_time_batch(ds_k, m)
    pred_cache = map(preds -> isempty(preds) ? zeros(ET, 0, T_len, B) : toArray(ds_k, preds), m.lstm_predictors)

    forc_3d = isempty(m.forcing) ? nothing : toArray(ds_k, m.forcing)

    static_kw, static_nn_states = _run_static_nns(m, ds_k, ps, st.static_NNs)
    phys_carry = _init_phys_carry(m, B, ET, ps, st, static_kw)

    global_names = [g for g in m.global_param_names if g ∉ m.state_names]
    global_kw = _named_from(global_names, g -> scale_single_param(g, ps[g], m.parameters))
    fixed_names = [f for f in m.fixed_param_names if f ∉ m.state_names]
    fixed_kw = _named_from(fixed_names, f -> st.fixed[f])
    static_non_state = [n for n in m.static_nn_param_names if n ∉ m.state_names]
    static_non_state_kw = _named_from(static_non_state, n -> static_kw[n])

    st_cells, st_projs = st.lstm_cells, st.projs

    nn_kw_1, lstm_carry, st_cells, st_projs = _run_lstms(
        m, pred_cache, 1, phys_carry, nothing, ps, st_cells, st_projs
    )
    result_1, phys_carry = _ode_process_step(
        m, nn_kw_1, phys_carry, forc_3d, 1, global_kw, fixed_kw, static_non_state_kw
    )

    result_names = collect(keys(result_1))
    result_trajs = NamedTuple{Tuple(result_names)}(Tuple(result_1[k] for k in result_names))
    nn_trajs = _named_from(m.all_lstm_param_names, n -> nn_kw_1[n])

    for t in 2:T_len
        nn_kw_t, lstm_carry, st_cells, st_projs = _run_lstms(
            m, pred_cache, t, phys_carry, lstm_carry, ps, st_cells, st_projs
        )
        result_t, phys_carry = _ode_process_step(
            m, nn_kw_t, phys_carry, forc_3d, t, global_kw, fixed_kw, static_non_state_kw
        )
        result_trajs = NamedTuple{Tuple(result_names)}(
            Tuple(vcat(result_trajs[k], result_t[k]) for k in result_names)
        )
        nn_trajs = NamedTuple{Tuple(m.all_lstm_param_names)}(
            Tuple(vcat(nn_trajs[n], nn_kw_t[n]) for n in m.all_lstm_param_names)
        )
    end

    output = merge(result_trajs, (; parameters = merge(nn_trajs, global_kw, fixed_kw, static_kw)))
    st_new = (; lstm_cells = st_cells, projs = st_projs, static_NNs = static_nn_states, fixed = st.fixed)
    return output, st_new
end

function _ode_time_batch(ds_k, m)
    for preds in values(m.lstm_predictors)
        isempty(preds) && continue
        a = toArray(ds_k, preds)
        return size(a, 2), size(a, 3), eltype(a)
    end
    if !isempty(m.forcing)
        a = toArray(ds_k, m.forcing)
        return size(a, 2), size(a, 3), eltype(a)
    end
    throw(ArgumentError("ODEHybridModel needs LSTM predictors or forcing to infer (time, batch)"))
end

function _run_static_nns(m, ds_k, ps, snn_states)
    static_kw = (;)
    static_nn_states = snn_states
    for (nn_name, nn) in pairs(m.static_NNs)
        nn_input = toArray(ds_k, collect(m.static_predictors[nn_name]))[:, 1, :]
        nn_out, st_nn = LuxCore.apply(nn, nn_input, ps.static_NNs[nn_name], static_nn_states[nn_name])
        static_nn_states = merge(static_nn_states, NamedTuple{(nn_name,)}((st_nn,)))
        nn_val = nn_out[1:1, :]
        if m.scale_nn_outputs
            nn_val = scale_single_param(nn_name, nn_val, m.parameters)
        end
        static_kw = merge(static_kw, NamedTuple{(nn_name,)}((nn_val,)))
    end
    return static_kw, static_nn_states
end

function _init_phys_carry(m, B, ET, ps, st, static_kw)
    carry = (;)
    for name in m.feedback_names
        val = if name in m.static_nn_param_names
            static_kw[name]
        elseif name in m.global_param_names
            scale_single_param(name, ps[name], m.parameters) .+ zeros(ET, 1, B)
        elseif name in m.fixed_param_names
            st.fixed[name] .+ zeros(ET, 1, B)
        else
            zeros(ET, 1, B)
        end
        carry = merge(carry, NamedTuple{(name,)}((val,)))
    end
    return carry
end

function _run_lstms(m, pred_cache, t, phys_carry, lstm_carry, ps, st_cells, st_projs)
    nn_kw = (;)
    new_carry = (;)
    new_st_cells = (;)
    new_st_projs = (;)
    for name in keys(m.lstm_cells)
        pred_t = pred_cache[name][:, t, :]
        fb_names = m.lstm_feedback[name]
        x = vcat(pred_t, ntuple(i -> phys_carry[fb_names[i]], length(fb_names))...)
        cell_in = lstm_carry isa NamedTuple ? (x, lstm_carry[name]) : x
        (h, c), st_c = Lux.apply(m.lstm_cells[name], cell_in, ps.lstm_cells[name], st_cells[name])
        raw, st_p = Lux.apply(m.projs[name], h, ps.projs[name], st_projs[name])
        pnames = m.lstm_param_names[name]
        scaled = ntuple(i -> begin
                slice = raw[i:i, :]
                m.scale_nn_outputs ? scale_single_param(pnames[i], slice, m.parameters) : slice
            end, length(pnames))
        nn_kw = merge(nn_kw, NamedTuple{Tuple(pnames)}(scaled))
        new_carry = merge(new_carry, NamedTuple{(name,)}((c,)))
        new_st_cells = merge(new_st_cells, NamedTuple{(name,)}((st_c,)))
        new_st_projs = merge(new_st_projs, NamedTuple{(name,)}((st_p,)))
    end
    return nn_kw, new_carry, new_st_cells, new_st_projs
end

function _ode_process_step(m, nn_kw, phys_carry, forc_3d, t, global_kw, fixed_kw, static_non_state_kw)
    if forc_3d !== nothing
        forc_t = forc_3d[:, t, :]
        forc_kw = (; zip(m.forcing, [forc_t[i:i, :] for i in 1:length(m.forcing)])...)
    else
        forc_kw = (;)
    end
    state_kw = _named_from(m.state_names, n -> phys_carry[n])
    result = m.mechanistic_model(; merge(nn_kw, global_kw, fixed_kw, static_non_state_kw, forc_kw, state_kw)...)

    new_states = NamedTuple{Tuple(m.state_names)}(
        Tuple(phys_carry[s] .+ result[d] for (s, d) in zip(m.state_names, m.deriv_names))
    )
    src = merge(phys_carry, nn_kw, result, new_states)
    new_carry = _named_from(m.feedback_names, n -> src[n])
    return result, new_carry
end

function _named_from(names::Vector{Symbol}, f)
    isempty(names) && return (;)
    return NamedTuple{Tuple(names)}(Tuple(f(n) for n in names))
end
