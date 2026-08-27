# # ODE-LSTM Hybrid Model with EasyHybrid.jl
#
# Couple an LSTM with an ODE: the LSTM predicts time-varying `rb`, a one-pool
# carbon step evolves `C` via Euler (`C ← C + dC`), and `C` is fed back into the
# LSTM at every timestep. Same process family as `example_synthetic_lstm.jl`,
# but with an internal state.
#
# ## 1. Load Packages

using Pkg
Pkg.activate("docs")
Pkg.develop(path = pwd())
Pkg.instantiate()

using EasyHybrid
using AxisKeys
using DimensionalData

# ## 2. Data

df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc");
df = df[1:1000, :];
first(df, 5);

# ## 3. Process step
#
# Keyword function like any EasyHybrid mechanistic model, plus a state `C` and
# its derivative `dC`. Intra-step physics (operator splitting, clamps, …) lives
# here; the framework only does `C ← C + dC`.

function mOnePool_step(; C, rb, Q10, ta, tref = 15.0f0)
    reco = rb .* C .* Q10 .^ (0.1f0 .* (ta .- tref))
    dC = .- reco
    return (; dC, reco, Q10, rb, C)
end

# ## 4. Parameters and construction
#
# `(default, lower, upper)`. `C` is the initial condition: leave it out of
# `global_param_names` to freeze it, or list it to learn a scalar `C₀`.

parameters = (
    rb = (3.0f0, 0.0f0, 13.0f0),
    Q10 = (2.0f0, 1.0f0, 4.0f0),
    C = (100.0f0, 10.0f0, 500.0f0),
)

forcing = [:ta]
predictors = [:sw_pot, :dsw_pot]
target = [:reco]

hode = constructHybridODE(
    predictors, forcing, target, mOnePool_step, parameters,
    [:rb], [:Q10];
    hidden_dims = 16,
    state = :C,
    deriv = :dC,
    scale_nn_outputs = true,
)

# ## 5. Forward + train
#
# Windowing is the same as the LSTM tutorial (`DataConfig.sequence_length`).

pref_array_type = :DimArray
input_window = 10
output_window = 1
output_shift = 1

sdf = split_data(
    df, hode;
    sequence_kwargs = (;
        input_window = input_window,
        output_window = output_window,
        output_shift = output_shift,
        lead_time = 0,
    ),
    array_type = pref_array_type,
);
(x_train, y_train), (x_val, y_val) = sdf;

ps, st = Lux.setup(Random.default_rng(), hode);
train_dl = EasyHybrid.DataLoader((x_train, y_train); batchsize = 32);
x_first = first(train_dl)[1]
frun = hode(x_first, ps, st);
frun[1].reco
frun[1].C
frun[1].dC

out_ode = train(
    hode, df;
    train_cfg = EasyHybrid.TrainConfig(
        nepochs = 2,
        batchsize = 128,
        opt = RMSProp(0.01),
        training_loss = :nseLoss,
        loss_types = [:nse],
        plotting = false,
        show_progress = false,
    ),
    data_cfg = EasyHybrid.DataConfig(
        sequence_length = input_window,
        sequence_output_window = output_window,
        sequence_output_shift = output_shift,
        sequence_lead_time = 0,
        array_type = pref_array_type,
    ),
);

out_ode.val_obs_pred

# ## 6. Static NN for `C₀`
#
# Per-window initial condition from features (same idea as H2CM `sm_max`).
# Do not list `C` in `global_param_names`.

hode_static = constructHybridODE(
    predictors, forcing, target, mOnePool_step, parameters,
    [:rb], [:Q10];
    hidden_dims = 16,
    state = :C,
    deriv = :dC,
    scale_nn_outputs = true,
    static_predictors = (; C = [:sw_pot, :dsw_pot]),
    static_hidden_layers = (; C = [8, 8]),
)

ps2, st2 = Lux.setup(Random.default_rng(), hode_static);
frun2 = hode_static(x_first, ps2, st2);
frun2[1].reco

out_ode_static = train(
    hode_static, df;
    train_cfg = EasyHybrid.TrainConfig(
        nepochs = 2,
        batchsize = 128,
        opt = RMSProp(0.01),
        training_loss = :nseLoss,
        loss_types = [:nse],
        plotting = false,
        show_progress = false,
        model_name = "mOnePool_ode_lstm_static_C0",
    ),
    data_cfg = EasyHybrid.DataConfig(
        sequence_length = input_window,
        sequence_output_window = output_window,
        sequence_output_shift = output_shift,
        sequence_lead_time = 0,
        array_type = pref_array_type,
    ),
);

out_ode.best_loss
out_ode_static.best_loss
