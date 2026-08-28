using EasyHybrid
using Test
using Random
using Statistics
using DataFrames

# -----------------------------------------------------------------------------
# Shared fixtures
# -----------------------------------------------------------------------------
function _uq_data(n; seed = 1)
    rng = MersenneTwister(seed)
    ta      = Float32.(rand(rng, n) .* 30 .- 5)
    sw_pot  = Float32.(rand(rng, n))
    dsw_pot = Float32.(rand(rng, n) .* 2 .- 1)
    rb   = 1.5f0 .+ 2.0f0 .* sw_pot .+ 0.5f0 .* (dsw_pot .^ 2)
    reco = rb .* 2.0f0 .^ (0.1f0 .* (ta .- 15.0f0))
    reco .+= 0.05f0 .* Float32.(randn(rng, n)) .* reco
    return DataFrame(ta = ta, sw_pot = sw_pot, dsw_pot = dsw_pot, reco = reco)
end

_RbQ10(; ta, Q10, rb, tref = 15.0f0) = (; reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref)), Q10, rb)
const _PARAMS = (rb = (3.0f0, 0.0f0, 13.0f0), Q10 = (2.0f0, 1.0f0, 4.0f0))
const _FORCING = [:ta]
const _PRED = [:sw_pot, :dsw_pot]
const _TARGET = [:reco]

_model_global() = constructHybridModel(
    _PRED, _FORCING, _TARGET, _RbQ10, _PARAMS, [:rb], [:Q10];
    hidden_layers = [8, 8], activation = relu, scale_nn_outputs = true, input_batchnorm = true,
)
_model_dropout() = constructHybridModel(
    _PRED, _FORCING, _TARGET, _RbQ10, _PARAMS, [:rb, :Q10], Symbol[];
    hidden_layers = Chain(Dense(8, 8, relu), Dropout(0.3), Dense(8, 8, relu), Dropout(0.3)),
    activation = relu, scale_nn_outputs = true, input_batchnorm = false,
)
_model_nodropout() = constructHybridModel(
    _PRED, _FORCING, _TARGET, _RbQ10, _PARAMS, [:rb, :Q10], Symbol[];
    hidden_layers = [8, 8], activation = relu, scale_nn_outputs = true, input_batchnorm = false,
)

const _TRAIN_KW = (; nepochs = 8, batchsize = 64, opt = RMSProp(0.01),
    show_progress = false, plotting = false, save_training = false)

@testset "Uncertainty: dropout detection" begin
    @test EasyHybrid.dropout_rates(_model_dropout()) == [0.3, 0.3]
    @test isempty(EasyHybrid.dropout_rates(_model_nodropout()))
    @test isempty(EasyHybrid.dropout_rates(_model_global()))
end

@testset "Uncertainty: bootstrap_resample" begin
    df = _uq_data(50)
    rng = MersenneTwister(7)
    dfb = EasyHybrid.bootstrap_resample(df, rng)
    @test size(dfb) == size(df)
    @test names(dfb) == names(df)

    ka = to_keyedArray(Float32.(df))
    kab = EasyHybrid.bootstrap_resample(ka, MersenneTwister(7))
    @test size(kab) == size(ka)
    # variable axis is preserved, observation axis is resampled to 1:n
    @test Symbol.(axiskeys(kab, :variable)) == Symbol.(axiskeys(ka, :variable))

    # sequence / 3-D data must be rejected
    @test_throws ArgumentError EasyHybrid.bootstrap_resample((ka, ka), MersenneTwister(1))
end

@testset "Uncertainty: MC dropout" begin
    df = _uq_data(300)
    m = _model_dropout()
    res = train(m, df; _TRAIN_KW...)
    u = estimate_uncertainty(MCDropout(n_samples = 20), m, df, res)

    @test u isa UncertaintyResult
    @test u.n == 20
    @test haskey(u.mean, :reco) && length(u.mean.reco) == nrow(df)
    @test length(u.std.reco) == nrow(df)
    @test size(u.samples.reco) == (nrow(df), 20)
    @test mean(u.std.reco) > 0                      # dropout is stochastic
    @test all(u.lower.reco .<= u.mean.reco .<= u.upper.reco)
    @test isempty(u.params)                          # no global params
    @test u.metadata.dropout_rates == [0.3, 0.3]
end

@testset "Uncertainty: MC dropout guards" begin
    df = _uq_data(200)
    # (a) global parameters estimated -> not applicable
    mg = _model_global()
    resg = train(mg, df; _TRAIN_KW...)
    @test_throws ArgumentError estimate_uncertainty(MCDropout(n_samples = 5), mg, df, resg)

    # (b) no dropout in the network -> not applicable
    mnd = _model_nodropout()
    resnd = train(mnd, df; _TRAIN_KW...)
    @test_throws ArgumentError estimate_uncertainty(MCDropout(n_samples = 5), mnd, df, resnd)
end

@testset "Uncertainty: deep ensemble" begin
    df = _uq_data(300)
    m = _model_global()
    u = estimate_uncertainty(DeepEnsemble(n_models = 3), m, df; verbose = false, _TRAIN_KW...)

    @test u.n == 3
    @test length(u.mean.reco) == nrow(df)
    @test mean(u.std.reco) >= 0
    @test haskey(u.params, :Q10)
    @test u.params.Q10.std >= 0
    @test length(u.params.Q10.samples) == 3
    @test u.metadata.bootstrap == false
    @test length(u.metadata.seeds) == 3
end

@testset "Uncertainty: pure NN model (SingleNNModel)" begin
    df = _uq_data(300)
    nn = constructNNModel(_PRED, [:reco];
        hidden_layers = Chain(Dense(8, 8, relu), Dropout(0.3), Dense(8, 8, relu), Dropout(0.3)),
        activation = relu, scale_nn_outputs = true, input_batchnorm = false)
    @test nn isa EasyHybrid.SingleNNModel
    @test EasyHybrid.dropout_rates(nn) == [0.3, 0.3]

    res = train(nn, df; _TRAIN_KW...)
    u = estimate_uncertainty(MCDropout(n_samples = 20), nn, df, res)
    @test length(u.mean.reco) == nrow(df)
    @test mean(u.std.reco) > 0            # dropout stochasticity
    @test isempty(u.params)               # pure NN has no global params

    # deep ensemble also works on a pure NN model
    ude = estimate_uncertainty(DeepEnsemble(n_models = 3), nn, df; verbose = false, _TRAIN_KW...)
    @test ude.n == 3
    @test isempty(ude.params)

    # a pure NN without dropout is rejected by MC dropout
    nn_nd = constructNNModel(_PRED, [:reco]; hidden_layers = [8, 8], activation = relu)
    res_nd = train(nn_nd, df; _TRAIN_KW...)
    @test_throws ArgumentError estimate_uncertainty(MCDropout(n_samples = 5), nn_nd, df, res_nd)
end

@testset "Uncertainty: bootstrap and combined ensemble" begin
    df = _uq_data(120)
    u_boot = estimate_uncertainty(Bootstrap(n_models = 3), _model_global(), df; verbose = false, _TRAIN_KW...)
    @test u_boot.n == 3
    @test u_boot.metadata.bootstrap == true
    @test haskey(u_boot.params, :Q10)

    u_combo = estimate_uncertainty(DeepEnsemble(n_models = 3, bootstrap = true), _model_global(), df; verbose = false, _TRAIN_KW...)
    @test u_combo.metadata.bootstrap == true
    @test u_combo.n == 3

    # explicit seeds honored / validated
    @test_throws ArgumentError estimate_uncertainty(
        DeepEnsemble(n_models = 3, seeds = [1, 2]), _model_global(), df; verbose = false, _TRAIN_KW...)
end
