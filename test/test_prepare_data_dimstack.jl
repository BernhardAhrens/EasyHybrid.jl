using EasyHybrid
using Test
using Random
using Statistics
using DataFrames
using DimensionalData

# RbQ10 physical model (same as other tests)
function RbQ10_dd(; ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end

function RbQ10_multi(; ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    reco2 = 1.5f0 .* reco
    return (; reco, reco2, Q10, rb)
end

const RbQ10_DD_PARAMS = (
    rb = (3.0f0, 0.0f0, 13.0f0),
    Q10 = (2.0f0, 1.0f0, 4.0f0),
)

function make_synth_df_dd(n::Int = 128; seed::Int = 7)
    rng = MersenneTwister(seed)
    ta = Float32.(10 .+ 10 .* randn(rng, n))
    sw_pot = Float32.(abs.(50 .+ 20 .* randn(rng, n)))
    dsw_pot = Float32.(vcat(0.0, diff(sw_pot)))
    reco = Float32.(3.0 .* (2.0 .^ (0.1 .* (ta .- 15.0))) .+ 0.1 .* randn(rng, n))
    return DataFrame(; ta, sw_pot, dsw_pot, reco)
end

# Build a DimStack whose named layers mirror the DataFrame columns (over an :obs dim).
function df_to_dimstack(df::DataFrame)
    obs = Dim{:obs}(1:nrow(df))
    layers = (; (Symbol(c) => df[!, c] for c in names(df))...)
    return DimStack(layers, (obs,))
end

# Helper: compare two prepared ((X, forcings), targets) tuples for numerical equality.
# NaN-aware equality: retained NaN targets must compare equal (NaN != NaN under `==`).
function prepared_equal(a, b)
    (Xa, Fa), Ta = a
    (Xb, Fb), Tb = b
    Xa isa NamedTuple || (isequal(Array(Xa), Array(Xb)) || return false)
    keys(Fa) == keys(Fb) || return false
    all(isequal(Array(Fa[k]), Array(Fb[k])) for k in keys(Fa)) || return false
    keys(Ta) == keys(Tb) || return false
    all(isequal(Array(Ta[k]), Array(Tb[k])) for k in keys(Ta)) || return false
    return true
end

@testset "prepare_data: DimStack ≡ DataFrame (unified cleaning path)" begin
    forcing = [:ta]
    predictors = [:sw_pot, :dsw_pot]
    target = [:reco]
    model = constructHybridModel(
        predictors, forcing, target, RbQ10_dd,
        RbQ10_DD_PARAMS, [:rb], [:Q10],
    )

    @testset "clean data: identical output" begin
        df = make_synth_df_dd(96)
        st = df_to_dimstack(df)

        prep_df = prepare_data(model, df; array_type = :DimArray)
        prep_st = prepare_data(model, st; array_type = :DimArray)
        @test prepared_equal(prep_df, prep_st)

        # DimStack also works through the default :KeyedArray output.
        prep_st_ka = prepare_data(model, st; array_type = :KeyedArray)
        (Xk, _), _ = prep_st_ka
        (Xd, _), _ = prep_st
        @test Array(Xk) == Array(Xd)
    end

    @testset "missing handling: identical row dropping" begin
        df = make_synth_df_dd(96)
        allowmissing!(df)
        df.sw_pot[3] = missing     # predictor missing -> sample dropped
        df.dsw_pot[10] = missing   # predictor missing -> sample dropped
        df.reco[5] = missing       # sole target missing -> sample dropped (needs ≥1 present target)
        st = df_to_dimstack(df)

        prep_df = prepare_data(model, df; array_type = :DimArray, drop_missing_rows = true)
        prep_st = prepare_data(model, st; array_type = :DimArray, drop_missing_rows = true)

        (Xd, _), Td = prep_df
        (Xs, _), Ts = prep_st
        @test size(Xd) == size(Xs)
        @test size(Xd, 2) == 93                       # 96 - 3 distinct dropped samples
        @test prepared_equal(prep_df, prep_st)
        # DataFrame and DimStack agree that no NaN target survives for a single-target model.
        @test !any(isnan, Array(Td.reco))
        @test !any(isnan, Array(Ts.reco))
    end

    @testset "multi-target: NaN target survives when another target is present" begin
        # Two targets so a sample with one present target is kept (target NaN retained).
        model2 = constructHybridModel(
            predictors, forcing, [:reco, :reco2], RbQ10_multi,
            RbQ10_DD_PARAMS, [:rb], [:Q10],
        )
        df = make_synth_df_dd(60)
        df.reco2 = df.reco .* 1.5f0
        allowmissing!(df)
        df.reco[5] = missing       # one target missing, other present -> sample kept
        st = df_to_dimstack(df)

        prep_df = prepare_data(model2, df; array_type = :DimArray, drop_missing_rows = true)
        prep_st = prepare_data(model2, st; array_type = :DimArray, drop_missing_rows = true)
        (_, _), Td = prep_df
        (_, _), Ts = prep_st
        @test prepared_equal(prep_df, prep_st)
        @test any(isnan, Array(Td.reco))              # NaN target retained
        @test any(isnan, Array(Ts.reco))
    end

    @testset "drop_missing_rows = false keeps everything" begin
        df = make_synth_df_dd(50)
        allowmissing!(df)
        df.sw_pot[4] = missing
        st = df_to_dimstack(df)

        prep_df = prepare_data(model, df; array_type = :DimArray, drop_missing_rows = false)
        prep_st = prepare_data(model, st; array_type = :DimArray, drop_missing_rows = false)
        (Xd, _), _ = prep_df
        (Xs, _), _ = prep_st
        @test size(Xd, 2) == 50
        @test prepared_equal(prep_df, prep_st)
    end

    @testset "train directly from a DimStack" begin
        df = make_synth_df_dd(128)
        st = df_to_dimstack(df)
        out = train(
            model, st, ();
            nepochs = 1, batchsize = 16,
            plotting = false, show_progress = false,
            model_name = "dimstack_train",
        )
        @test !isnothing(out)
    end
end
