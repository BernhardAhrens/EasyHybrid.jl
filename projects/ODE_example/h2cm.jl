# H2CM in EasyHybrid: one keyword step + constructHybridODE.
# Physics follows zavud/h2cm (water_cycle_forward + carbon_cycle_forward).
# Operator splitting stays inside the step; EasyHybrid only does Euler `u ← u + du`.
#
# Mapping:
#   H2CM LSTM 1 (water partitioning) → group :water
#   H2CM LSTM 2 (CUE, fAPAR)        → group :cue
#   H2CM LSTM 3 (rb, alpha_Es)      → group :rb
#   H2CM FC  (WUE, alpha_T)         → group :wue  (`recurrent=false`; H2CM uses a dense net)
#   H2CM FC_Static (sm_max, alpha_Ei) → groups with site-only predictors (auto-detected static)
#   H2CM 12-D static embedding      → site columns in each group's predictors (repeated in time)
#   H2CM spin-up                    → not yet; windows start from zero ICs
# State / process carry is listed in each group's `predictors` (`:wue` has none).

using EasyHybrid
using DimensionalData

function h2cm_step(;
        swe, SM, GW,
        alpha_snow, alpha_r_soil, alpha_r_gw, alpha_Es, alpha_T, alpha_Ei,
        rb, cue, fAPAR, wue, sm_max,
        Q10, beta_snow, beta_baseflow, beta_co2,
        rn, prec, tair, vpd, CO2,
    )
    _ = vpd
    ϵ = 1.0f-8
    tair_c = tair .- 273.15f0
    is_snow = ifelse.(tair_c .<= 0, 1.0f0, 0.0f0)
    is_rain = 1.0f0 .- is_snow

    snow_acc = prec .* beta_snow .* is_snow
    snow_melt = min.(swe, max.(tair_c, 0.0f0) .* alpha_snow)
    swe_next = max.(swe .+ snow_acc .- snow_melt, 0.0f0)

    rn_mm = max.(rn .* 0.0864f0 ./ 2.45f0, 0.0f0)
    rainfall = prec .* is_rain
    Ei = min.(min.(rainfall, fAPAR .* alpha_Ei), rn_mm)
    rn_mm = rn_mm .- Ei

    ET_pot = min.(rn_mm, SM)
    Es = (1.0f0 .- fAPAR) .* ET_pot .* alpha_Es
    SM_es = SM .- Es
    rn_mm = rn_mm .- Es

    ET_pot = min.(rn_mm, SM_es)
    T = fAPAR .* ET_pot .* alpha_T
    SM_T = SM_es .- T

    water_in = rainfall .+ snow_melt .- Ei
    sm_deficit = sm_max .- SM_T
    r_soil_frac = min.(1.0f0, sm_deficit ./ max.(water_in, ϵ)) .* alpha_r_soil
    r_soil = r_soil_frac .* water_in
    SM_next = SM_T .+ r_soil
    rel_SM = SM_next ./ sm_max

    r_gw_frac = (1.0f0 .- r_soil_frac) .* alpha_r_gw
    r_gw = r_gw_frac .* water_in
    runoff_s = (1.0f0 .- r_soil_frac) .* (1.0f0 .- alpha_r_gw) .* water_in
    baseflow = GW .* beta_baseflow
    GW_next = GW .+ r_gw .- baseflow
    runoff = runoff_s .+ baseflow
    ET = Ei .+ Es .+ T
    tws = swe_next .+ GW_next .+ SM_next

    gpp = T .* wue .* CO2 .* beta_co2
    npp = gpp .* cue
    Ra = gpp .- npp
    Rh = rb .* Q10 .^ (0.1f0 .* (tair_c .- 15.0f0))
    ter = Ra .+ Rh
    nee = ter .- gpp

    return (;
        dswe = swe_next .- swe, dSM = SM_next .- SM, dGW = GW_next .- GW,
        ET, runoff, tws, gpp, npp, nee, ter, rel_SM, fAPAR,
        swe = swe_next, SM = SM_next, GW = GW_next,
    )
end

# Bounds stand in for H2CM's sigmoid / softplus activations.
parameters = (
    alpha_snow = (1.0f0, 0.0f0, 20.0f0),
    alpha_r_soil = (0.5f0, 0.0f0, 1.0f0),
    alpha_r_gw = (0.5f0, 0.0f0, 1.0f0),
    alpha_Es = (0.5f0, 0.0f0, 1.0f0),
    alpha_T = (0.5f0, 0.0f0, 1.0f0),
    alpha_Ei = (1.0f0, 0.0f0, 20.0f0),
    rb = (1.0f0, 0.0f0, 20.0f0),
    cue = (0.5f0, 0.0f0, 1.0f0),
    fAPAR = (0.4f0, 0.0f0, 1.0f0),
    wue = (1.0f0, 0.0f0, 20.0f0),
    sm_max = (200.0f0, 1.0f0, 1000.0f0),
    Q10 = (1.5f0, 1.0f0, 10.0f0),
    beta_snow = (0.5f0, 0.0f0, 1.0f0),
    beta_baseflow = (0.1f0, 0.0f0, 1.0f0),
    beta_co2 = (0.0015f0, 0.0f0, 0.01f0),
)

static_cols = [:clay, :sand, :silt, :elevation]
forcing = [:rn, :prec, :tair, :vpd, :CO2]
targets = [:ET, :gpp, :nee, :tws, :runoff]

h2cm = constructHybridODE(
    (
        water = [:rn, :prec, :rel_SM, :swe, :GW, :fAPAR, static_cols...],
        cue = [:rn, :tair, :vpd, :CO2, :rel_SM, :fAPAR, :npp, static_cols...],
        rb = [:rn, :prec, :npp, :fAPAR, static_cols...],
        wue = [:rn, :vpd, static_cols...],
        sm_max = static_cols,
        alpha_Ei = static_cols,
    ),
    forcing,
    targets,
    h2cm_step,
    parameters,
    (
        water = [:alpha_snow, :alpha_r_soil, :alpha_r_gw],
        cue = [:cue, :fAPAR],
        rb = [:rb, :alpha_Es],
        wue = [:wue, :alpha_T],
        sm_max = [:sm_max],
        alpha_Ei = [:alpha_Ei],
    ),
    [:Q10, :beta_snow, :beta_baseflow, :beta_co2];
    hidden_dims = 16,
    recurrent = (wue = false, sm_max = false, alpha_Ei = false),
    state = [:swe, :SM, :GW],
    deriv = [:dswe, :dSM, :dGW],
    scale_nn_outputs = true,
)

# Dummy window: (variable, time, batch)
T, B = 8, 4
vars = unique([forcing; static_cols; targets])
x = DimArray(
    rand(Float32, length(vars), T, B),
    (Dim{:variable}(vars), Dim{:time}(1:T), Dim{:batch_size}(1:B)),
)
x[variable = At(:tair)] .+= 280.0f0
for c in static_cols
    sl = x[variable = At(c)]
    sl .= sl[1:1, :]
end
ps, st = Lux.setup(Random.default_rng(), h2cm)
out, _ = h2cm(x, ps, st)
out.ET
out.nee
out.rel_SM
