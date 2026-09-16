# Same-step GPP → Reco for algebraic `constructHybridModel` (proposal only).
#
# Today MultiNN runs every net on data columns, then calls the process once.
# Reco cannot see live GPP from the GPP/RUE net in that forward.
# Precomputed GPP as a *table column* already works; these three ways feed
# *this step's* GPP (or RUE) into the Reco net. No `feedback` keyword:
# names in `predictors` that are not data are resolved in-step.
#
# Shared physics (FluxPart / Q10 partitioning):
#   GPP  = SW_IN * RUE / 12.011
#   Reco = Rb * Q10^((TA - 15)/10)
#   NEE  = Reco - GPP

mGPP(; SW_IN, RUE) = SW_IN .* RUE ./ 12.011f0

mReco(; Rb, Q10, TA, tref = 15.0f0) = Rb .* Q10 .^ (0.1f0 .* (TA .- tref))

function mNEE(; SW_IN, TA, RUE, Rb, Q10)
    GPP = mGPP(; SW_IN, RUE)
    Reco = mReco(; Rb, Q10, TA)
    return (; NEE = Reco .- GPP, GPP, Reco, RUE, Rb, Q10)
end

parameters = (
    RUE = (1.5f0, 0.0f0, 3.0f0),
    Rb = (1.0f0, 0.0f0, 12.0f0),
    Q10 = (1.5f0, 1.0f0, 4.0f0),
)
forcing = [:SW_IN, :TA]
targets = [:NEE]

# -----------------------------------------------------------------------------
# 1) NN → NN (topological order)
#
# Reco predictors may include another *group name*. That group's NN output is
# concatenated in the same forward, then the process runs once.
# Reco sees RUE (or a group that directly predicts GPP), not process GPP.
# Closest to current MultiNN (`keys(predictors)` stay neural params).

# proposed:
# constructHybridModel(
#     (RUE = [:SW_IN, :VPD], Rb = [:TA, :RUE]),
#     forcing, targets, mNEE, parameters, [:Q10],
# )
#
# Forward: RUE net (data only) → Rb net (data + RUE) → mNEE.
# Cycle in the predictor graph would error.

# -----------------------------------------------------------------------------
# 2) Staged process (Reco sees flux GPP)
#
# Reco predictors include `:GPP`, which is not data and not an NN. After the
# data-only nets, a GPP fragment runs; Reco net sees that flux; then NEE.
# Same "list the name in predictors" rule as ODE carry, but same algebraic step.

# proposed:
# constructHybridModel(
#     (RUE = [:SW_IN, :VPD], Rb = [:TA, :GPP]),
#     forcing, targets,
#     (GPP = mGPP, NEE = mNEE),
#     parameters, [:Q10],
# )
#
# Forward: RUE net → mGPP(; SW_IN, RUE) → Rb net (data + GPP) → mNEE.
# Mechanistic model as a NamedTuple of stages; later stages see earlier outputs
# plus NN params. This is the one that matches "GPP net then Reco net" with
# GPP as photosynthesis, not as an extra NN head.

# -----------------------------------------------------------------------------
# 3) Two-pass, one process function (no new constructor args)
#
# Keep `mechanistic_model = mNEE`. If a predictor is missing from the data,
# run all nets whose predictors are data-only, call mNEE, take matching names
# from that return (GPP), run the remaining nets, call mNEE again.
#
# proposed (API unchanged):
# constructHybridModel(
#     (RUE = [:SW_IN, :VPD], Rb = [:TA, :GPP]),
#     forcing, targets, mNEE, parameters, [:Q10],
# )
#
# Needs mNEE to accept a missing/unused Rb on the first call, *or* a small
# allow-list of first-pass outputs (e.g. only names already determined by
# RUE + forcing). More implicit than (2); process is invoked twice.

# Prefer (2) when GPP is physics; (1) when Reco should see another NN output
# directly; (3) only if we refuse extra constructor surface.
