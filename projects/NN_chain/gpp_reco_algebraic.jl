# Same-step GPP → Reco without changing the mechanistic model (proposal).
#
# User API (same `mNEE` with or without the GPP→Rb edge — no `feedback` kw,
# no `Rb(GPP)` callable in the process):
#
#   constructHybridModel(
#       (RUE = [:SW_IN, :VPD], Rb = [:TA, :GPP]),
#       forcing, targets, mNEE, parameters, [:Q10],
#   )
#
# `mNEE` always takes **values** (`Rb` is an array), same as today's hybrid
# and the same as ODEHybrid. `:GPP` in `Rb`'s predictors is not a data column;
# EasyHybrid resolves it in this forward, then calls `mNEE` as usual.
#
# ODEHybrid already does this across time: list `:GPP` (or `:npp`) in predictors,
# process unchanged, next step sees last process output. Algebraic needs the
# same listing rule *inside one forward*.

mGPP(; SW_IN, RUE) = SW_IN .* RUE ./ 12.011f0

function mNEE(; SW_IN, TA, RUE, Rb, Q10)
    GPP = mGPP(; SW_IN, RUE)
    Reco = Rb .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
    return (; NEE = Reco .- GPP, GPP, Reco, RUE, Rb, Q10)
end

parameters = (
    RUE = (1.5f0, 0.0f0, 3.0f0),
    Rb = (1.0f0, 0.0f0, 12.0f0),
    Q10 = (1.5f0, 1.0f0, 4.0f0),
)
forcing = [:SW_IN, :TA]
targets = [:NEE]

# Resolve each name in a group's predictors:
#   in the data              → column (as today)
#   another NN group name    → that net's output (topo order)          [A]
#   else                     → process output, not yet a value         [B]
#
# [A] never needs a second process call. [B] does, unless we wait for ODE.

# -----------------------------------------------------------------------------
# A) Sibling NN output (topo). Process once. mNEE unchanged.
#
#   (RUE = [:SW_IN, :VPD], Rb = [:TA, :RUE])
#
# Reco sees RUE, not flux GPP. Fine when the extra predictor *is* a neural
# param. Not the FluxPart case (`GPP = SW_IN * RUE / 12`).

# -----------------------------------------------------------------------------
# B) Process output as predictor — two-pass, mNEE unchanged (closest to ODE).
#
#   (RUE = [:SW_IN, :VPD], Rb = [:TA, :GPP])
#
# Forward:
#   1. NNs whose predictors are all data (RUE).
#   2. Call mNEE with those values + a stand-in for missing neural params (Rb).
#   3. Read GPP from the return (same names as ODE carry).
#   4. Run remaining NNs (Rb with TA + GPP).
#   5. Call mNEE again with the real Rb. Keep this return.
#
# Stand-in for step 2 (mNEE still requires `Rb` as a value):
#   B1. `default(parameters).Rb` (already on ParameterContainer)
#   B2. zeros, same shape as a batch
#   B3. omit Rb if we introspect kwargs — would *require* a default on mNEE
#       → user *does* change the process; drop this
#
# Prefer B1: first pass is a legal call of the same function. For FluxPart,
# GPP does not depend on Rb, so the stand-in only affects first-pass Reco/NEE,
# which we discard. Do **not** `ignore_derivatives` on pass 1: RUE → GPP →
# Rb-net must stay in the tape. Drop first-pass Reco/NEE only.
#
# If GPP depended on Rb (algebraic cycle), B1 is wrong; then C.

# -----------------------------------------------------------------------------
# C) Fixed point, still one mNEE.
#
#   Rb₀ = default(parameters).Rb
#   repeat: GPP = mNEE(...; Rb).GPP; Rb = NN(TA, GPP)  until stable
#   final mNEE with that Rb
#
# Needed only if GPP and Reco depend on each other. FluxPart is a DAG; B is enough.

# -----------------------------------------------------------------------------
# D) Callable `Rb(GPP)` injected as a kwarg (previous discussion).
#
# Process becomes `rb = Rb(GPP)` instead of `Reco = Rb .* …`. That *is* a
# second mechanistic model — the thing we want to avoid. Keep D as an
# internal implementation of B (hybrid builds a closure, evaluates it
# between the two mNEE calls). User-facing `mNEE` still sees an array.

# Prefer B1 for "list :GPP in predictors, mNEE unchanged". Same rule as
# ODEHybrid; the extra cost is a second process eval in one algebraic step.
