# Keyword ODE step: return a derivative for each state. Euler is `u ← u + du`.
# Several states / several LSTMs: see h2cm.jl.

function mOnePool_step(; C, rb, Q10, ta, tref = 15.0f0)
    reco = rb .* C .* Q10 .^ (0.1f0 .* (ta .- tref))
    dC = .- reco
    return (; dC, reco, Q10, rb, C)
end

function mOnePool_GPP_step(; C, SW_IN, TA, RUE, Rb, Q10)
    GPP = SW_IN .* RUE ./ 12.011f0
    RECO = Rb .* C .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
    dC = RECO .- GPP
    return (; dC, RECO, GPP, Q10, RUE, Rb)
end
