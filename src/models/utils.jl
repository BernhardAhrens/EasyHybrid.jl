export scale_single_param, default, lower, upper, hard_sigmoid, inv_hard_sigmoid, inv_sigmoid, scale_single_param_minmax, param_scale, parameter_scales

# Define the hard sigmoid activation function
function hard_sigmoid(x)
    return clamp.(0.2 .* x .+ 0.5, 0.0, 1.0)
end

# Inverse of `hard_sigmoid` on the linear region (0, 1).
# Saturated inputs (y ≤ 0 or y ≥ 1) are extrapolated linearly since the
# clamp makes the forward map non-invertible there.
function inv_hard_sigmoid(y)
    return (y .- 0.5) ./ 0.2
end

function default(p::ParameterContainer)
    return p.table[:, :default]
end

function lower(p::ParameterContainer)
    return p.table[:, :lower]
end

function upper(p::ParameterContainer)
    return p.table[:, :upper]
end

pnames(p::ParameterContainer) = keys(p.table.axes[1])

const _PARAM_SCALES = (:linear, :log)

"""
    parameter_scales(values::NamedTuple)

Per-parameter scale (`:linear` or `:log`) from each `(default, lower, upper[, scale])` spec.
"""
function parameter_scales(values::NamedTuple)
    return NamedTuple{keys(values)}(map(_param_scale, values))
end

function _param_scale(spec)
    length(spec) == 3 && return :linear
    length(spec) == 4 || throw(ArgumentError(
        "parameter spec must be (default, lower, upper) or (default, lower, upper, :linear|:log), got $spec"
    ))
    scale = spec[4]
    scale in _PARAM_SCALES || throw(ArgumentError(
        "parameter scale must be :linear or :log, got $(repr(scale))"
    ))
    if scale === :log && !(spec[1] > 0 && spec[2] > 0 && spec[3] > 0)
        throw(ArgumentError("log-scale parameter bounds must be positive, got $spec"))
    end
    return scale
end

param_scale(p::ParameterContainer, name) = getfield(p.scales, name)

"""
    scale_single_param(name, raw_val, parameters)

Map an unconstrained value through a sigmoid onto `[lower, upper]`.
With `:log` scale the interpolation is in log space, so the returned
value is still the physical parameter (`exp` of the log-space mix).
"""
function scale_single_param(name, raw_val, hm::ParameterContainer)
    ℓ = lower(hm)[name]
    u = upper(hm)[name]
    s = sigmoid.(raw_val)
    if param_scale(hm, name) === :log
        return exp.(log(ℓ) .+ (log(u) - log(ℓ)) .* s)
    end
    return ℓ .+ (u .- ℓ) .* s
end

inv_sigmoid(y) = log.(y ./ (1 .- y))

"""
    scale_single_param_minmax(name, hm::AbstractHybridModel)

Unconstrained initialization that maps back to the parameter default
under [`scale_single_param`](@ref). `:log` uses the same interpolation
in log space.
"""
function scale_single_param_minmax(name, hm::ParameterContainer)
    ℓ = lower(hm)[name]
    u = upper(hm)[name]
    d = default(hm)[name]
    if param_scale(hm, name) === :log
        return inv_sigmoid((log(d) - log(ℓ)) / (log(u) - log(ℓ)))
    end
    return inv_sigmoid((d - ℓ) / (u - ℓ))
end
