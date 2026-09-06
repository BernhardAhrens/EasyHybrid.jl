export display_parameter_bounds, construct_dispatch_functions, build_parameter_matrix

function construct_dispatch_functions(f)
    function new_f end  # Create a new generic function

    println("constructing on KeyedArray function for $f")
    function new_f(forcing_data::KeyedArray, parameters::NamedTuple, forcing_names::Vector{Symbol})
        forcing = toNamedTuple(forcing_data, forcing_names)
        parameter_container = ParameterContainer(parameters)
        return f(; forcing..., values(default(parameter_container))...)
    end

    function new_f(forcing_data::DataFrame, parameters::NamedTuple, forcing_names::Vector{Symbol})
        forcing = (; (name => forcing_data[!, name] for name in forcing_names)...)
        parameter_container = ParameterContainer(parameters)
        return f(; forcing..., values(default(parameter_container))...)
    end

    println("repeating kwargs style functions for $f (orignal function: $f)")
    function new_f(; kwargs...)
        return f(; kwargs...)
    end

    return new_f
end


"""
    build_parameter_matrix(parameter_defaults_and_bounds::NamedTuple)

Build a ComponentArray matrix from a NamedTuple containing parameter defaults and bounds.

This function converts a NamedTuple where each value is a tuple of (default, lower, upper)
or `(default, lower, upper, :log)` into a ComponentArray with named axes. Only the first
three elements are stored; the optional fourth (`:linear` or `:log`) is read by
[`ParameterContainer`](@ref) / [`scale_single_param`](@ref).

# Arguments
- `parameter_defaults_and_bounds::NamedTuple`: A NamedTuple where each key is a parameter name and each value is
  `(default, lower, upper)` or `(default, lower, upper, :linear|:log)`.

# Returns
- `ComponentArray`: A 2D ComponentArray with:
  - Row axis: Parameter names (from the NamedTuple keys)
  - Column axis: Bound types (:default, :lower, :upper)
  - Data: The parameter values organized in a matrix format

# Example
```julia
# Define parameter defaults and bounds
parameter_defaults_and_bounds = (
    θ_s = (0.464f0, 0.302f0, 0.700f0),     # Saturated water content [cm³/cm³]
    h_r = (1500.0f0, 1500.0f0, 1500.0f0),  # Pressure head at residual water content [cm]
    α   = (0.103f0, 0.01f0, 7.874f0, :log),  # Shape parameter [cm⁻¹], log scale
    n   = (3.163f0 - 1, 1.100f0 - 1, 20.000f0 - 1, :log),  # Shape parameter [-], log scale
)

# Build the ComponentArray
parameter_matrix = build_parameter_matrix(parameter_defaults_and_bounds)

# Access specific parameter bounds
parameter_matrix.θ_s.default  # Get default value for θ_s
parameter_matrix[:, :lower]   # Get all lower bounds
parameter_matrix[:, :upper]   # Get all upper bounds
```

# Notes
- The first three elements are always (default, lower, upper); an optional fourth is `:linear` or `:log`
- Prefer `(default, lower, upper, :log)` over pre-logging the three values so `parameters.name` stays in physical units
- The resulting ComponentArray can be used for parameter optimization and constraint handling
"""
function build_parameter_matrix(parameter_defaults_and_bounds::NamedTuple)
    param_names = collect(keys(parameter_defaults_and_bounds))
    bound_names = (:default, :lower, :upper)
    data = [ parameter_defaults_and_bounds[p][i] for p in param_names, i in 1:length(bound_names) ]
    row_ax = ComponentArrays.Axis(param_names)
    col_ax = ComponentArrays.Axis(bound_names)
    return ComponentArray(data, row_ax, col_ax)
end
