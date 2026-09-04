export prepare_data

function prepare_data(hm, data::KeyedArray; cfg = DataConfig(), kwargs...)
    predictors, forcings, targets = get_prediction_target_names(hm)
    # KeyedArray: use () syntax for views that are differentiable
    if predictors isa NamedTuple
        X = NamedTuple(name => Array(data(p)) for (name, p) in pairs(predictors))
    else
        X = Array(data(predictors))
    end
    forcings_nt = NamedTuple(forcing => Array(data(forcing)) for forcing in forcings)
    targets_nt = NamedTuple(target => Array(data(target)) for target in targets)
    return ((X, forcings_nt), targets_nt)
end

function prepare_data(hm, data::AbstractDimArray; kwargs...)
    predictors, forcings, targets = get_prediction_target_names(hm)
    # KeyedArray: use () syntax for views that are differentiable
    X_arr = data[variable = At(predictors)]
    forcings_nt = NamedTuple(forcing => data[variable = At(forcing)] for forcing in forcings)
    targets_nt = NamedTuple(target => data[variable = At(target)] for target in targets)
    # DimArray: use [] syntax (copies, but differentiable)
    return ((X_arr, forcings_nt), targets_nt)
end

function prepare_data(hm, data::DataFrame; array_type = :KeyedArray, drop_missing_rows = true)
    predictors, forcings, targets = get_prediction_target_names(hm)
    ds = _clean_to_labeled_array(
        predictors, forcings, targets, v -> data[!, v];
        array_type = array_type, drop_missing_rows = drop_missing_rows,
    )
    return prepare_data(hm, ds)
end

function prepare_data(hm, data::AbstractDimStack; array_type = :DimArray, drop_missing_rows = true)
    predictors, forcings, targets = get_prediction_target_names(hm)
    ds = _clean_to_labeled_array(
        predictors, forcings, targets, v -> vec(parent(data[v]));
        array_type = array_type, drop_missing_rows = drop_missing_rows,
    )
    return prepare_data(hm, ds)
end

function prepare_data(hm, data::Tuple; kwargs...)
    return data
end

"""
    _clean_to_labeled_array(predictors, forcings, targets, getcol; array_type, drop_missing_rows)

Shared cleaning/assembly routine for tabular inputs (DataFrame, DimStack, or any
column-addressable source). `getcol(name::Symbol)` must return the column/layer
for `name` as an `AbstractVector`; all columns must share the same length.

Steps (identical for every source):
1. Select only the variables referenced by the model (`predictors ∪ forcings ∪ targets`).
2. Coerce each column to `Float32`, mapping `missing` to `NaN`.
3. If `drop_missing_rows`, keep only samples whose predictors/forcings are all
   finite *and* that have at least one present (non-NaN) target.
4. Wrap the result as a `(:variable, :batch_size)` `KeyedArray` or `DimArray`.

Returns a labeled 2D array ready for the array-based `prepare_data` methods.
"""
function _clean_to_labeled_array(predictors, forcings, targets, getcol; array_type = :KeyedArray, drop_missing_rows = true)
    all_vars = unique([vcat(predictors...); forcings; targets])

    cols = [Float32.(coalesce.(getcol(v), NaN32)) for v in all_vars]
    nsamples = isempty(cols) ? 0 : length(first(cols))
    all(c -> length(c) == nsamples, cols) ||
        throw(ArgumentError("All variables must have the same number of samples; got lengths $(length.(cols))"))

    M = Matrix{Float32}(undef, length(all_vars), nsamples)
    for i in eachindex(cols)
        @inbounds M[i, :] = cols[i]
    end

    if drop_missing_rows
        predforce = setdiff(all_vars, targets)
        pf_idx = findall(in(predforce), all_vars)
        t_idx = findall(in(targets), all_vars)
        keep = trues(nsamples)
        for j in 1:nsamples
            pf_ok = isempty(pf_idx) ? true : all(!isnan, @view M[pf_idx, j])
            t_ok = isempty(t_idx) ? true : any(!isnan, @view M[t_idx, j])
            keep[j] = pf_ok & t_ok
        end
        M = M[:, keep]
    end

    return _wrap_labeled_array(M, all_vars, array_type)
end

function _wrap_labeled_array(M::AbstractMatrix, all_vars, array_type::Symbol)
    if array_type == :KeyedArray
        return KeyedArray(M; variable = all_vars, batch_size = 1:size(M, 2))
    elseif array_type == :DimArray
        return DimArray(M, (Dim{:variable}(all_vars), Dim{:batch_size}(1:size(M, 2))))
    else
        throw(ArgumentError("array_type must be :KeyedArray or :DimArray, got :$array_type"))
    end
end

"""
    prepare_data(hm, data::DataFrame; array_type=:KeyedArray, drop_missing_rows=true)
    prepare_data(hm, data::AbstractDimStack; array_type=:DimArray, drop_missing_rows=true)
    prepare_data(hm, data::KeyedArray)
    prepare_data(hm, data::AbstractDimArray)
    prepare_data(hm, data::Tuple)

Prepare data for training by extracting predictor/forcing and target variables based on the hybrid model's configuration.

`DataFrame` and `AbstractDimStack` inputs share one identical cleaning path
(`_clean_to_labeled_array`): select the model's variables, coerce to `Float32`,
map `missing` to `NaN`, optionally drop incomplete samples, and wrap into a
labeled `(:variable, :batch_size)` array. A `DimStack` therefore behaves like a
tabular source while keeping its named layers. Already-labeled `KeyedArray` /
`AbstractDimArray` inputs are assumed clean and are only variable-selected.

# Arguments:
- `hm`: The Hybrid Model
- `data`: The input data: a `DataFrame`, `AbstractDimStack`, `KeyedArray`, `AbstractDimArray`, or a pre-prepared `Tuple`.
- `array_type`: (DataFrame/DimStack only) Output array type: `:KeyedArray` or `:DimArray`.
- `drop_missing_rows`: (DataFrame/DimStack only) If `true` (default), drop samples where any predictor/forcing is NaN or all targets are NaN.

# Returns:
- For `DataFrame`/`AbstractDimStack`: a tuple of (predictors_forcing, targets) as KeyedArrays or DimArrays depending on `array_type`.
- For `KeyedArray`: a tuple of (predictors_forcing, targets) as KeyedArrays.
- For `AbstractDimArray`: a tuple of (predictors_forcing, targets) as DimArrays.
- For a `Tuple`: returned as-is.
"""
function prepare_data end

"""
    get_prediction_target_names(hm)
Utility function to extract predictor/forcing and target names from a hybrid model.

# Arguments:
- `hm`: The Hybrid Model

Returns a tuple of (predictors_forcing, targets) names.
"""
function get_prediction_target_names(hm)
    targets = hm.targets
    predictors = hm.predictors
    forcings = hm.forcing

    if isempty(predictors)
        @warn "Note that you don't have predictors variables."
    end
    if isempty(forcings)
        @warn "Note that you don't have forcing variables."
    end
    if isempty(targets)
        @warn "Note that you don't have target names."
    end
    return predictors, forcings, targets
end
