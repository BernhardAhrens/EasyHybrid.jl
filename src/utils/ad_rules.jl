# Keyword arguments are pervasive in EasyHybrid's differentiated code: the
# mechanistic model is called as `f(; forcings..., params...)` and the training
# loss is evaluated through `compute_loss`. Julia implements keyword calls by
# splitting the caller's NamedTuple into the callee's declared and defaulted
# arguments, which is done by the symbol-bookkeeping helpers below.
#
# Those helpers return `Bool`s and tuples of `Symbol`s, so they carry no
# derivative information, but Zygote does not know that and builds an
# *uninferred* pullback for each of them on every keyword call. Profiling a
# gradient through a mechanistic model showed that single uninferred pullback
# accounting for roughly a quarter of the samples, making a keyword call about
# 3x more expensive than the equivalent positional one.
#
# Declaring them non-differentiable removes the pullbacks without changing any
# gradient value. They are Base internals, so each rule is guarded on the
# function still existing to avoid breaking on future Julia versions.
for (fname, nargs) in ((:diff_names, 2), (:sym_in, 2), (:merge_names, 2))
    isdefined(Base, fname) || continue
    args = ntuple(_ -> :(::Any), nargs)
    @eval ChainRulesCore.@non_differentiable Base.$fname($(args...))
end
