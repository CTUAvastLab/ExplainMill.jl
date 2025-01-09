function get_thresholds(cg, abs_tol, rel_tol)
    if isnothing(abs_tol) && isnothing(rel_tol)
        @warn "No tolerance specified, setting rel_tol=0.9" maxlog=5
        rel_tol = 0.9
    end
    if isnothing(abs_tol)
        @assert 0 ≤ rel_tol ≤ 1 "Relative tolerance must be in [0, 1]!"
        rel_tol .* cg
    else
        @assert all(abs_tol .≤ cg) "Absolute tolerance must be smaller than the confidence gap!"
        cg .- abs_tol
    end
end

"""
    explain(e, ds::AbstractMillNode, model::AbstractMillModel, class; clustering = ExplainMill._nocluster, pruning_method=:LbyL_HArr,
        abs_tol=nothing, rel_tol=nothing, adjust_mask = identity)
    explain(e, ds::AbstractMillNode, model::AbstractMillModel, class; kwargs...)
    explain(e, ds::AbstractMillNode, model::AbstractMillModel, extractor::AbstractExtractor; kwargs...)

    explain the decision of `model` on sample `ds` using heuristic method `e`
    determining importance of subtrees (features) of the sample and using 
    `pruning_method`. If `class` that is wished to be explained is not provided, 
    the most probable class is used.
"""
function explain(e, ds::AbstractMillNode, model::AbstractMillModel, class; clustering = ExplainMill._nocluster, pruning_method=:LbyL_HArr,
        abs_tol=nothing, rel_tol=nothing, adjust_mask = identity)
    cg = logitconfgap(model, ds, class)
    @assert all(0 .≤ cg) "Cannot explain class with negative confidence gap!"
    mk = stats(e, ds, model, class, clustering)
    mk = adjust_mask(mk)
    thresholds = get_thresholds(cg, abs_tol, rel_tol)
    fₚ(o) = sum(min.(logitconfgap(o, class) .- thresholds, 0))
    prune!(mk, model, ds, fₚ, pruning_method)
    mk
end


function explain(e, ds::AbstractMillNode, model::AbstractMillModel; kwargs...)
    class = Flux.onecold(softmax(model(ds)))
    if length(unique(class))  > 1 
    	@warn "Two or more classes predicted by the model!, wish you know what you are doing."
    end
    explain(e, ds, model, class; kwargs...)
end

function explain(e, ds::AbstractMillNode, model::AbstractMillModel, extractor::Extractor; kwargs...)
    dssm = Mill.dropmeta(ds)
    mk = explain(e, dssm, model; kwargs...)
    e2boolean(ds, mk, extractor)
end


"""
    explainf(e, ds::AbstractMillNode, model::AbstractMillModel, fₛ, fₚ)
    
    A more low-level api allowing to prescribe the function used to (heuristically) 
    calculate importance of features (`fₛ(o)`) and used during pruning (`fₚ(o)`),
    where `o` is the output of the model on the sample. During pruning, we try to 
    find the smallest subset of the sample such that `fₚ > 0`.
"""
function explainf(e, ds::AbstractMillNode, model::AbstractMillModel, fₛ, fₚ; clustering = ExplainMill._nocluster, pruning_method=:LbyL_HArr,
        abs_tol=nothing, rel_tol=nothing, adjust_mask = identity)
    mk = statsf(e, ds, model, fₛ, clustering)
    mk = adjust_mask(mk)
    prune!(mk, model, ds, fₚ, pruning_method)
    mk
end
