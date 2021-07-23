module ExplainMill
using Mill
using Duff
using SparseArrays
using StatsBase
using Distances
using Clustering
using Flux
using Zygote
using JSON
using HierarchicalUtils
using DataFrames
using JSON
using Zygote
using JsonGrinder
using Setfield
using DataStructures
import Mill: partialeval
import Flux: onecold

output(ds::ArrayNode) = ds.data
output(x::AbstractArray) = x

include("masks/masks.jl")
export prunemask, diffmask, HeuristicMask, SimpleMask
include("output/logic_output.jl")

include("heuristics/const_explainer.jl")
include("heuristics/daf_explainer.jl")
include("heuristics/gnn_explainer.jl")
include("heuristics/grad_explainer.jl")
include("heuristics/stochastic_explainer.jl")
export GnnExplainer, GradExplainer, ConstExplainer
export StochasticExplainer, DafExplainer, GreedyGradient
export stats, heuristic

include("prunemissing.jl")
include("predict.jl")
include("matching.jl")
include("pruning/pruning.jl")
include("utils/lensutils.jl")
include("explain.jl")
export pruning_methods

include("hierarchical_utils.jl")

Base.show(io::IO, ::T) where T <: AbstractStructureMask = print(io, Base.nameof(T))
Base.show(io::IO, ::MIME"text/plain", n::AbstractStructureMask) = HierarchicalUtils.printtree(io, n; trav=false, htrunc=3, vtrunc=20)
Base.getindex(n::AbstractStructureMask, i::AbstractString) = HierarchicalUtils.walk(n, i)

export explain, e2boolean, confidence, prunemissing, prune, e2boolean
export removeabsent, removemissing
end # module
