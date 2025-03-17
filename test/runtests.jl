using Test

using Duff
using ExplainMill
using FiniteDifferences
using Flux
using JsonGrinder
using Mill
using Random
using Setfield
using SparseArrays
using StatsBase

import ExplainMill: Mask, invalidate!, foreach_mask, participate, mask, logitconfgap
import ExplainMill: FeatureMask, SparseArrayMask, CategoricalMask, ProductMask, BagMask
import ExplainMill: NGramMatrixMask, EmptyMask, FollowingMasks, yarason, GradientMask
import ExplainMill: FlatView, identifycolumns, create_mask_structure, ParticipationTracker
import ExplainMill: partialeval, collectmasks, collect_masks_with_levels, mapmask, present
import ExplainMill: findnonempty, ModelLens, updateparticipation!

function specimen_sample()
    an = ArrayNode(reshape(collect(1:10), 2, 5))
    on = ArrayNode(Mill.maybehotbatch([1, 2, 3, 1, 2], 1:4))
    cn = ArrayNode(sparse(Float32[1 0 3 0 5; 0 2 0 4 0]))
    sn = ArrayNode(NGramMatrix(["a","b","c","d","e"], 3, 256, 2053))
    BagNode(BagNode(ProductNode(; an, on, cn, sn),
             AlignedBags([1:2, 3:3, 4:5])),
             AlignedBags([1:3]))
end

# TODO test that explanations are indeed subsets of the original json
# TODO call e2boolean with full and sampled masks

@testset "ExplainMill.jl" begin
    include("structuralmasks.jl")
    include("flatmasks.jl")
    include("heuristics.jl")
    include("partialeval.jl")
    include("explain.jl")
    include("json_output.jl")
end
