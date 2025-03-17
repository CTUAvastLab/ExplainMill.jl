# heuristic_pruning_methods = [:Flat_HAdd, :Flat_HArr, :Flat_HArrft, :LbyL_HAdd, :LbyL_HArr, :LbyL_HArrft]
heuristic_pruning_methods = [:Flat_HAdd, :Flat_HArr, :LbyL_HAdd, :LbyL_HArr]
partial_pruning_methods = [:LbyLo_HAdd, :LbyLo_HArr]
greedy_pruning_methods = [:Flat_Gadd, :Flat_Garr, :LbyL_Gadd, :LbyL_Garr]
@testset "Checking integration with pruner" begin
    ds = specimen_sample()
    model = reflectinmodel(ds, d -> Dense(d, 4), SegmentedMean, all_imputing = true)

    @testset "heuristic methods (GnnExplainer)" begin
        @test_broken false
    end

    @testset "heuristic methods" begin
        # for e in [ConstExplainer(), StochasticExplainer(), GnnExplainer(200), GradExplainer(), ExplainMill.DafExplainer()]
        for e in [ConstExplainer(), StochasticExplainer(), GradExplainer(), ExplainMill.DafExplainer()]
            mk = ExplainMill.add_participation(stats(e, ds, model))

            o = softmax(model(ds))[:]
            τ = 0.9 * maximum(o) 
            class = argmax(softmax(model(ds))[:])
            f = () -> softmax(model(ds[mk]))[class] - τ
            @test !ExplainMill.flatsearch!(f, mk)
            @test !ExplainMill.levelbylevelsearch!(f, mk)

            @test !ExplainMill.flatsearch!(f, mk, random_removal = true)
            @test !ExplainMill.levelbylevelsearch!(f, mk, random_removal = true)

            # this test is more like a test that it passes
            for pruning_method in heuristic_pruning_methods
                @test explain(e, ds, model; rel_tol=0.9, pruning_method) isa ExplainMill.AbstractStructureMask
            end

            for pruning_method in partial_pruning_methods
                @test explain(e, ds, model; rel_tol=0.9, pruning_method) isa ExplainMill.AbstractStructureMask
            end
        end
    end

    @testset "non-heuristic methods" begin
        for e in [ConstExplainer()]
            mk = ExplainMill.add_participation(stats(e, ds, model))

            o = softmax(model(ds))[:]
            τ = 0.9 * maximum(o) 
            class = argmax(softmax(model(ds))[:])
            f = () -> softmax(model(ds[mk]))[class] - τ

            @test !ExplainMill.flatsfs!(f, mk, random_removal = true)
            @test !ExplainMill.levelbylevelsfs!(f, mk, random_removal = true)

            # this test is more like a test that it passes
            for pruning_method in greedy_pruning_methods
                @test explain(e, ds, model; rel_tol=0.9, pruning_method) isa ExplainMill.AbstractStructureMask
            end
        end
    end

    @testset "level-by-level search with partial evaluation (GnnExplainer)" begin
        @test_broken false
    end

    @testset "level-by-level search with partial evaluation" begin
        for e in [ConstExplainer(), StochasticExplainer(), GradExplainer(), DafExplainer()]
        # for e in [ConstExplainer(), StochasticExplainer(), GnnExplainer(200), GradExplainer(), DafExplainer()]
            mk = ExplainMill.add_participation(stats(e, ds, model))

            o = softmax(model(ds))[:]
            τ = 0.9 * maximum(o) 
            class = argmax(softmax(model(ds))[:])
            f = (model, ds, mk) -> softmax(model(ds[mk]))[class] - τ
            fₚ = o -> softmax(o)[class] - τ

            @test !ExplainMill.levelbylevelsearch!(f, model, ds, mk, random_removal = true)
            @test !ExplainMill.prune!(mk, model, ds, fₚ, :LbyLo_HAdd)
            @test !ExplainMill.prune!(mk, model, ds, fₚ, :LbyLo_HArr)
        end
    end

    @testset "Testing shared ObservationMasks" begin
        function adjust_mask(mk)
            mx = mk.child.child.childs.on.mask
            m = ObservationMask(mx)
            @set! mk.child.child.childs.an = m
            @set! mk.child.child.childs.cn = m
            @set! mk.child.child.childs.on = m
            @set! mk.child.child.childs.sn = m
            mk
        end

        e = ConstExplainer()
        # just test that the explanation passes
        for pruning_method in heuristic_pruning_methods
            @test explain(e, ds, model; rel_tol=0.9, pruning_method, adjust_mask) isa ExplainMill.AbstractStructureMask
        end
        for pruning_method in greedy_pruning_methods
            @test explain(e, ds, model; rel_tol=0.9, pruning_method, adjust_mask) isa ExplainMill.AbstractStructureMask
        end
        for pruning_method in partial_pruning_methods
            @test explain(e, ds, model; rel_tol=0.9, pruning_method, adjust_mask) isa ExplainMill.AbstractStructureMask
        end
    end
end

@testset "Simple end2end test with a strict equality (GnnExplainer)" begin
    @test_broken false
end

@testset "Simple end2end test with a strict equality" begin
    ds = ArrayNode(ones(Float32, 6, 1))
    motif = [1, 0, 0, 0, 0, 1]
    model = ArrayModel(preimputing_dense(6, 2))
    model.m.weight.W .= [1 0 0 0 0 1;-1 0 0 0 0 -1]
    model.m.bias .= 0
    # for e in [ConstExplainer(), StochasticExplainer(), GnnExplainer(200), GradExplainer(), ExplainMill.DafExplainer()]
    for e in [ConstExplainer(), StochasticExplainer(), GradExplainer(), ExplainMill.DafExplainer()]

        mk = ExplainMill.add_participation(stats(e, ds, model))
        class = [1]
        τ = logitconfgap(model, ds, class)
        f = () -> sum(logitconfgap(model, ds[mk], class) .- τ)

        ExplainMill.flatsfs!(f, mk, random_removal = true)
        @test prunemask(mk.mask) ≈ motif

        ExplainMill.levelbylevelsfs!(f, mk, random_removal = true)
        @test prunemask(mk.mask) ≈ motif

        for pruning_method in setdiff(ExplainMill.pruning_methods(), [:Flat_HAdd, :Flat_Gadd, :LbyL_HAdd, :LbyL_Gadd, :LbyLo_HAdd, :LbyLo_Gadd])
            @test prunemask(explain(e, ds, model; rel_tol=1, pruning_method).mask) ≈ motif
        end

        # Addition methods cannot remove useless parts of the explanation, therefore
        # we can only test that the explanation is a superset of a motif
        for pruning_method in [:Flat_HAdd, :Flat_Gadd, :LbyL_HAdd, :LbyL_Gadd, :LbyLo_HAdd, :LbyLo_Gadd]
            @test all(prunemask(explain(e, ds, model; rel_tol=1, pruning_method).mask) .≥ motif)
        end
    end
end
