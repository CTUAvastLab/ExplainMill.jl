@testset "partialeval on mill structures" begin
    metadata = fill("metadata", 4)
    an1 = ArrayNode(rand(3, 4))
    b = BagNode(an1, [1:4, 0:-1], metadata)
    an2 = ArrayNode(randn(5, 4))
    wb = WeightedBagNode(an2, [1:2, 3:4], rand(4), metadata)
    pn = ProductNode(; b, wb)
    an3 = ArrayNode(rand(10, 2))
    ds = ProductNode((pn, an3))
    m = reflectinmodel(ds, d -> f64(Chain(Dense(d, 4, relu), Dense(4, 3))),
                       d -> f64(SegmentedMeanMax(d)))

    mm = m[2]
    dd = ds.data[2]
    @test partialeval(mm, dd, an3)[1] ≡ mm
    @test partialeval(mm, dd, an3)[2] ≡ dd
    @test partialeval(mm, dd, an1)[1] ≡ identity
    @test partialeval(mm, dd, an1)[2] ≈ mm(dd)
    tm, td = partialeval(mm, dd, an3)
    @test tm(td) ≈ mm(dd)
    @test partialeval(mm, dd, an1)[2] ≈ mm(dd)

    mm = m[1][:b]
    dd = ds.data[1][:b]
    @test partialeval(mm, dd, an1)[1] ≡ mm
    @test partialeval(mm, dd, an1)[2] ≡ dd
    @test partialeval(mm, dd, b)[1] ≡ mm
    @test partialeval(mm, dd, b)[2] ≡ dd
    tm, td = partialeval(mm, dd, an3)
    @test tm(td) ≈ mm(dd)
    @test partialeval(mm, dd, an3)[1] ≡ identity
    @test partialeval(mm, dd, an3)[2] ≈ mm(dd)

    mm = m[1][:wb]
    dd = ds.data[1][:wb]
    @test partialeval(mm, dd, an2)[1] ≡ mm
    @test partialeval(mm, dd, an2)[2] ≡ dd
    @test partialeval(mm, dd, wb)[1] ≡ mm
    @test partialeval(mm, dd, wb)[2] ≡ dd
    tm, td = partialeval(mm, dd, an3)
    @test tm(td) ≈ mm(dd)
    @test partialeval(mm, dd, an3)[1] ≡ identity
    @test partialeval(mm, dd, an3)[2] ≈ mm(dd)

    mm = m[1]
    dd = ds.data[1]
    @test partialeval(mm, dd, an1)[1][:b] ≡ mm[:b]
    @test partialeval(mm, dd, an1)[2].data[1] ≡ dd.data[1]
    @test partialeval(mm, dd, an1)[1][:wb] ≡ ArrayModel(identity)
    @test partialeval(mm, dd, an1)[2].data[2].data ≈ mm[:wb](dd.data[2])
    @test partialeval(mm, dd, an3)[1] ≡ identity
    @test partialeval(mm, dd, an3)[2] ≈ mm(dd)
    tm, td = partialeval(mm, dd, an1)
    @test tm(td) ≈ mm(dd)
    @test partialeval(m[2], ds.data[2], an1)[2] ≈ m[2](ds.data[2])

    @test partialeval(m, ds, 1)[1] ≡ identity
    @test partialeval(m, ds, 1)[2] ≈ m(ds)
    tm, td = partialeval(m, ds, an1)
    @test tm(td) ≈ m(ds)
end

# In partial evaluation we test that masks that are not 
# needed for inspecting changes of a particular mask 
# are evaluated to `ArrayNode` and that outputs 
# provides the same outputs
@testset "Partial Evaluation" begin 
    ds = specimen_sample()
    model = reflectinmodel(ds, d -> Dense(d, 4), SegmentedMean, all_imputing = true)

    for mfun in [
        d -> SimpleMask(fill(true, d)),
        d -> ParticipationTracker(SimpleMask(fill(true, d))),
    ]

        mk = create_mask_structure(ds, mfun)

        @testset "topmost bagmask" begin
            for mr in [mk, mk.mask]
                mₚ, dsₚ, mkₚ, keep = partialeval(model, ds, mk, [mr])
                @test mₚ.im.m == identity
                @test dsₚ.data isa ArrayNode
                @test mkₚ.child isa EmptyMask
                @test mkₚ.mask === mk.mask
                @test keep
                @test model(ds[mk]) ≈ mₚ(dsₚ[mkₚ])
                for i in 1:length(mkₚ.mask)
                    mkₚ.mask[i] = false
                    @test model(ds[mk]) ≈ mₚ(dsₚ[mkₚ])
                    @test model(ds, mk) ≈ mₚ(dsₚ, mkₚ)
                    mkₚ.mask[i] = true
                end
            end
        end

        @testset "topmost bagmask" begin
            for mr in [mk.child, mk.child.mask]
                mₚ, dsₚ, mkₚ, keep = partialeval(model, ds, mk, [mr])
                @test mₚ.im.im.m == identity
                @test dsₚ.data.data isa ArrayNode
                @test mkₚ.child.child isa EmptyMask
                @test mkₚ.child.mask == mk.child.mask
                @test keep
                @test model(ds[mk]) ≈ mₚ(dsₚ[mkₚ])
                fv = FlatView(mkₚ)
                for i in 1:length(fv)
                    fv[i] = false
                    @test model(ds[mk]) ≈ mₚ(dsₚ[mkₚ])
                    @test model(ds, mk) ≈ mₚ(dsₚ, mkₚ)
                    fv[i] = true
                end
            end
        end

        @testset "bottom productmasks" begin
            for k in keys(mk.child.child)
                for mr in [mk.child.child[k], mk.child.child[k].mask]
                    mₚ, dsₚ, mkₚ, keep = partialeval(model, ds, mk, [mr])
                    for l in setdiff(keys(mk.child.child), [k])
                        @test mₚ.im.im[l].m == identity
                        @test dsₚ.data.data[l] isa ArrayNode
                        @test mkₚ.child.child[l] isa EmptyMask
                    end
                    @test mk.child.child[k].mask == mkₚ.child.child[k].mask
                    @test keep
                    @test model(ds[mk]) ≈ mₚ(dsₚ[mkₚ])
                    fv = FlatView(mkₚ)
                    for i in 1:length(fv)
                        fv[i] = false
                        @test model(ds[mk]) ≈ mₚ(dsₚ[mkₚ])
                        @test model(ds, mk) ≈ mₚ(dsₚ, mkₚ)
                        fv[i] = true
                    end
                end
            end
        end
    end
end
