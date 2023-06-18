####
# Unit test for extraction of logical formulas and their matching
####

# The idea behind the test is to take the sample, mask, and the extractor,
# recreate JSON from it. Then, extract this json to Mill, pass it through 
# the model. This should be equal to the original masked sample. 
# yarason(ds, mk, ex) |> ex |> model ≈ model(ds[mk])
# 
# 
@testset "logical output" begin 
	@testset "ExtractScalar" begin
		e = ExtractScalar(Float32, 0, 1)
		x = [1, 2, 3, 4, 5]
		ds = reduce(catobs, map(i -> e(i, store_input = true), x))
		model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing = true)
		randn!(model.m.weight.ψ)
		for (b, v) in [(true, x), (false, fill(nothing, length(x)))]
			mk = create_mask_structure(ds, d -> SimpleMask(rand([b], d)))
			rx = yarason(ds, mk, e)
			@test isequal(rx[:], v)
			dr = reduce(catobs, map(e, rx))
			@test isequal(ds[mk].data , dr.data)
			@test model(ds[mk]) ≈ model(dr)
		end

		mk = create_mask_structure(ds, d -> SimpleMask(trues(d)))
		present_childs = Bool[0, 1, 0, 0, 0] 
		@test yarason(ds, mk, e, present_childs) == [2;;]
		present_childs = Bool[0, 1, 0, 0, 1] 
		@test yarason(ds, mk, e, present_childs) == [2 5]
	end

	@testset "ExtractString" begin
		e = ExtractString()
		x = ["a","b","c","d","e"]
		ds = reduce(catobs, map(i -> e(i, store_input = true), x))
		model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing = true)
		randn!(model.m.weight.ψ)
		mk = create_mask_structure(ds, d -> SimpleMask(rand([true, false], d)))
		for i in 1:10
			rand!(mk.mask.x)
			rx = yarason(ds, mk, e)
			dr = reduce(catobs, map(e, rx))
			@test isequal(ds[mk].data , dr.data)
			@test model(ds[mk]) ≈ model(dr)
		end
		mk = create_mask_structure(ds, d -> SimpleMask(trues(d)))
		present_childs = Bool[0, 1, 0, 0, 0] 
		@test yarason(ds, mk, e, present_childs) == ["b";;]
		present_childs = Bool[0, 1, 0, 0, 1] 
		@test yarason(ds, mk, e, present_childs) == ["b" "e"]
	end

	@testset "ExtractCategorical" begin
		x = ["a","b","c","d","e"]
		e = ExtractCategorical(["a","b","c","d"])
		ds = reduce(catobs, map(i -> e(i, store_input = true), x))
		model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing = true)
		randn!(model.m.weight.ψ)
		mk = create_mask_structure(ds, d -> SimpleMask(rand([true, false], d)))
		for i in 1:10
			rand!(mk.mask.x)
			rx = yarason(ds, mk, e)
			dr = reduce(catobs, map(e, rx))
			@test isequal(ds[mk].data , dr.data)
			@test model(ds[mk]) ≈ model(dr)
		end

		mk = create_mask_structure(ds, d -> SimpleMask(trues(d)))
		present_childs = Bool[0, 1, 0, 0, 0] 
		@test yarason(ds, mk, e, present_childs) == ["b";]
		present_childs = Bool[0, 1, 0, 0, 1] 
		@test yarason(ds, mk, e, present_childs) == ["b" "e"]
	end

	@testset "Product" begin
		e = ExtractDict(Dict(:a => ExtractCategorical(["ca","cb","cc","cd"]), 
			:b => ExtractString()))
		js = map(x -> Dict(:a => x[1], :b => x[2]), zip(["ca","cb","cc","cd","ce"], ["sa","sb","sc","sd","se"]))
		ds = reduce(catobs, map(i -> e(i, store_input = true), js))
		mk = create_mask_structure(ds, d -> SimpleMask(rand([true, false], d)))
		model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing = true)
		for i in 1:10
			rand!(mk.childs[:a].mask.x)
			rand!(mk.childs[:b].mask.x)
			rx = yarason(ds, mk, e)
			dr = reduce(catobs, map(e, rx))
			@test model(ds[mk]) ≈ model(dr)
		end

		mk.childs[:a].mask.x .= false
		mk.childs[:b].mask.x .= true
		rx = yarason(ds, mk, e)
		dr = reduce(catobs, map(e, rx))
		@test model(ds[mk]) ≈ model(dr)
		
		mk.childs[:a].mask.x .= true
		mk.childs[:a].mask.x .= false
		rx = yarason(ds, mk, e)
		dr = reduce(catobs, map(e, rx))
		@test model(ds[mk]) ≈ model(dr)
	end

	@testset "Simple Bags" begin
		e = ExtractArray(ExtractCategorical(["a","b","c","d"]))
		ds = reduce(catobs, map(i -> e(i, store_input = true), [["a","b"],["c"],["d","e"]]))
		mk = create_mask_structure(ds, d -> SimpleMask(rand([true, false], d)))
		model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing = true)
		randn!(model.im.m.weight.ψ)
		for i in 1:10
			rand!(mk.mask.x)
			rand!(mk.child.mask.x)
			rx = yarason(ds, mk, e)
			dr = reduce(catobs, map(e, rx))
			@test isequal(ds[mk].data.data.I, dr.data.data.I)
			@test model(ds[mk]) ≈ model(dr)
		end

		mk.mask.x .= 0
		mk.child.mask.x .= 0
		rx = yarason(ds, mk, e)
		dr = reduce(catobs, map(e, rx))
		@test isequal(ds[mk].data.data.I, dr.data.data.I)
		@test model(ds[mk]) ≈ model(dr)
	end

	@testset "Bags of Bags" begin
		e = ExtractArray(ExtractArray(ExtractCategorical(["a","b","c","d"])))
		ds = reduce(catobs, map(i -> e(i, store_input = true), [[["a","b"],["c"]],[["d","e"]]]))
		mk = create_mask_structure(ds, d -> SimpleMask(rand([true, false], d)))
		model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing = true)
		randn!(model.im.im.m.weight.ψ)
		randn!(model.im.a.a[1].ψ)
		randn!(model.im.a.a[2].ψ)
		randn!(model.a.a[1].ψ)
		randn!(model.a.a[2].ψ)
		for i in 1:10
			rand!(mk.mask.x)
			rand!(mk.child.mask.x)
			rand!(mk.child.child.mask.x)
			rx = yarason(ds, mk, e)
			dr = reduce(catobs, map(e, rx))
			@test isequal(ds[mk].data.data.data.I, dr.data.data.data.I)
			@test model(ds[mk]) ≈ model(dr)
		end
	end

	@testset "ExtractKeyAsField" begin
		e = JsonGrinder.ExtractKeyAsField(
			JsonGrinder.ExtractString(),
			JsonGrinder.ExtractArray(
				JsonGrinder.ExtractString()),
		)
		s = [Dict(
			"ka" => ["a1", "a2", "a3"],
			),
			Dict(
			"kc" => ["c1"],
			"ka" => ["a1", "a2"],
			),]
		
		ds = reduce(catobs, map(i -> e(i, store_input = true), s))
		mk = create_mask_structure(ds, d -> SimpleMask(trues(d)))
		model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing = true)
		randn!(model.im[:key].m.weight.ψ)
		randn!(model.im[:item].im.m.weight.ψ)
		randn!(model.im[:item].a.a[1].ψ)
		randn!(model.im[:item].a.a[2].ψ)
		randn!(model.a.a[1].ψ)
		randn!(model.a.a[2].ψ)
		for i in 1:20
			rand!(mk.mask.x)
			rand!(mk.child[:key].mask.x)
			rand!(mk.child[:item].mask.x)
			rand!(mk.child[:item].child.mask.x)
			rx = yarason(ds, mk, e)
			dr = reduce(catobs, map(e, rx))
			@test model(ds[mk]) ≈ model(dr)
		end
	end 

	@testset "MultipleRepresentation" begin
		e = MultipleRepresentation(
			(ExtractCategorical(["a", "b","c","d"]),
			JsonGrinder.ExtractString())
			)

		s = ["a", "b", "c", "d", "e"]
		ds = reduce(catobs, map(i -> e(i, store_input = true), s))
		mk = create_mask_structure(ds, d -> SimpleMask(trues(d)))
		model = reflectinmodel(ds, d -> Dense(d, 4), all_imputing = true)
		randn!(model[:e1].m.weight.ψ)
		randn!(model[:e2].m.weight.ψ)
		for i in 1:10
			rand!(mk[:e1].mask.x)
			mk[:e1].mask.x .= mk[:e2].mask.x
			rx = yarason(ds, mk, e)
			dr = reduce(catobs, map(e, rx))
			@test model(ds[mk]) ≈ model(dr)
		end
	end
end
