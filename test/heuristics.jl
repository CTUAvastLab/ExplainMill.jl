@testset "correctness of heuristic functions" begin 
	ds = specimen_sample()
	model = f64(reflectinmodel(ds, d -> Dense(d, 4), SegmentedMean, all_imputing = true))

	@testset "ConstExplainer" begin
		mk = stats(ConstExplainer(), ds, model)
		h = heuristic(FlatView(mk))
		@test all(h .== 1)
	end

	@testset "StochasticExplainer" begin
		mk = stats(StochasticExplainer(), ds, model)
		h₁ = heuristic(FlatView(mk))
		@test length(unique(h₁)) == length(FlatView(mk))

		mk = stats(StochasticExplainer(), ds, model)
		h₂ = heuristic(FlatView(mk))
		@test all(h₁ .!= h₂)
	end

	@testset "GradExplainer" begin
		mk = create_mask_structure(ds, d -> SimpleMask(ones(Float64, d)))
		fv = FlatView(mk)
		y = ExplainMill.gnntarget(model, ds)
		f = x -> begin 
			fv .= x 
			sum(softmax(model(ds, mk)) .* y)
		end 
		hᵣ = grad(central_fdm(5,1), f , ones(Float64, length(fv)))[1]
		ps = Flux.Params(map(x -> x.x, fv.masks))
		foreach(x -> x .= 1, ps)
		gs = gradient(() -> sum(softmax(model(ds, mk)) .* y), ps)

		h = stats(GradExplainer(true), ds, model) |> FlatView |> heuristic
		@test h ≈ abs.(hᵣ)

		h = stats(GradExplainer(false), ds, model) |> FlatView |> heuristic
		@test h ≈ abs.(hᵣ)
	end

	@testset "BanzExplainer" begin
		#this is just to check repeatability befor the second test
		Random.seed!(1)
		h₁ = stats(DafExplainer(200), ds, model) |> FlatView |> heuristic
		Random.seed!(1)
		h₂ = stats(DafExplainer(200), ds, model) |> FlatView |> heuristic
		@test h₁ ≈ h₂

		# This is kind of awkward, as it is reimplementation 
		# of DAF score using FlatView mask. Not sure if it make 
		# much sense as a unit-test, but at least it checks that
		# values are corresponding
		Random.seed!(1)
		e = DafExplainer(0)
		mk = stats(e, ds, model)
		fv = FlatView(mk)
		y = ExplainMill.gnntarget(model, ds)
		s = Duff.Daf(length(fv))
		for i in 1:200
			sample!(mk)
			updateparticipation!(mk)
			o = sum(softmax(model(ds[mk])) .* y)
			for j in 1:length(fv)
				!e.banzhaf && !participate(fv)[j] && continue
				Duff.update!(s, o, fv[j] & participate(fv)[j], j)
			end
		end
		@test Duff.meanscore(s) ≈ h₂
	end

	@testset "GnnExplainer" begin
		@test_broken false
	end
end
