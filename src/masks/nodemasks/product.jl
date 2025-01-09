struct ProductMask{C} <: AbstractNoMask
	childs::C
end

Flux.@layer ProductMask

Base.getindex(m::ProductMask, i::Symbol) = m.childs[i]
Base.getindex(m::ProductMask, i::Int) = m.childs[i]
Base.keys(m::ProductMask) = keys(m.childs)

mask(::ProductMask) = nothing
participate(::ProductMask) = nothing

function create_mask_structure(ds::ProductNode{T,M}, m::ProductModel, create_mask, cluster) where {T<:NamedTuple,M}
	ks = keys(ds.data)
	s = (;[k => create_mask_structure(ds.data[k], m.ms[k], create_mask, cluster) for k in ks]...)
	ProductMask(s)
end

function create_mask_structure(ds::ProductNode{T,M}, create_mask) where {T<:NamedTuple,M}
	ks = keys(ds.data)
	s = (;[k => create_mask_structure(ds.data[k], create_mask) for k in ks]...)
	ProductMask(s)
end

function foreach_mask(f, mk::ProductMask, level, visited)
	foreach(m -> foreach_mask(f, m, level, visited), mk.childs)
end

function mapmask(f, mk::ProductMask, level, visited)
	ProductMask(map(m -> mapmask(f, m, level, visited), mk.childs))
end

function present(mk::ProductMask, obs)
	mapreduce(m -> present(m, obs), (a,b) -> max.(a,b), mk.childs)
end
# function present(mk::ProductMask, obs)
# 	mapreduce(m -> present(m, obs), (a,b) -> max.(a,b), mk.childs)
# 	@show obs
# 	mapreduce((a,b) -> max.(a,b), mk.childs) do m 
# 		@show present(m, obs)
# 		present(m, obs)
# 	end
# end

function invalidate!(mk::ProductMask, observations::Vector{Int})
	for c in mk.childs
		invalidate!(c, observations)
	end
end

function Base.getindex(ds::ProductNode{T,M}, mk::ProductMask, presentobs=fill(true, numobs(ds))) where {T<:NamedTuple, M}
	s = map(ds.data, mk.childs) do sub_ds, sub_mk
       sub_ds[sub_mk, presentobs]
	end
	ProductNode(s)
end

function Base.getindex(ds::ProductNode{T,M}, mk::ProductMask, presentobs=fill(true, numobs(ds))) where {T<:Tuple, M}
	s = tuple([ds.data[k][mk.childs[k], presentobs] for k in 1:length(ds.data)]...)
	ProductNode(s)
end

function (m::Mill.ProductModel{MS,M})(x::ProductNode{P,T}, mk::ProductMask) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
    xx = vcat([m[k](x[k], mk[k]) for k in keys(m.ms)]...)
    m.m(xx)
end

function (m::Mill.ProductModel{MS,M})(x::ProductNode{P,T}, mk::ProductMask) where {P<:Tuple,T,MS<:Tuple, M} 
    xx = vcat([m.ms[k](x.data[k], mk.childs[k]) for k in 1:length(m.ms)]...)
    m.m(xx)
end

function partialeval(model::ProductModel{MS,M}, ds::ProductNode{P,T}, mk::ProductMask, masks) where {P<:NamedTuple,T,MS<:NamedTuple, M} 
	ks = keys(model.ms)
	mods = map(ks) do k
		partialeval(model.ms[k], ds.data[k], mk[k], masks)
	end
	childmodels = map(f -> f[1], mods)
	childds = map(f -> f[2], mods)
	childms = map(f -> f[3], mods)
	if any(f[4] for f in mods)
		return(ProductModel((;zip(ks, childmodels)...), model.m), ProductNode((;zip(ks, childds)...), ds.metadata), ProductMask((;zip(ks, childms)...)), true)
	end
	@assert all(cm isa ArrayModel{typeof(identity)} for cm in childmodels)
	x = model.m(reduce(vcat, map((cd, cm) -> cd[cm].data, childds, childms)))
	return(ArrayModel(identity), ArrayNode(x), EmptyMask(), false)
end

function partialeval(model::ProductModel, ds::ProductNode, mk::EmptyMask, masks)
	return(ArrayModel(identity), model.m(vcat(childds...)), EmptyMask(), false)
end

_nocluster(m::ProductModel, ds::ProductNode) = numobs(ds)

prunemask(m::ProductMask) = nothing
