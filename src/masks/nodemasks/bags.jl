struct BagMask{C,B,M} <: AbstractStructureMask
	child::C
	bags::B
	mask::M
end

Flux.@layer BagMask

function create_mask_structure(ds::BagNode, model::BagModel, create_mask, cluster)
	child_mask = create_mask_structure(ds.data, model.im, create_mask, cluster)
	cluster_assignments = cluster(model, ds)
	BagMask(child_mask, ds.bags, create_mask(cluster_assignments))
end

function create_mask_structure(ds::BagNode, create_mask)
	child_mask = create_mask_structure(ds.data, create_mask)
	BagMask(child_mask, ds.bags, create_mask(numobs(ds.data)))
end

function Base.getindex(m::BagMask, i::Mill.VecOrRange)
    nb, ii = Mill.remapbag(m.bags, i)
    isempty(ii) && return(EmptyMask())
    BagNode(m.child[i], nb, m.mask[ii])
end

function foreach_mask(f, m::BagMask, level, visited)
	if !haskey(visited, m.mask)
		f(m.mask, level)
		foreach_mask(f, m.child, level + 1, visited)
		visited[m.mask] = nothing
	end
end

function mapmask(f, m::BagMask, level, visited)
	new_mask = get!(visited, m.mask, f(m.mask, level))
	BagMask(mapmask(f, m.child, level + 1, visited),
		m.bags,
		new_mask
		)
end

function invalidate!(mk::BagMask, invalid_observations::AbstractVector{Int})
	invalid_instances = isempty(invalid_observations) ? invalid_observations : reduce(vcat, [collect(mk.bags[i]) for i in invalid_observations])
	invalidate!(mk.mask, invalid_instances)
	invalid_instances = unique(vcat(invalid_instances, findall(.!(prunemask(mk.mask) .& participate(mk.mask)))))
	invalidate!(mk.child, invalid_instances)
end

function present(mk::BagMask, obs)
	child_obs = present(mk.child, prunemask(mk.mask))
	map(enumerate(mk.bags)) do (i, b) 
		obs[i] & any(@view child_obs[b])
	end
end

function Base.getindex(ds::BagNode, mk::BagMask, presentobs=fill(true, numobs(ds)))
	if !any(presentobs)
		return(ds[0:-1])
	end
	# @show presentobs
	# @show prunemask(mk.mask)
	present_childs = present(mk.child, prunemask(mk.mask))
	# @show present_childs
	for (i,b) in enumerate(ds.bags) 
	    presentobs[i] && continue
	    present_childs[b] .= false
	end
	x = ds.data[mk.child, present_childs]
	bags = Mill.adjustbags(ds.bags, present_childs)
	if ismissing(x.data)
		bags.bags .= [0:-1]
	end
	BagNode(x, bags[presentobs])
end

function (model::Mill.BagModel)(x::BagNode, mk::BagMask)
	ismissing(x.data) && return(model.bm(ArrayNode(model.a(x.data, x.bags))))
	xx = model.im(x.data, mk.child)
    model.bm(model.a(xx, x.bags, mk))
end

#TODO: SimpleMask for now, but we should add a proper abstract

function (bc::Mill.BagCount)(x::Mill.Maybe{AbstractArray}, bags::Mill.AbstractBags, mk::BagMask)
	present_childs = ChainRulesCore.@ignore_derivatives present(mk.child, prunemask(mk.mask))
    o1 = bc.a(x, bags, mk)
    T = eltype(o1)
    o2 = Mill.segmented_sum_forw(transpose(T.(present_childs)), ones(T, 1), bags, nothing)
    o2 = log.(one(T) .+ o2)
    vcat(o1, o2)
end

function (a::Mill.SegmentedMax)(x::Matrix, bags::Mill.AbstractBags, mk::BagMask)
	present_childs = ChainRulesCore.@ignore_derivatives present(mk.child, prunemask(mk.mask))
	m = transpose(diffmask(mk.mask) .* present_childs)
	xx = m .* x .+ (1 .- m) .* a.ψ 
	a(xx, bags)
end	

# TODO: This might be done better
function (a::Mill.SegmentedMean)(x::Matrix{T}, bags::Mill.AbstractBags, mk::BagMask) where {T}
	present_childs = ChainRulesCore.@ignore_derivatives present(mk.child, prunemask(mk.mask))
	m = T.(transpose(diffmask(mk.mask) .* present_childs))
	xx = m .* x
	o = a(xx, bags)
	n = max.(Mill.segmented_mean_forw(m, [eps(T)], bags, nothing), eps(T))
	o ./ n
end	

function (m::Mill.BagModel)(x::BagNode, mk::EmptyMask)
	m(x)
end

function partialeval(model::BagModel, ds::BagNode, mk::BagMask, masks)
	im, ids, childms, keep = partialeval(model.im, ds.data, mk.child, masks)
	if (mk ∈ masks) | (mk.mask ∈ masks) | keep
		return(BagModel(im, model.a,  model.bm), BagNode(ids, ds.bags, ds.metadata), BagMask(childms, mk.bags, mk.mask), true)
	end
	present_childs = present(mk.child, prunemask(mk.mask))
	x = ids[present_childs].data
	nb = Mill.adjustbags(ds.bags, present_childs)
	return(ArrayModel(identity), ArrayNode(model.bm(model.a(x, nb))), EmptyMask(), false)
end

function partialeval(model::BagModel, ds::BagNode, mk::EmptyMask, masks)
	return(ArrayModel(identity), model(ds), EmptyMask(), false)
end

_nocluster(m::BagModel, ds::BagNode) = numobs(ds.data)
