"""
struct Mask{I<:Union{Nothing, Vector{Int}}}
	mask::Array{Bool,1}
	participate::Array{Bool,1}
	outputid::Array{Int,1}
	daf::Daf
	cluster_membership::I
end

holds the structure for explaining 
mask --- a binary indicator vector, where each item corresponds to one explainable feature in the sample. 
	It is used to prune or modulate the sample. 

participate --- identifies if the corresponding item in the mask participate in the creation of modulated samples. 
	For example imagine to have two bags, one nested in the other. If we remove instance(s) in the top bag, then this 
	correnspods to whole bags in the bottom mil problem. Now whatever values the mask of these (removed bag) has, 
	it does not have any effect on the sample, as they are effectively removed. For these samples we set participate
	to zero, such that these items will not be counted in the statistics.

outputid --- this is a vector which identifies to which sample the corresponding item belongs. This is used to 
	speeedup the update of stats when multiple samples are used

daf --- Shappley value statistics for each item (cluster of items)

cluster_membership --- this identifies to which cluster the item belongs to. 
	This is created if clustering of items is on
"""
struct Mask{I<:Union{Nothing, Vector{Int}}}
	mask::Array{Bool,1}
	participate::Array{Bool,1}
	outputid::Array{Int,1}
	daf::Daf
	cluster_membership::I
end

participate(m::Mask) = m.participate
mask(m::Mask) = m.mask

Base.getindex(m::Mask{Nothing}, i::Int) = m.mask[i]
Base.getindex(m::Mask{Vector{Int}}, i::Int) = m.mask[m.cluster_membership .== i]
Base.setindex!(m::Mask{Nothing}, v, i::Int) = m.mask[i] = v
Base.setindex!(m::Mask{Vector{Int}}, v, i::Int) = m.mask[m.cluster_membership .== i] .= v

####
#	Explaination without clustering, where each item is independent of others
####
Mask(d::Int) = Mask(fill(true, d), fill(true, d), fill(0, d), Daf(d), nothing)

function StatsBase.sample!(m::Mask{Nothing})
	m.mask .= sample([true, false], length(m.mask))
end

function normalize_clusterids(x)
	n = length(unique(x))
	isempty(setdiff(1:n, unique(x))) && return(x)
	u = unique(x)
	k2id = Dict([u[i] => i for i in 1:length(u)])
	map(k -> k2id[k], x)
end
####
#	Explaination, where items are clustered together
####
function Mask(cluster_membership::Vector{Int}) 
	cluster_membership = normalize_clusterids(cluster_membership)
	n = length(unique(cluster_membership))
	!isempty(setdiff(1:n, unique(cluster_membership))) && @show cluster_membership
	d = length(cluster_membership)
	Mask(fill(true, d), fill(true, d), fill(0, d), Daf(n), cluster_membership)
end

_cluster_membership(ij::Vector{Int}, i) = ij[i]
_cluster_membership(ij::Nothing, i) = i

function Duff.update!(d::Mask, v::AbstractArray)
	s = d.daf
	for i in 1:length(d.mask)
		!d.participate[i] && continue
		f = v[d.outputid[i]]
		j = _cluster_membership(d.cluster_membership, i)
		Duff.update!(s, f, d.mask[i], j)
	end
end

function StatsBase.sample!(m::Mask{Vector{Int64}})
	ci = m.cluster_membership
	_mask = sample([true, false], maximum(ci))
	for (i,k) in enumerate(ci)
		m.mask[i] = _mask[k]
	end 
end
