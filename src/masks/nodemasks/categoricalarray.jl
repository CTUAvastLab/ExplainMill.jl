const OneHotFlux = ArrayNode{<:Flux.OneHotMatrix, <:Any}
const OneHotNode = ArrayNode{<:Mill.MaybeHotMatrix, <:Any}

struct CategoricalMask{M} <: AbstractListMask
	mask::M
end

Flux.@layer CategoricalMask

function create_mask_structure(ds::OneHotNode, m::ArrayModel, create_mask, cluster)
	cluster_assignments = cluster(m, ds)
	CategoricalMask(create_mask(cluster_assignments))
end

function create_mask_structure(ds::OneHotNode, create_mask)
	CategoricalMask(create_mask(numobs(ds.data)))
end

create_mask_structure(ds::OneHotFlux, m::ArrayModel, create_mask, cluster) = EmptyMask()
create_mask_structure(ds::OneHotFlux, create_mask) = EmptyMask()

function Base.getindex(ds::OneHotNode, mk::Union{ObservationMask,CategoricalMask}, presentobs=fill(true, numobs(ds)))
	pm = prunemask(mk.mask)
	nrows = size(ds.data, 1)
	ii = map(findall(presentobs)) do j
		i = ds.data.I[j]
		pm[j] ? i : missing
	end
	x = Mill.maybehotbatch(ii, 1:nrows)
	ArrayNode(x, ds.metadata)
end

function invalidate!(mk::CategoricalMask, observations::Vector{Int})
	invalidate!(mk.mask, observations)
end

function present(mk::CategoricalMask, obs)
	map((&), obs, prunemask(mk.mask))
end

function foreach_mask(f, m::CategoricalMask, level, visited)
	if !haskey(visited, m.mask)
		f(m.mask, level)
		visited[m.mask] = nothing
	end
end

function mapmask(f, m::CategoricalMask, level, visited)
	new_mask = get!(visited, m.mask, f(m.mask, level))
	CategoricalMask(new_mask)
end

#######
# This part is annoying. We need to handle at Dense layer and Chain
# but the first element of chain has to be Dense with PostImputingMatrix
# We wrap arguments (ds, mk) to Tuple, such that Chain is OK with it 
# and unwrap them once they hit the Dense layer
#######
function (m::Mill.ArrayModel)(ds::OneHotNode, mk::AbstractStructureMask)
	m.m((ds.data, mk))
end

function (m::Dense{<:Any, <:PostImputingMatrix,<:Any})(xmk::Tuple{<:MaybeHotMatrix,<:AbstractStructureMask})
	m(xmk...)
end

function (m::Dense{<:Any, <:PostImputingMatrix,<:Any})(x::MaybeHotMatrix, mk::AbstractStructureMask)
	W, b, σ = m.weight, m.bias, m.σ
	dm = reshape(diffmask(mk.mask), 1, :)
	y = W * x
	y = @. dm * y + (1 - dm) * W.ψ
	σ.(y .+ b)
end

_nocluster(m::ArrayModel, ds::OneHotNode) = numobs(ds.data)
