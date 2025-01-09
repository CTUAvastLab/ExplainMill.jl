struct NGramMatrixMask{M} <: AbstractListMask
	mask::M
end

const NGramNode = ArrayNode{<:Mill.NGramMatrix,<:Any}

Flux.@layer NGramMatrixMask

function create_mask_structure(ds::NGramNode, m::ArrayModel, create_mask, cluster)
	cluster_assignments = cluster(m, ds)
	NGramMatrixMask(create_mask(cluster_assignments))
end

function create_mask_structure(ds::NGramNode, create_mask)
	NGramMatrixMask(create_mask(numobs(ds.data)))
end

function invalidate!(mk::NGramMatrixMask, invalid_observations::AbstractVector{Int})
	invalidate!(mk.mask, invalid_observations)
end

function Base.getindex(ds::NGramNode, mk::Union{ObservationMask,NGramMatrixMask}, presentobs=fill(true, numobs(ds)))
	x = ds.data
	pm = prunemask(mk.mask) 
	s = map(findall(presentobs)) do i 
		pm[i] ? x.S[i] : missing
	end
	ArrayNode(NGramMatrix(s, x.n, x.b, x.m), ds.metadata)
end

function present(mk::NGramMatrixMask, obs)
	map((&), obs, prunemask(mk.mask))
end

function foreach_mask(f, m::NGramMatrixMask, level, visited)
	if !haskey(visited, m.mask)
		f(m.mask, level)
		visited[m.mask] = nothing
	end
end

function mapmask(f, m::NGramMatrixMask, level, visited)
	new_mask = get!(visited, m.mask, f(m.mask, level))
	NGramMatrixMask(new_mask)
end

function (m::Mill.ArrayModel)(ds::NGramNode, mk::Union{ObservationMask,NGramMatrixMask})
	m.m((ds.data, mk))
end

function (m::Dense{<:Any, <:PostImputingMatrix,<:Any})(xmk::Tuple{<:NGramMatrix,<:AbstractStructureMask})
	m(xmk...)
end

function (m::Dense{<:Any, <:PostImputingMatrix,<:Any})(x::NGramMatrix, mk::AbstractStructureMask)
	W, b, σ = m.weight, m.bias, m.σ
	dm = reshape(diffmask(mk.mask), 1, :)
	y = W * x
	y = @. dm * y + (1 - dm) * W.ψ
	σ.(y .+ b)
end



_nocluster(m::ArrayModel, ds::NGramNode) = numobs(ds.data)
