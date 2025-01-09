"""
	struct FeatureMask{M} <: AbstractListMask
		mask::M
		rows::Int
		cols::Int
	end


	Explaining individual items of Dense matrices. Each item within the matrix has its own mask.
	It is assumed and 

"""
struct FeatureMask{M} <: AbstractListMask
	mask::M
	rows::Int
	cols::Int
end

Flux.@layer FeatureMask

function create_mask_structure(ds::ArrayNode{T,M}, m::ArrayModel, create_mask, cluster) where {T<:Matrix, M} 
	create_mask_structure(ds, create_mask)
end

function create_mask_structure(ds::ArrayNode{T,M}, create_mask) where {T<:Matrix, M} 
	FeatureMask(create_mask(size(ds.data, 1)), size(ds.data)...)
end

function Base.getindex(ds::ArrayNode{T,M}, mk::FeatureMask, presentobs=fill(true, numobs(ds))) where {T<:Matrix, M}
	x = ds.data[:,presentobs]
	if eltype(x) <: Real
		x = Matrix{Union{Missing, eltype(x)}}(x)
	end
	x[.!prunemask(mk.mask), :] .= missing
	ArrayNode(x, ds.metadata)
end

function Base.getindex(ds::ArrayNode{T,M}, mk::ObservationMask, presentobs=fill(true, numobs(ds))) where {T<:Matrix, M}
	x = ds.data[:, presentobs]
	if eltype(x) <: Real
		x = Matrix{Union{Missing, eltype(x)}}(x)
	end
	x[:, (.!prunemask(mk.mask))[presentobs]] .= missing
	ArrayNode(x, ds.metadata)
end

function foreach_mask(f, m::FeatureMask, level, visited)
	if !haskey(visited, m.mask)
		f(m.mask, level)
		visited[m.mask] = nothing
	end
end

function mapmask(f, m::FeatureMask, level, visited)
	new_mask = get!(visited, m.mask, f(m.mask, level))
	FeatureMask(new_mask, m.rows, m.cols)
end

function invalidate!(mk::FeatureMask, observations::Vector{Int})
	
end

function present(mk::FeatureMask, obs)
	any(prunemask(mk.mask)) && return(obs)
	fill(false, length(obs))
end

function (m::Mill.ArrayModel)(ds::ArrayNode{<:Matrix,<:Any}, mk::FeatureMask)
    m.m(diffmask(mk.mask) .* ds.data)
end

function (m::Mill.ArrayModel)(ds::ArrayNode{<:Matrix,<:Any}, mk::ObservationMask)
    m.m(transpose(diffmask(mk.mask)) .* ds.data)
end

function (m::Mill.ArrayModel)(ds::Matrix, mk::AbstractStructureMask)
	m.m((ds.data, mk))
end

function (m::Dense{<:Any, <:PreImputingMatrix,<:Any})(xmk::Tuple{<:Matrix,<:AbstractStructureMask})
	m(xmk...)
end

function (m::Dense{<:Any, <:PreImputingMatrix,<:Any})(x::Matrix, mk::FeatureMask)
	W, b, σ = m.weight, m.bias, m.σ
	dm = diffmask(mk.mask)
	y = W * x
	y = @. dm * y + (1 - dm) * W.ψ
	σ.(y .+ b)
end

function (m::Dense{<:Any, <:PreImputingMatrix,<:Any})(x::Matrix, mk::ObservationMask)
	W, b, σ = m.weight, m.bias, m.σ
	dm = reshape(diffmask(mk.mask), 1, :)
	y = W * x
	y = @. dm * y + (1 - dm) * W.ψ
	σ.(y .+ b)
end

"""
	partialeval(model::AbstracModel, ds::AbstractMillNode, mk::StructureMask, masks)

	identify subset of `model`, sample `ds`, and structural mask `mk` that are 
	sensitive to `masks` and evaluate and replace the rest of the model, sample, 
	and masks by `identity`, `ArrayNode`, and `EmptyMask`. Partial evaluation 
	is useful when we are explaining only subset of full mask (e.g. level-by-level) 
	explanation.
"""
function partialeval(model::M, ds::ArrayNode, mk, masks) where {M<:ArrayModel}
	mk ∈ masks && return(model, ds, mk, true)
	mk.mask ∈ masks && return(model, ds, mk, true)
	return(ArrayModel(identity), ArrayNode(model(ds[mk])), EmptyMask(), false)
end

_nocluster(::ArrayModel, ds::ArrayNode{T,M}) where {T<:Matrix, M} = size(ds.data, 2)
