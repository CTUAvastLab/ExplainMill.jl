struct SparseArrayMask{M} <: AbstractListMask
	mask::M
	columns::Vector{Int}
end

Flux.@functor(SparseArrayMask)

function create_mask_structure(ds::ArrayNode{T,M}, m::ArrayModel, create_mask, cluster) where {T<:SparseMatrixCSC, M}
	nnz(ds.data) == 0 && return(EmptyMask())
	column2cluster = cluster(m, ds)
	columns = identifycolumns(ds.data)
	cluster_assignments = [column2cluster[c] for c in columns]
	SparseArrayMask(create_mask(cluster_assignments), columns)
end

function create_mask_structure(ds::ArrayNode{T,M}, create_mask) where {T<:SparseMatrixCSC, M}
	nnz(ds.data) == 0 && return(EmptyMask())
	columns = identifycolumns(ds.data)
	SparseArrayMask(create_mask(nnz(ds.data)), columns)
end

function identifycolumns(x::SparseMatrixCSC)
	columns = findall(!iszero, x);
	columns = [c.I[2] for c in columns]
end

function invalidate!(mk::SparseArrayMask, observations::Vector{Int})
	for (i,c) in enumerate(mk.columns)
		if c ∈ observations
			mk.mask.participate[i] = false
		end
	end
end

function Base.getindex(ds::ArrayNode{T,M}, m::SparseArrayMask, presentobs=fill(true,nobs(ds))) where {T<:Mill.SparseMatrixCSC, M}
	x = deepcopy(ds.data)
	x.nzval[.!prunemask(mk.mask)] .= 0
	ArrayNode(x[:,presentobs], ds.metadata)
end

function (m::Mill.ArrayModel)(ds::ArrayNode, mk::SparseArrayMask)
	x = ds.data
	xx = SparseMatrixCSC(x.m, x.n, x.colptr, x.rowval, x.nzval .* diffmask(mk.mask))
    ArrayNode(m.m(xx))
end

_nocluster(m::ArrayModel, ds::ArrayNode{T,M})  where {T<:SparseMatrixCSC, M} = unique(identifycolumns(ds.data))