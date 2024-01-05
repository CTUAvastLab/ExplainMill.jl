"""
    Absent
    
	represent a part of an explanation which is not important
"""
_retrieve_obs(::LazyNode, i) = error("LazyNode in Mill.jl does not support metadata (yet)")
_retrieve_obs(ds::ArrayNode{<:NGramMatrix, Nothing}, i) = ds.data.s[i]
_retrieve_obs(ds::ArrayNode{<:Flux.OneHotMatrix, Nothing}, i) = ds.data.data[i].ix
_retrieve_obs(ds::ArrayNode{<:Mill.MaybeHotMatrix, Nothing}, i) = ds.data.data[i].ix
_retrieve_obs(ds::ArrayNode{<:Matrix, Nothing}, i, j) = ds.data[i, j]
_retrieve_obs(ds::ArrayNode{<:AbstractMatrix, <:AbstractMatrix}, i, j) = ds.metadata[i, j]
_retrieve_obs(ds::ArrayNode{<:AbstractMatrix, <:AbstractVector}, j) = ds.metadata[j]
_reversedict(d) = Dict(map(reverse, collect(d)))

"""
	contributing(m)

	returns a mask of items contributing to the explanation
"""
contributing(m::AbstractStructureMask, _) = prunemask(m.mask)
contributing(m::EmptyMask, l) = fill(true, l)

function yarason(ds::ArrayNode{<:Union{Flux.OneHotMatrix, Mill.MaybeHotMatrix}, <:Any}, m::AbstractStructureMask, e::ExtractCategorical, exportobs=fill(true, numobs(ds)))
    c = contributing(m, numobs(ds))
    x = map(i -> c[i] ? _retrieve_obs(ds, i) : nothing, findall(exportobs))
    length(x) > 1 ? reduce(hcat, x) : x
end

function yarason(ds::ArrayNode{<:Matrix, <:Any}, m, e::ExtractScalar, exportobs=fill(true, numobs(ds)))
    rows, cols = size(ds.data)
    c = contributing(m, rows)
    x = map(findall(exportobs)) do j
        [c[i] ? _retrieve_obs(ds, i, j) : nothing for i in 1:rows]
    end
    hcat(x...)
end

function yarason(ds::ArrayNode{<:Matrix, <:AbstractVector{M}}, m, e::ExtractVector, exportobs=fill(true, numobs(ds))) where M <: AbstractVector
    c = contributing(m, numobs(ds))
    any(c) ? Float32[] : ds.data[c, :]
end

function yarason(ds::ArrayNode{<:NGramMatrix}, m, e::ExtractString, exportobs = fill(true, numobs(ds)))
    c = contributing(m, numobs(ds))
    x = map(i -> c[i] ? _retrieve_obs(ds, i) : nothing, findall(exportobs))
    hcat(x...)
end

function yarason(ds::LazyNode, m, e, exportobs=fill(true, numobs(ds)))
    c = contributing(m, numobs(ds))
    x = map(i -> c[i] ? _retrieve_obs(ds, i) : absent, findall(exportobs))
    addor(m, x, exportobs)
end


function yarason(ds::BagNode, mk::BagMask, e::JsonGrinder.ExtractArray, exportobs=fill(true, numobs(ds)))
    if !any(exportobs)
        return(nothing)
    end
    present_childs = present(mk.child, prunemask(mk.mask))
    for (i,b) in enumerate(ds.bags) 
        exportobs[i] && continue
        present_childs[b] .= false
    end
    x = yarason(ds.data, mk.child, e.item, present_childs)
    x === nothing && return(fill(nothing, sum(exportobs)))
    bags = Mill.adjustbags(ds.bags, present_childs)
    map(b -> x[b], bags[exportobs])
end

function yarason(ds::BagNode, mk, e::JsonGrinder.ExtractKeyAsField, exportobs=fill(true, numobs(ds)))
    if !any(exportobs)
        return(nothing)
    end
    present_childs = present(mk.child, prunemask(mk.mask))
    for (i,b) in enumerate(ds.bags) 
        exportobs[i] && continue
        present_childs[b] .= false
    end
    x = yarason(ds.data, mk.child, e, present_childs)
    x === nothing && return(fill(nothing, sum(exportobs)))
    bags = Mill.adjustbags(ds.bags, present_childs)
    map(bags[exportobs]) do b 
        isempty(b) && return(nothing)
        reduce(merge, x[b])
    end
end

function yarason(ds::ProductNode{T,M}, m, e::JsonGrinder.ExtractDict, exportobs=fill(true, numobs(ds))) where {T<:NamedTuple, M}
    S =  eltype(keys(e.dict))
    ks = sort(collect(intersect(keys(ds.data), Symbol.(keys(e.dict)))))
    s = map(ks) do k
        k => yarason(ds[k], m[k], e.dict[S(k)], exportobs)
    end
    s = filter(j -> j.second !== nothing && !isempty(j.second), s)
    soa2aos(s, exportobs)
end

function soa2aos(s, exportobs)
    map(1:sum(exportobs)) do i 
        ss = map(s) do (k,v)
            k => v[i]
        end 
        ss = filter(j -> !isnothing(j.second), ss)
        Dict(ss)
    end
end

function yarason(ds::ProductNode, m, e::JsonGrinder.MultipleRepresentation, exportobs=fill(true, numobs(ds)))
	numobs(ds) == 0 && return(zeroobs())
	!any(exportobs) && emptyexportobs()
	s = map(sort(collect(keys(ds.data)))) do k
        yarason(ds[k], m[k], e.extractors[k], exportobs)
    end
    reduce(mergeexplanations, s)
end

mergeexplanations(a, b) = map(x -> logicaland(x...), zip(a,b))
logicaland(a::Vector, b::Vector) = intersect(a,b)
logicaland(::Nothing, a) = a
logicaland(a, ::Nothing) = a
logicaland(a, b) = a
logicaland(::Nothing, ::Nothing) = nothing


function yarason(ds::ProductNode{T,M}, m, e::JsonGrinder.ExtractKeyAsField, exportobs=fill(true, numobs(ds))) where {T<:NamedTuple, M}
	k = yarason(ds[:key], m[:key],  e.key, exportobs)
	d = yarason(ds[:item], m[:item],  e.item, exportobs)
	kvpair(k, d)
end

kvpair(k::AbstractArray, v::AbstractArray) = map(x -> Dict(x[1] => x[2]), zip(k, v))
function kvpair(k::AbstractArray, v::Nothing)
    isempty(k) && return(nothing)
    map(x -> Dict(x => nothing), k)
end

function kvpair(k::Nothing, v::AbstractArray)
    isempty(v) && return(nothing)
    if length(v) > 1
        @info "cannot accurately restore due to keys being irrelevant"
    end
    return(map(x -> Dict(nothing => x), v))
end

kvpair(k::Nothing, v::Nothing) = nothing

"""
    prunejson

    remove empty arrays and nothings from the the explanation
"""
prunejson(::Nothing) = nothing 
prunejson(s) = s === nothing ? nothing : s
function prunejson(ss::Vector)
    ss = map(prunejson, ss)
    ss = filter(!isnothing, ss)
    isempty(ss) ? nothing : ss
end

function prunejson(d::Dict)
    ss = map(s -> s.first => prunejson(s.second), collect(d))
    ss = filter(s -> s.second !== nothing, ss)
    isempty(ss) ? nothing : Dict(ss)
end

function e2boolean(dss::AbstractMillNode, pruning_mask, extractor)
    js = yarason(dss, pruning_mask, extractor)
    js === nothing && return(nothing)
    numobs(dss) == 1 ? only(js) : js
end
