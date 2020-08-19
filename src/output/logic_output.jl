using FillArrays
OR(xs) = length(xs) > 1 ? Dict(:or => filter(!ismissing, xs)) : only(xs)

"""
	addor(m::Mask, x)

	returns "or" relationships if they are needed in the explanation, by 
	substituing each item with an "or" of items in clusters

	`m` is the mask and 
	`x` is the output of the explanation of bottom layers
"""
function addor(m::Mask{I, D}, x, active) where {I<:Vector{Int}, D}
	xi = m.cluster_membership[active]
	map(1:length(x)) do i
		ismissing(x[i]) ? missing : OR(x[xi .== xi[i]])
	end
end

addor(m::Mask{I, D}, x::Missing, active)  where {I<: Vector{Int}, D}= missing
addor(m::Mask{I, D}, x, active) where {I<: Nothing, D} = x
addor(m::Mask{I, D}, x::Missing, active) where {I<: Nothing, D}  = missing
addor(m::AbstractExplainMask, x, active) = addor(m.mask, x, active)
addor(m::EmptyMask, x, active) = x


"""
	contributing(m)
	
	returns a mask of items contributing to the explanation
"""
contributing(m::AbstractExplainMask, l) = participate(m) .& prunemask(m)
contributing(m::EmptyMask, l) = Fill(true, l)

function yarason(ds::ArrayNode{T}, m::AbstractExplainMask, e::ExtractCategorical, exportobs = fill(true, nobs(ds))) where {T<:Flux.OneHotMatrix}
	c = contributing(m, nobs(ds))
	items = map(i -> i.ix, ds.data.data)
	!any(c) && return(fill(missing, sum(exportobs)))
	d = reversedict(e.keyvalemap);
	idxs = map(i -> c[i] ? get(d, items[i], "__UNKNOWN__") : missing, findall(exportobs))
	addor(m, idxs, exportobs)
end

function yarason(ds::ArrayNode{T}, m::AbstractExplainMask,  e, exportobs = fill(true, nobs(ds))) where {T<:Matrix}
	items = contributing(m, size(ds.data,1))
	x = map(findall(exportobs)) do j 
		[items[i] ? ds.data[i,j] : missing for i in 1:length(items)]
	end
	x
end

function yarason(ds::ArrayNode{T}, m::AbstractExplainMask, e, exportobs = fill(true, nobs(ds))) where {T<:Mill.NGramMatrix}
	c =  contributing(m, nobs(ds))
	x = map(i -> c[i] ? ds.data.s[i] : missing, findall(exportobs))
	addor(m, x, exportobs)
end

function yarason(ds::LazyNode, m::AbstractExplainMask, e, exportobs = fill(true, nobs(ds)))
	c =  contributing(m, nobs(ds))
	x = map(i -> c[i] ? ds.data[i] : missing, findall(exportobs))
	addor(m, x, exportobs)
end

# This hack is needed for cases, where scalars are joined to a single matrix
# function yarason(m::Mask, ds::ArrayNode, e::ExtractDict)
# 	ks = keys(e.vec)
# 	s = join(map(i -> "$(i[1]) = $(i[2])" , zip(ks, ds.data[:])))
# 	repr_boolean(:and, unique(s))
# end

function yarason(ds::BagNode, m, e::ExtractArray, exportobs = fill(true, nobs(ds)))
    ismissing(ds.data) && return(fill(missing, sum(exportobs)))
    nobs(ds.data) == 0 && return(fill(missing, sum(exportobs)))

    #get indexes of c clusters
	present_childs = Vector(contributing(m, nobs(ds.data)))
	for b in ds.bags[.!exportobs]
	    present_childs[b] .= false
	end

	x = yarason(ds.data, m.child, e.item, present_childs)
	x = addor(m, x, present_childs)
	bags = Mill.adjustbags(ds.bags, present_childs)[exportobs]
	map(b -> x[b], bags)
end

# function yarason(ds::BagNode, m::AbstractExplainMask, e::JsonGrinder.ExtractKeyAsField)
#     ismissing(ds.data) && return(missing)

#     #get indexes of c clusters
# 	c = contributing(m, nobs(ds.data))
# 	all(.!c) && return(missing)
# 	ss = yarason(ds.data, m.child, e.item)
# 	addor(m, ss, c)
# end

# function yarason(m::EmptyMask, ds::BagNode, e)
#     ismissing(ds.data) && return(missing)
#     nobs(ds.data) == 0 && return(missing)
#     yarason(ds.data, m, e.item);
# end

function yarason(ds::ProductNode{T,M}, m::AbstractExplainMask, e) where {T<:NamedTuple, M}
	nobs(ds) == 0 && return(missing)
	s = map(sort(collect(keys(ds.data)))) do k
        subs = yarason(m[k], ds[k], e[k])
        isempty(subs) ? nothing : k => subs
    end
    Dict(filter(!isnothing, s))
end

function yarason(ds::ProductNode{T,M}, m::AbstractExplainMask, e::JsonGrinder.ExtractKeyAsField) where {T<:NamedTuple, M}
	nobs(ds) == 0 && return(missing)
	(Dict(
		Symbol(yarason(ds[:key], m[:key],  e.key)) => yarason(ds[:item], m[:item], e.item),
	))
end

# function yarason(ds::ProductNode{T,M}, m::AbstractExplainMask, e::MultipleRepresentation) where {T<:Tuple, M}
# 	nobs(ds) == 0 && return(missing)
# 	s = map(sort(collect(keys(ds.data)))) do k
#         subs = yarason(m.childs[k], ds.data[k], e.extractors[k])
#         isempty(subs) ? nothing : k => subs
#     end
#     Dict(filter(!isnothing, s))
# end

# function e2boolean(pruning_mask, dss, extractor)
# 	d = map(1:nobs(dss)) do i 
# 		mapmask(pruning_mask) do m 
# 			participate(m) .= true
# 		end
# 		invalidate!(pruning_mask,setdiff(1:nobs(dss), i))
# 		repr(MIME("text/json"),pruning_mask, dss, extractor);
# 	end
# 	d = filter(!isempty, d)
# 	isempty(d) && return([Dict{Symbol,Any}()])
# 	d = unique(d)
# 	d = length(d) > 1 ? ExplainMill.repr_boolean(:or, d) : d
# end
