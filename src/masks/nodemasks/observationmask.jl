"""
	`ObservationMask` is a mask that can be applied 
	to different type of nodes: Strings, Categorical Variables, Dense Arrays, etc.
	The length of wrapped mask should correspond to the number of observations in 
	the wrapped node
"""
struct ObservationMask{M} <: AbstractListMask
	mask::M
end

Flux.@layer ObservationMask

function invalidate!(mk::ObservationMask, invalid_observations::AbstractVector{Int})
	invalidate!(mk.mask, invalid_observations)
end

function present(mk::ObservationMask, obs)
	map((&), obs, prunemask(mk.mask))
end

function foreach_mask(f, m::ObservationMask, level, visited)
	if !haskey(visited, m.mask)
		f(m.mask, level)
		visited[m.mask] = nothing
	end
end

function foreach_mask(f, m::ObservationMask{<:FollowingMasks}, level, visited)
	for x in m.mask.masks 
		if !haskey(visited, x)
			f(x, level)
			visited[x] = nothing
		end
	end 
end

function mapmask(f, m::ObservationMask, level, visited)
	new_mask = get!(visited, m.mask, f(m.mask, level))
	ObservationMask(new_mask)
end

function mapmask(f, m::ObservationMask{<:FollowingMasks}, level, visited)
	mapped = map(m.mask.masks) do x 
		get!(visited, x, f(x, level))
	end 
	ObservationMask(FollowingMasks(mapped))
end


