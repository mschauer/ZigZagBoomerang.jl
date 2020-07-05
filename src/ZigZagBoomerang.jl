module ZigZagBoomerang


# ZigZag1d and Boomerang1d reference implementation
include("types.jl")
include("common.jl")
include("dynamics.jl")
export ZigZag1d, Boomerang1d


include("poissontime.jl")
export poisson_time

include("localzigzag.jl")
export LocalZigZag

include("zigzagboom1d.jl")
export pdmp, eventtime, eventposition


include("trace.jl")
include("discretize.jl")
export discretize


end
