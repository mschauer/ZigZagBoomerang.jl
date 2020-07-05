module ZigZagBoomerang


# ZigZag1d and Boomerang1d reference implementation
include("types.jl")
include("common.jl")
include("dynamics.jl")
export ZigZag1d, Boomerang1d, ZigZag, FactBoomerang


include("poissontime.jl")
export poisson_time

include("fact_samplers.jl")
export ZigZag

include("zigzagboom1d.jl")
export pdmp, eventtime, eventposition


include("trace.jl")
include("discretize.jl")
export discretize


end
