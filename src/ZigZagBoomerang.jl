module ZigZagBoomerang


# Zig zag and Boomerang reference implementation
pos(x) = max(zero(x), x)
include("poissontime.jl")
export poisson_time



include("pdmps.jl")
export Boomerang, Bps, pdmp, eventtime, eventposition


include("fact_pdmps.jl")
export ZigZag, FactBoomerang, LocalZigZag, discretize 

include("discretize.jl")
include("trace.jl")
include("localzigzag.jl")
end
