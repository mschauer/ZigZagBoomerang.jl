module ZigZagBoomerang


# Zig zag and Boomerang reference implementation

include("poissontime.jl")
export poisson_time



include("pdmps.jl")
export Boomerang, Bps, pdmp, eventtime, eventposition


include("fact_pdmps")
export ZigZag, FactBoomerang

include("discretize.jl")
include("trace.jl")
include("localzigzag.jl")
end
