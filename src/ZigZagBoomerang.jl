module ZigZagBoomerang
using Random

# ZigZag1d and Boomerang1d reference implementation
include("types.jl")
include("common.jl")
include("dynamics.jl")
export ZigZag1d, Boomerang1d, ZigZag, FactBoomerang
const LocalZigZag = ZigZag
export LocalZigZag, BouncyParticle, Boomerang

include("poissontime.jl")
export poisson_time

include("fact_samplers.jl")
include("not_fact_samplers.jl")

include("priorityqueue.jl")
#const SPriorityQueue = PriorityQueue
include("sfact.jl")



include("zigzagboom1d.jl")
export pdmp, spdmp, eventtime, eventposition


include("trace.jl")
include("discretize.jl")
export discretize


end
