module ZigZagBoomerang
using Random
using Requires
using Statistics
using ProgressMeter
using RandomNumbers.Xorshifts
using RandomNumbers: gen_seed
const Rng = Xoroshiro128Plus 
Seed() = gen_seed(UInt64, 2)

#using AbstractMCMC

# ZigZag1d and Boomerang1d reference implementation
include("types.jl")
include("common.jl")
include("dynamics.jl")
export ZigZag1d, Boomerang1d, ZigZag, FactBoomerang
const LocalZigZag = ZigZag
export LocalZigZag, BouncyParticle, Boomerang
#include("laplace.jl")

include("poissontime.jl")
export poisson_time
include("jointflow.jl")
include("fact_samplers.jl")
include("not_fact_samplers.jl")

include("priorityqueue.jl")
include("morepriorityqueues.jl")
#const SPriorityQueue = PriorityQueue
include("sfact.jl")
include("local.jl")

include("parallel.jl")
include("sfactiter.jl")
include("notfactiter.jl")


include("zigzagboom1d.jl")
export pdmp, spdmp, eventtime, eventposition

include("ss_fact.jl")
export sspdmp

include("stickyzz.jl")
export stickyzz
include("sparsestickyzz.jl")
export sparsestickyzz

include("ss_not_fact.jl")
export sticky_pdmp

include("trace.jl")
include("condition.jl")
include("discretise.jl")
const discretise = discretize 
export discretise, discretize, sdiscretize, subtrace
include("staticarrays.jl")

#function __init__()
#    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" include("staticarrays.jl")
#end

end
