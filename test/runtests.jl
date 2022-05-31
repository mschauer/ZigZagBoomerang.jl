using Test
using Statistics
using Random
using LinearAlgebra
using ZigZagBoomerang
using ZigZagBoomerang: poisson_time, sep
const CI = true

Random.seed!(1)

include("poisson.jl")
include("test1d.jl")
include("maintest.jl")
include("testiter.jl")
include("testparallel.jl")
include("staticarrays.jl")

include("sticky.jl")
include("sticky_test.jl")

#include("forwarddiff.jl")
