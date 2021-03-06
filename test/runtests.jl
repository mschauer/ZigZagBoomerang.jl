using Test
using Statistics
using Random
using LinearAlgebra
using ZigZagBoomerang
using ZigZagBoomerang: poisson_time
Random.seed!(1)


include("test1d.jl")
include("maintest.jl")

include("staticarrays.jl")

include("sticky.jl")