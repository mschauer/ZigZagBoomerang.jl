using Revise
using Makie, ZigZagBoomerang, SparseArrays, LinearAlgebra

function ϕ(x, i, μ)
    x[i] - μ[i]
end
κ = 10.0
n = 2
μ = [1.8, 0.2]
x0 = randn(n)
θ0 = rand((-1.0,1.0), n)
T = 2000.0
c = 10ones(n)

#@time trace0, _ = ZigZagBoomerang.spdmp(ϕ, 0.0, x0, θ0, T, c, ZigZag(sparse(1.0I,n,n), zeros(n)), μ)

@time trace0, _ = ZigZagBoomerang.sspdmp(ϕ, 0.0, x0, θ0, T, c, κ, ZigZag(sparse(1.0I,n,n), zeros(n)), μ)
ts0, xs0 = splitpairs(trace0)

lines(ts0, getindex.(xs0,1))
ts0, xs0 = splitpairs(discretize(trace0, 0.01))
p1 = scatter(getindex.(xs0,1), getindex.(xs0,2), color=(:black, 0.4), markersize=0.01)
save(joinpath("figures", "spikeandslab.png"), title(p1, "Spike and Slab"))
