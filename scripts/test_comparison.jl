using SparseArrays
using LinearAlgebra
using ZigZagBoomerang
using Random
using Statistics
ϕ1(x) = cos(π*x) + x^2/2 # not needed
# gradient of ϕ(x)
∇ϕ1(x) = -π*sin(π*x) + x # (REPLACE IT WITH AUTOMATIC DIFFERENTIATION)
# Example: Boomerang
Random.seed!(1)
x0 = randn()
θ0 = randn()
T = 1000.0
B = Boomerang1d(0.0, 1.0)
out2, acc = ZigZagBoomerang.pdmp(∇ϕ1, x0, θ0, T, 3.5π, B)
mean(getindex.(out2,2))


ϕ(x) = [cos(π*x[1]) + x[1]^2/2] # not needed
# gradient of ϕ(x)
∇ϕ(x) = [-π*sin(π*x[1])]
∇ϕ(x, i) = ∇ϕ(x)[i] # (REPLACE IT WITH AUTOMATIC DIFFERENTIATION)
c = [3.5π]
λref = 1.0
n = 1
x0 = randn(n)
θ0 = randn(n)
t0 = 0.0
T = 10000.0
Γ = sparse(Matrix(1.0I, n, n))
B = FactBoomerang(Γ, x0*0, λref)
trace, _,  acc = pdmp(∇ϕ, t0, x0, θ0, T, c, B)
xs = mean(last.(collect(trace)))
