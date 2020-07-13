using ZigZagBoomerang
using LinearAlgebra
using Random
using SparseArrays
using Test
using Profile
#using ProfileView
include("gridlaplace.jl")

# Define precision operator of a Gaussian random field
n = 100
Γ = 0.01I + gridlaplacian(Float64, n, n)

# Γ is very sparse
@show nnz(Γ)/length(Γ) # 0.000496

# ∇ϕ gives the negative partial derivative of log density
∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # partial derivative of ϕ(x) with respect to x[i]

# Random initial values
t0 = 0.0
x0 = randn(n*n)
θ0 = rand([-1.0,1.0], n*n)

# Rejection bounds
c = [norm(Γ[:, i], 2) for i in 1:n*n]

# Define ZigZag
Z = ZigZag(Γ, x0*0)
# or try the FactBoomerang
#Z = FactBoomerang(Γ, x0*0, 0.1)

# Run sparse ZigZag for T time units and collect trajectory
T = 30.0
trace, _, (acc, num) = spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ);
@time spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ);
0
