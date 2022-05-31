using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
#using Revise
using ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using FileIO
using Statistics
using GLMakie
using Revise
include("reversiblejump.jl")
include("fastgridlapl.jl")

Random.seed!(1)

n = 30
const σ2  = 0.5
Γ0 = gridlaplacian(Float64, n, n)
c₁ = 2.0
c₂ = 0.1
Γ = c₁*Γ0 +  c₂*I
mat(x) = reshape(x, (n, n)) # vector to matrix
function mat0(y)
    mat(y  .- 0.1)
end
# Γ is very sparse
@show nnz(Γ)/length(Γ) # 0.000496

# Corresponding Gaussian potential
# ϕ(x, Γ, y) = 0.5*x'*Γ*x  + dot(x - y, x - y)/(2*σ2) # not used by the program

# Define ∇ϕ(x, i, Γ) giving the partial derivative of ϕ(x) with respect to x[i]
# ∇ϕ(x, i, Γ, y) = ZigZagBoomerang.idot(Γ, i, x)  + (x[i]-y[i])/σ2 # more efficient that dot(Γ[:, i], x)


# Random initial values
t0 = 0.0
h(x, y) = x^2+(5y/4-sqrt(abs(x)))^2
sz = 3.0
heart = [ max.(1 - h(x, y), 0) for x in range(-1.5-sz,1.5+sz, length=n), y   in range(-1.1-sz,1.9+sz, length=n)]
# image(heart)
μ0 = 5.0*vec(heart) 
y = μ = μ0 + randn(n*n)*sqrt(σ2)
μpost = (Γ + I/σ2)\y/σ2 # ok
Γpost = (Γ + I/σ2)
x = copy(μpost)
κi = 0.15
# w = 1/(1 + (1/sqrt(2*pi/c₂))/κi)
# wi = 1/κi
# w = 1/(1+wi)
# error("")
w = 0.01
N = 10000
Z = [rand() < w for i in eachindex(x)]
Z[1] = true
ββ, ZZ = reversible_jump(sparse(Γpost), μpost, w, N, x, Z, 0.1,  1)
# trace2 = [ββ[i].*ZZ[i] for i in 1:length(ZZ)]
trace2b = [ββ[i][j].*ZZ[i][j] for i in 1:length(ZZ), j in 1:length(ZZ[1])] 
@show mean(ZZ[end])
@show mean(ZZ[1])
scene2a, _ = heatmap(mat(mean(trace2b[end÷2:end,:],dims = 1)[:]))
#mat(mean(trace2b[end:end,:],dims = 1)[:])

# mat(mean(trace2b[end-50:end,:],dims = 1)[:])[1,1]
heatmap(mat(μpost))
heatmap(mat(μ0))
heatmap(mat(y))
display(scene2a)
