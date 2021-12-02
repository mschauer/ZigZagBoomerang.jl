### BENCHMARKS
using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
using LinearAlgebra, ZigZagBoomerang
using Distributions
include("./gibbs.jl")
d = 100
n = 300
# DATA CREATION: non sparse case
Γnull = Matrix(Tridiagonal(-ones(d-1), 2*ones(d), -ones(d-1)))
Γnull[1,1] = Γnull[d,d] =  1.0
Γnull2 = Γnull^2 + I*0.01
inv(Γnull2) # Covariance Matrix
Chl = cholesky(Symmetric(Γnull2))
Atr = Chl.U \ randn(d, n) 
A = Atr'
# Choleski
xtrue = [rand() < 0.2 ? 1.0 : 0.0 for _ in 1:d]
# xtrue = randn(d)*sqrt(σ0) 
const σ2 = 1.0
const σ0 = 10.0
y = A*xtrue + randn(n)*sqrt(σ2) 


using SparseArrays
Γpost = sparse(Atr*A/σ2 + I/σ0)
# Γpost = droptol!(Γpost, 0.1)
x0 = μpost = (Γpost \ A'y)/σ2
[xtrue x0] ## Look correct
# ∇ϕ(x, i, A, y) = -dot(A[:,i], (y - A*x))/σ2 + x[i]/σ0 # non sparse
# ∇ϕ(x, i, Γpost, μpost) = dot(Γpost[:,i],(x - μpost)) # non sparse
# κ1 = 0.4*ones(length(x0))
# c = fill(0.001, d)
# T = 1.0
# θ0 = rand([-0.1,0.1], d)
# t0 = 0.0
# Z = ZigZag(Γpost, μpost) 
# ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ1, A, y)
# trace, acc = ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ1, Γpost, μpost)
# ts2, xs2 = ZigZagBoomerang.sep(collect(trace))





function benchmark_1(A, Atr, y, T, N)
    # N number of iteration of the Gibbs sampler
    # T final clock of the sticky pdmp
    d = size(Atr*A, 1)
    x0 = fill(10.0, d)
    n = length(y)
    Γpost = Atr*A/σ2 + I/σ0
    sΓpost = sparse(Γpost) 
    μpost = (Γpost \ A'y)/σ2
    Z = ZigZag(sΓpost, μpost) 
    ∇ϕ(x, i, Γpost, μpost) = dot(Γpost[:,i],(x - μpost))

    # prior w = 0.2
    wi = 0.2
    ki = 1/(2*π*sqrt(σ0))/(1/wi - 1)
    println("k equal to $(ki)")
    κ = ki*ones(length(x0))
    c = fill(0.001, d)
    θ0 = rand([-0.1,0.1], d)
    t0 = 0.0
    # initialize all the objects needed
    # start from far in the tail
    w = fill(wi, d) # to fix
    Z1 = w .> 0
    ## prior slab gibbs c0*σ2 = σ0
    c0 = σ0/σ2 
    println("run once to trigger precompilation")
    ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, Γpost, μpost)
    gibbs_linear(y, A, w, N, x0, Z1, σ2, c0)
    println("sticky Zig-Zag")
    # timer inside sspdmp2
    trace, acc = ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, Γpost, μpost)
    println("Gibbs sampler")
    ββ, ZZ = @time gibbs_linear(y, A, w, N, x0, Z1, σ2, c0)
    return trace, ββ
end
T = 1000.0
N = 50
trace, tracegibs  = benchmark_1(A, Atr, y, T, N)
ts2, xs2 = ZigZagBoomerang.sep(collect(trace))

produce_plots = true
if produce_plots
end


function benchmark_2(dd, nn, rep)
    col = dd
    row = nn
    Q1 = zeros(row, col, rep)
    Q2 = zeros(row, col, rep)
    # run onces to trigger compilation
    for d in dd
        for n in nn
            for _ in rep
            # time sticky pdmp version with an exit value when the right model is explored
            # save it in Q1
            # time the gibbs sampler in the same way
            # save it in Q2
            end
        end
    end
    Q1,Q2
end


# sparse case change A 







