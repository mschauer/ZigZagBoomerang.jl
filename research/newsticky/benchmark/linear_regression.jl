### BENCHMARKS
using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
using LinearAlgebra, ZigZagBoomerang
using Distributions, GLMakie, Random, SparseArrays
Random.seed!(1)
include("./gibbs.jl")
d = 300
n = 400
# DATA CREATION: non sparse case
function gridlaplacian(T, m, n)
    S = sparse(T(0.0)I, n*m, n*m)
    linear = LinearIndices((1:m, 1:n))
    for i in 1:m
        for j in 1:n
            for (i2, j2) in ((i + 1, j), (i, j + 1))
                if i2 <= m && j2 <= n
                    S[linear[i, j], linear[i2, j2]] -= 1.
                    S[linear[i2, j2], linear[i, j]] -= 1.

                    S[linear[i, j], linear[i, j]] += 1.
                    S[linear[i2, j2], linear[i2, j2]] += 1.
                end
            end
        end
    end
    S
end

# Define precision operator of a Gaussian random field (sparse matrix operating on `vec`s of `n*n` matrices)
dd = 10
Γnull = gridlaplacian(Float64, dd, dd)
d = size(Γnull, 1)
# Γnull = Matrix(Tridiagonal(-ones(d-1), 2*ones(d), -ones(d-1)))
# Γnull[1,1] = Γnull[d,d] =  1.0
Γnull2 = Γnull^4 + I*0.01
# inv(Matrix(Γnull2)) # Covariance Matrix
Chl = cholesky(Symmetric(Γnull2))
Atr = Chl.U \ randn(d, n) 
A = Atr'
# Choleski
xtrue = [ i < 0.2*d ? 1.0 : 0.0 for i in 1:d]
# xtrue = randn(d)*sqrt(σ0) 
const σ2 = 1.0
const σ0 = 20.0
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
    ki = 1/(sqrt(2*π*σ0))/(1/wi - 1)
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
    x0 = fill(10.0, d)
    trace, acc = ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, Γpost, μpost)
    subiter = 1
    println("Gibbs sampler")
    x0 = fill(10.0, d)
    Z1 = w .> 0
    ββ, ZZ = @time gibbs_linear(y, A, w, N, x0, Z1, σ2, c0, subiter)
    return trace, ββ, ZZ
end
T = 5000.0
N = 100
trace, ββ, ZZ   = benchmark_1(A, Atr, y, T, N)
d = length(ZZ[1])
ts2, xs2 = ZigZagBoomerang.sep(collect(trace))
ts2b, xs2b = ZigZagBoomerang.sep(collect(discretize(trace, T/N)))
traceb = [xs2b[i][j] for i in 1:length(xs2b), j in 1:length(ZZ[1])]
trace2 = [ββ[i].*ZZ[i] for i in 1:length(ZZ)]
trace2b = [ββ[i][j].*ZZ[i][j] for i in 1:length(ZZ), j in 1:length(ZZ[1])] 

produce_heatmap = true
if produce_heatmap
    fig1 = Figure()
    ax = [Axis(fig1[1, j]) for j in 1:2]

    heatmap!(ax[1], trace2b[100:end,:])
    heatmap!(ax[2], traceb[100:end,:])
end
fig1
produce_plots = true
if produce_plots
    using GLMakie, Colors, ColorSchemes
    f = Figure(backgroundcolor = RGB(0.98, 0.98, 0.98),
            resolution = (1000, 500))
    ax = [Axis(f[1, j]) for j in 1:2]
    i1, i2 = findfirst(x -> x != 0.0, xtrue), findfirst(x -> x == 0.0, xtrue)
    lines!(ax[1], ts2, getindex.(xs2, i1), color = :black)
    lines!(ax[1], ts2, getindex.(xs2, i2), color = :red) 
    scatter!(ax[2], eachindex(trace2), getindex.(trace2, i1), color = :black)
    scatter!(ax[2], eachindex(trace2), getindex.(trace2, i2), color = :red)
    hlines!(ax[1], [xtrue[i1], xtrue[i2]])
    hlines!(ax[2], [xtrue[i1], xtrue[i2]])
end
f
getindex.(trace2, i1)
error("")



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


