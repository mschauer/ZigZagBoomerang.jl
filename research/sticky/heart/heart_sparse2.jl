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
using Revise
using FillArrays
using RandomNumbers

function γ(i)
    c = ℂ[i]
    jj = Int[i]
    d = 0
    for s in (CartesianIndex(1,0), CartesianIndex(-1,0),CartesianIndex(0,1),CartesianIndex(0,-1))
        c + s in ℂ || continue
        d += 1
        push!(jj, 𝕃[c + s])
    end
    jj
end
Random.seed!(1)


# Define precision operator of a Gaussian random field (sparse matrix operating on `vec`s of `n*n` matrices)

n = 600
const σ2  = 0.5

const c₁ = 2.0
const c₂ = 0.1

mat(x, n = n) = reshape(x, (n, n)) # vector to matrix

# Γ is very sparse
nnz_(x) = sum(x .!= 0)
nnz_(x::SparseVector) = nnz(x)
nnz_(x::SparseMatrixCSC) = nnz(x)

sparsity(Γ) = nnz_(Γ)/length(Γ)

∇ϕsp(u, i, Γ) = ZigZagBoomerang.idot(Γ, i, u) + (u[i][2] - h(i))/σ2

# Random initial values
t0 = 0.0
h(x, y) = x^2+(5y/4-sqrt(abs(x)))^2
sz = 3.0
const ℂ = CartesianIndices((n,n))
const 𝕃 = LinearIndices(ℂ)

const n_ = n
const sz_ = 3.0
const r1_ = range(-1.5-sz_, 1.5+sz_, length=n_)
const r2_ = range(-1.1-sz_, 1.9+sz_, length=n_)
const rng = RandomNumbers.Xorshifts.Xoroshiro128Plus(0xe8c63206f9a9cc10, 0xf4cd3d6619a36c10)
function h(i) # observation and seeded noise
    c = ℂ[i]
    rng.x, rng.y = RandomNumbers.Xorshifts.init_seed(i, UInt64, Val{2}())
    5max(1 - h(r1_[c[1]], r2_[c[2]]), 0) + sqrt(σ2)*randn(rng)
end
# image(mat(h.(1:500*500)))


x0 = sparse(FillArrays.Zeros(n*n))

κ1 = 0.4
T = 100.

su = false
adapt = false


c1 = 20.0                                   
println("Sparse implementation")  
Z = ZigZag(γ, 0.0, 1.0)
ZigZagBoomerang.neighbours(::typeof(γ), i) = γ(i)
function ZigZagBoomerang.idot(::typeof(γ), j, u::ZigZagBoomerang.SparseState)
    nbrs = γ(j)
    s = (c₂ + c₁*(length(nbrs)-1))*u[j][2]
    for k in 2:length(nbrs)
        s += -c₁*u[nbrs[k]][2]
    end
    s
end

using RandomNumbers
#RandomNumbers.Xorshifts.Xoroshiro128Plus(seed::Integer) = RandomNumbers.Xorshifts.Xoroshiro128Plus(RandomNumbers.Xorshifts.init_seed(seed, UInt64, Val{2}()))
function RandomNumbers.Xorshifts.init_seed(seed, ::Type{UInt64}, _::Val{2})
    x = seed % UInt64
    x1 = RandomNumbers.Xorshifts.splitmix64(x)::UInt64
    x2 = RandomNumbers.Xorshifts.splitmix64(x1)::UInt64
    (x1, x2)
end

using Profile
trace2, acc2 = ZigZagBoomerang.sspdmp3(∇ϕsp, ZigZagBoomerang.sparsestickystate(x0), T, c1, nothing, Z, κ1[1], γ;
                                                strong_upperbounds = su,
                                                adapt = adapt, progress=true)


#
sub = vec(LinearIndices(ℂ)[ℂ[1:10:end,1:10:end]])
@time ŷ2 = mat(mean(subtrace(trace2, sub)), n÷10)

error("stop")                                                
using GLMakie
fig1 = fig = Figure()

ax = [Axis(fig[1,i], title = "posterior mean $i") for i in 1:1]
heatmap!(ax[1], ŷ2, colorrange=(-1,6))

r = LinearIndices(ℂ)[CartesianIndex(n÷2, n÷2)]
st2, sx2 = ZigZagBoomerang.sep(collect(ZigZagBoomerang.subtrace(trace2, r-5:r+5)))

ax = [Axis(fig[2,i], title = "trace $i") for i in 1:1]

lines!(ax[1], st2, getindex.(sx2,1))
