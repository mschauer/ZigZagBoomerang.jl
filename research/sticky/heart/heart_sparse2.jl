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

#n = 1000
n = 100
const σ2  = 0.5

const c₁ = 2.0
const c₂ = 0.1

mat(x, n = n) = reshape(x, (n, n)) # vector to matrix

# Γ is very sparse
nnz_(x) = sum(x .!= 0)
nnz_(x::SparseVector) = nnz(x)
nnz_(x::SparseMatrixCSC) = nnz(x)
nnz_(x::ZigZagBoomerang.SparseState) = nnz(x)


sparsity(Γ) = nnz_(Γ)/length(Γ)


∇ϕsp(u, i, Γ) = ZigZagBoomerang.idot(Γ, i, u) + (u[i][2] - h(i))/σ2

# Random initial values
t0 = 0.0
h(x, y) = x^2+(5y/4-sqrt(abs(x)))^2
sz = 5.0
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
function h0(i) # observation and seeded noise
    c = ℂ[i]
    5max(1 - h(r1_[c[1]], r2_[c[2]]), 0) 
end

println("Sparse implementation")  

# image(mat(h.(1:500*500)))
@show sum(0 != h0(i) for i in 1:n*n)/(n*n)

x0 = sparse(FillArrays.Zeros(n*n))


cluster = true
if cluster
    clusterα = 0.5
    κ1 = 0.03
    T = 1000. 
else
    clusterα = 1.0
    κ1 = 0.05
    T = 1000.
end

adapt = false

c1 = 5.0


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
# x - (0.1 + 8 + 8 + 2)/x
# x - (c₂ + c₁*(4 + 4) + 1/σ2)/x 


using RandomNumbers
#RandomNumbers.Xorshifts.Xoroshiro128Plus(seed::Integer) = RandomNumbers.Xorshifts.Xoroshiro128Plus(RandomNumbers.Xorshifts.init_seed(seed, UInt64, Val{2}()))
function RandomNumbers.Xorshifts.init_seed(seed, ::Type{UInt64}, _::Val{2})
    x = seed % UInt64
    x1 = RandomNumbers.Xorshifts.splitmix64(x)::UInt64
    x2 = RandomNumbers.Xorshifts.splitmix64(x1)::UInt64
    (x1, x2)
end

using Profile
trace2, acc2, uT = ZigZagBoomerang.sspdmp3(∇ϕsp, ZigZagBoomerang.sparsestickystate(x0), T, c1, nothing, Z, κ1[1], γ;
                                                adapt = adapt, progress=true, progress_stops = 50, clusterα=clusterα)

@show length(trace2.events)
                                            

@show sparsity(uT)

ss = 1
sub = vec(LinearIndices(ℂ)[ℂ[1:ss:end,1:ss:end]])
@time ŷ2 = mat(mean(subtrace(trace2, sub)), n÷ss)
xtrue = mat(h0.(sub), n÷ss)
#error("stop")                                                
using GLMakie
fig1 = fig = Figure()

ax = [Axis(fig[1,i], title = ["posterior mean","sqr err"][i]) for i in 1:2]
heatmap!(ax[1], ŷ2, colorrange=(-1,6))
heatmap!(ax[2], (xtrue - ŷ2).^2, colorrange=(-1,6))

r = LinearIndices(ℂ)[CartesianIndex(n÷2, n÷2)]
st2, sx2 = ZigZagBoomerang.sep(collect(ZigZagBoomerang.subtrace(trace2, r-5:r+5)))

ax = [Axis(fig[2,i:i+1], title = "trace $i") for i in 1:1]

lines!(ax[1], st2, getindex.(sx2,1))
lines!(ax[1], st2, getindex.(sx2,6))
lines!(ax[1], st2, getindex.(sx2,11))

pic = mat(map(i->((uT[i][2]≠0)⊻(h0(i)≠0)), sub), n÷ss)
@show sum(pic)
fp = sum(map(i->((uT[i][2]≠0)&(h0(i)==0)), sub))
fn = sum(map(i->((uT[i][2]==0)&(h0(i)≠0)), sub))
tp = sum(map(i->((uT[i][2]≠0)&(h0(i)≠0)), sub))
tn = sum(map(i->((uT[i][2]==0)&(h0(i)==0)), sub))

println("$fp fp + $fn fn = ", sum(pic), " f, $tp tp + $tn tn")

sub2 = vec(LinearIndices(ℂ)[ℂ[1:100,1:100]])
pic2 = mat(map(i->((uT[i][2]≠0)), sub2), 100)

fig2 = heatmap(pic)
save("largeheart$(n)-$(clusterα)-b.png", fig2)

save("largeheart$(n)-$(clusterα).png", fig1)
fig1
#=
sum((0 != h0(i) for i = 1:n * n)) / (n * n) = 0.03097
Progress: 100%|████████████████████████████████| Time: 0:14:49
890.480499 seconds (3.50 G allocations: 178.750 GiB, 3.76% gc time, 0.14% compilation time)
acc 0.29409087558578345
length(trace2.events) = 124934117
sparsity(uT) = 0.057117
 34.194060 seconds (792.10 k allocations: 336.415 MiB, 9.47% gc time)
sum(pic) = 1394
1173 fp + 221 fn = 1394 f, 1015 tp + 37591 tn

sum((0 != h0(i) for i = 1:n * n)) / (n * n) = 0.03097
Progress: 100%|████████████████████████████████| Time: 0:25:34
1535.794409 seconds (3.52 G allocations: 169.055 GiB, 10.12% gc time, 0.12% compilation time)
acc 0.293928480952627
length(trace2.events) = 169466907
sparsity(uT) = 0.061446
 56.935668 seconds (1.47 k allocations: 294.867 MiB, 6.53% gc time, 2.97% compilation time)
sum(pic) = 1847
1419 fp + 428 fn = 1847 f, 808 tp + 37345 tn

=#
