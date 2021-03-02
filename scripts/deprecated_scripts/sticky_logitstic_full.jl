### FULL MATRIX LOGISTIC REGRESSION ###
using LinearAlgebra
using Random
# using ReverseDiff
using Optim
using Plots
using SparseArrays
using ZigZagBoomerang
const ZZB = ZigZagBoomerang
#generate data


# partial derivative
include("./fulldesign.jl")
Random.seed!(1)
p = 100
N = 1000
w = 0.95
ξtrue = 2*(rand(p).> w)
(X, y) = generate_data(ξtrue, p, N)

N, d = size(X)


### Finding the mode ###
# Energy function logistic
# smooth component of the prior measure
const prec = 0.01
function pri(x)
    sqrt(prec)*exp(-prec*dot(x,x)*0.5)/sqrt(2π)
end

# gradient of the smooth component of the prior
function ∇pri(x)
    -prec.*x
end

function ∇pri(x, i)
    -prec*x[i]
end

# Ψ such that μ(dx) = \exp(-ψ)∏(dx_i + k^(-1) δ_0(dx_i))
function ϕ(x, A, y)
    sum(log.(1 .+ exp.(A*x)) .- y.*A*x) #- pri(x)
end

# jth term of the kth partial derivative
function ∇ϕs(x, k, j, A, y)
    A[j,k]*exp(dot(A[j,:], x))/(1 + exp(dot(A[j,:], x))) - y[j]*A[j,k] #- ∇pri(x, j)
end

include("preprocessing_logistic.jl")
∇ϕξref = [∇ϕ(ξref, k, X, y) for k in 1:length(ξref)]

### STANDARD STICKY ZIG-ZAG WITH CONTROL VARIATES
function ∂ϕ!(ξ, k, ∇ϕξref, ξref, ∇pri, y, X, N)
    j = rand(1:length(y)) # subsampling over sample size
    E_tilde = ∇ϕs(ξ, k, j, X, y)
    E_tilde′ = ∇ϕs(ξref, k, j, X, y) # could be pre-computed
    return ∇ϕξref[k] + N*(E_tilde - E_tilde′) - ∇pri(ξ, k)
end

struct MyBoundLog
    c::Float64
end

c = [MyBoundLog(0.0) for i in 1:d]
for k in 1:d
    C_k = 0.0
    for j in 1:N
        C_k = max(C_k, N*0.25*abs.(X[j,k]).*norm(X[j,:]))
    end
    c[k] = MyBoundLog(C_k)
end

# a and b
function ZZB.ab(G, i, x, θ, c::Vector{MyBoundLog}, F::ZigZag, ∇ϕξ′, ξ′, precision, args...)
    #max(0, θ[i]*∇ϕξ′[i]) + abs(θ[i])*c[i]*norm(ξ - ξ′), abs(θ[i])*c[i]*norm(θ)
    # with prior
    # max(0, θ[i]*(∇ϕξ′[i] + precision*x[i])) + abs(θ[i])*c[i].c*norm(x - ξ′), abs(θ[i])*c[i].c*norm(θ) + θ[i]^2*precision
    max(0, θ[i]*(∇ϕξ′[i] - ∇pri(x, i))) + abs(θ[i])*c[i].c*norm(x - ξ′), abs(θ[i])*c[i].c*norm(θ) - ∇pri(θ, i)^2
    # NB: if ξ and θ are sparse vectors, this implemention is not efficient
    # To Check θ[i]*(∇ϕξ′[i] + precision*x[i]))
end


F = ZigZag(ones(d,d), zeros(d))
adapt = false
su = true #strong upperbounds which continue to be upperbounds after
κ = getkappa(w, pri)
t0, x0, θ0, T = 0.0, randn(d), ones(d), 100.0
Ξ, (t, x, θ), (acc, num), c = ZZB.sspdmp(∂ϕ!, t0, x0, θ0, T, c, F, κ,
                                    ∇ϕξref, ξref, ∇pri, y, X, N;
                                    strong_upperbounds = su ,
                                     adapt = adapt)

error("")

first2((t,x)) = t => x[1:2]
sep(x) = first.(x), last.(x)
ts1, xs1 = sep(discretize(Ξ, 0.05))
p1 = plot(ts1, getindex.(xs1,1), leg = false)
savefig(p1, "figure1" )
p2 = plot(ts1, getindex.(xs1,7), leg = false)
savefig(p2, "figure2")
save
#
#
# #############################################################################
# ####### SUBSAMPLING OVER DIMENSIONS WITH CONTROL VARIATES ###################
# #############################################################################
# function gradϕ!(ξ, ∇ϕξref, ξref, precision, y, X, N, d)
#     k = rand(1:d) # subsampling over dimension
#     j = rand(1:N) # subsampling over sample size
#     E_tilde = ∇ϕs(ξ, k, j, X, y)
#     E_tilde′ = ∇ϕs(ξref, k, j, X, y)
#     return (∇ϕξref[k] + d*N*(E_tilde - E_tilde′) + precision*(ξ[k] - ξref[k]), k)
# end
# struct MyBoundSub
#     c::Float64
# end
# c = MyBoundSub(0.0)
# C_k = 0.0
# for k in 1:d
#     for j in 1:N
#         global C_k = max(C_k, N*d*0.25*abs.(X[j,k]).*norm(X[j,:]))
#     end
#     global c = MyBoundSub(C_k)
# end
#
# function ZZB.ab(x, θ, c::MyBoundSub, F::ZigZag, ∇ϕξ′, ξ′, precision, y, X, N, d, args...)
#     #max(0, θ[i]*∇ϕξ′[i]) + abs(θ[i])*c[i]*norm(ξ - ξ′), abs(θ[i])*c[i]*norm(θ)
#     # with prior
#     ref = max([θ[i]*(∇ϕξ′[i] + precision*x[i]) for i in 1:d]..., 0)
#     θstar = max([abs(θ[i]) for i in 1:d]...)
#     ref + θstar*c.c*norm(x - ξ′), θstar*c.c*norm(θ) + θstar^2*precision # NB: if ξ and θ are sparse vectors, this implemention is not efficient
# end
#
# Ξ, (t, x, θ), (acc, num), c = pdmp_sub(gradϕ!, t0, x0, θ0, T, c, F, ∇ϕξref, ξref, precision, y, X, N, d; adapt=false, factor=2.0)
