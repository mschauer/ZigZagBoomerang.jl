### FULL MATRIX LOGISTIC REGRESSION ###
using LinearAlgebra
using Random
using ReverseDiff
using Optim
using Plots
using SparseArrays
using ZigZagBoomerang
const ZZB = ZigZagBoomerang
#generate data


# partial derivative
include("./fulldesign.jl")
Random.seed!(1)
(X, y, ξtrue) = generate_data(10, 2000)
N, d = size(X)

### Finding the mode ###
# Energy function logistic
function ϕ(x, A, y, precision)
    sum(log.(1 .+ exp.(A*x)) .- y.*A*x) + precision*dot(x,x)*0.5
end

# jth term of the kth partial derivative
function ∇ϕs(x, k, j, A, y)
    A[j,k]*exp(dot(A[j,:], x))/(1 + exp(dot(A[j,:], x))) - y[j]*A[j,k]
end


include("preprocessing_logistic.jl")
∇ϕξref = [∇ϕ(ξref, k, X, y, precision) for k in 1:length(ξref)]
norm(ξref - ξtrue)

### STANDARD STICKY ZIG-ZAG WITH CONTROL VARIATES
function ∂ϕ!(ξ, k, ∇ϕξref, ξref, precision, y, X, N)
    j = rand(1:length(y)) # subsampling over sample size
    E_tilde = ∇ϕs(ξ, k, j, X, y)
    E_tilde′ = ∇ϕs(ξref,k, j, X, y)
    return ∇ϕξref[k] + N*(E_tilde - E_tilde′) + precision*(ξ[k] - ξref[k])
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
    max(0, θ[i]*(∇ϕξ′[i] + precision*x[i])) + abs(θ[i])*c[i].c*norm(x - ξ′), abs(θ[i])*c[i].c*norm(θ) + θ[i]^2*precision
    # NB: if ξ and θ are sparse vectors, this implemention is not efficient
    # To Check θ[i]*(∇ϕξ′[i] + precision*x[i]))
end


F = ZigZag(ones(d,d), zeros(d))
adapt=false
su = true
κ = 0.5
precision = 1.0
t0, x0, θ0, T = 0.0, randn(d), ones(d), 100.0
Ξ, (t, x, θ), (acc, num), c = sspdmp(∂ϕ!, t0, x0, θ0, T, c, F, κ,
                                    ∇ϕξref, ξref, precision, y, X, N;
                                    strong_upperbounds = su ,
                                     adapt = adapt)


error("")


#############################################################################
####### SUBSAMPLING OVER DIMENSIONS WITH CONTROL VARIATES ###################
#############################################################################
function gradϕ!(ξ, ∇ϕξref, ξref, precision, y, X, N, d)
    k = rand(1:d) # subsampling over dimension
    j = rand(1:N) # subsampling over sample size
    E_tilde = ∇ϕs(ξ, k, j, X, y)
    E_tilde′ = ∇ϕs(ξref, k, j, X, y)
    return (∇ϕξref[k] + d*N*(E_tilde - E_tilde′) + precision*(ξ[k] - ξref[k]), k)
end
struct MyBoundSub
    c::Float64
end
c = MyBoundSub(0.0)
C_k = 0.0
for k in 1:d
    for j in 1:N
        global C_k = max(C_k, N*d*0.25*abs.(X[j,k]).*norm(X[j,:]))
    end
    global c = MyBoundSub(C_k)
end

function ZZB.ab(x, θ, c::MyBoundSub, F::ZigZag, ∇ϕξ′, ξ′, precision, y, X, N, d, args...)
    #max(0, θ[i]*∇ϕξ′[i]) + abs(θ[i])*c[i]*norm(ξ - ξ′), abs(θ[i])*c[i]*norm(θ)
    # with prior
    ref = max([θ[i]*(∇ϕξ′[i] + precision*x[i]) for i in 1:d]..., 0)
    θstar = max([abs(θ[i]) for i in 1:d]...)
    ref + θstar*c.c*norm(x - ξ′), θstar*c.c*norm(θ) + θstar^2*precision # NB: if ξ and θ are sparse vectors, this implemention is not efficient
end

Ξ, (t, x, θ), (acc, num), c = pdmp_sub(gradϕ!, t0, x0, θ0, T, c, F, ∇ϕξref, ξref, precision, y, X, N, d; adapt=false, factor=2.0)
