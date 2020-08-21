#################################################################################
# Comparison of Zig-Zag for diffusion bridges with tailored Poisson rates       #
# and with adapted Poisson rate. Reference: https://arxiv.org/abs/2001.05889.   #
#################################################################################

using ZigZagBoomerang, SparseArrays, LinearAlgebra
#using CairoMakie
include("../faberschauder.jl")
const ZZB = ZigZagBoomerang

# Drift
const α = 1.0
const L = 6
const T = 50.0
b(x) = α * sin(x)
# First derivative
b′(x) = α * cos(x)
# Second derivative
b″(x) = -α * sin(x)


# Zig-Zag impmentation
n = (2 << L) + 1
ξ0 = 0randn(n)
u, v = -π, 3π  # initial and fianl point
ξ0[1] = u / sqrt(T)
ξ0[end] = v / sqrt(T)
θ0 = rand((-1.0, 1.0), n)
θ0[end] = θ0[1] = 0.0 # fix final point
T′ = 2000.0 # final clock of the pdmp
Γ = sparse(1.0I, n, n)

####################################################################
# Overloading Poisson times in order to have tighter upperbounds   #
####################################################################
struct MyBound
    c::Float64
end
function ZZB.adapt!(b::Vector{MyBound}, i, x)
    b[i] = MyBound(b[i].c * x)
end

"""
    poisson_time(a, b, c, u)
Obtaining waiting time for inhomogeneous Poisson Process
with rate of the form λ(t) = a + (b + c*t)^+, where `c`,`a`> 0 ,`b` ∈ R, `u` uniform random variable
"""
function ZZB.poisson_time((a, b, c)::NTuple{3}, u = rand()) # formula (22)
    if b > 0
        return (-(b + a) + sqrt((b + a)^2 - 2.0 * c * log(u))) / c # positive solution of quadratic equation c*0.5 x^2 + (b + a) x + log(u) = 0
    elseif a * b / c <= log(u)
        return -log(u) / a
    else
        return (-(a + b) + sqrt((a + b)^2 - 2.0 * c * (b * b * 0.5 / c + log(u)))) / c    # positive solution of quadratic equation c*0.5 x^2 + (b + a) x + log(u) + b*b*0.5/c = 0
    end
end

ZZB.sλ̄((a, b, c)::NTuple{3}, Δt) = a + ZZB.pos(b + c * Δt)
"""
    abc(G, i, x, θ, c, Flow)
Returns the constant term `a` and linear term `b` when computing the Poisson times
from the upper upper bounding rates λᵢ(t) = max(a + b*t)^2. The factors `a` and `b`
can be function of the current position `x`, velocity `θ`, tuning parameter `c` and
the Graph `G`
"""
function ZZB.ab(G, i, x, θ, c::Vector{MyBound}, F::ZigZag)
    if i == 1
        a = c[i].c + T^(1.5)*0.5*(α^2 + α) * abs(θ[i])  # initial point
        b1 = θ[i]*(x[i] - x[end])
        b2 = θ[i]*(θ[i] - θ[end])
    elseif i == (2 << L) + 1
        a = c[i].c + T^(1.5)*0.5*(α^2 + α) * abs(θ[i])  # final point
        b1 = θ[i]*(x[i] - x[1])
        b2 = θ[i]*(θ[i] - θ[1])
    else
        l = lvl(i, L)
        a = c[i].c + T^(1.5) / 2^((L - l) * 1.5 + 2) * (α^2 + α) * abs(θ[i]) # formula (22)
        b1 = x[i] * θ[i]
        b2 = θ[i] * θ[i]
    end
    return a, b1, b2
end

c = [MyBound(0.0) for i in 1:n]
trace, (t, ξ, θ), (acc, num), c = @time spdmp(∇ϕmoving, 0.0, ξ0, θ0, T′, c, ZigZag(Γ, ξ0 * 0),
                        SelfMoving(), L, T, adapt = false);

trace
@show acc/num
ts, ξs = splitpairs(discretize(trace, T′/n))
S = T*(0:n)/(n+1)

using Makie
using CairoMakie
p1 = lines(S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
for ξ in ξs[1:5:end]
    lines!(p1, S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
end
display(p1)


#####################################################
# Try Bouncy particle sampler with exact bounds     #
#####################################################

c_bps = MyBound(0.0)
n = (2 << L) + 1
ξ0, θ0 = randn(n), randn(n)
θ0[end] = θ0[1] = 0.0 # fix final point
u, v = -π, 3π  # initial and fianl point
ξ0[1] = u / sqrt(T)
ξ0[end] = v / sqrt(T)


# overloading Poisson times
# WARNING: this function works only for diffusionn bridges and not processes. to generalize it
# change [T^(1.5)/2^((L - lvl(i, L)) * 1.5 + 2) * (α^2 + α) for i in eachindex(x)]
function ZZB.ab(x, θ, c::MyBound, F::BouncyParticle)
    a = c.c + dot([T^(1.5)/2^((L - lvl(i, L)) * 1.5 + 2) * (α^2 + α) for i in eachindex(x)], abs.(θ))
    b1 = dot(x, θ)
    b2 = dot(θ, θ)
    return a, b1, b2
end

function ZZB.adapt!(b::MyBound, x)
    b = MyBound(b.c * x)
end

#Do not reflect the first and the last coefficient
function ZZB.reflect!(∇ϕx, x, θ, F::BouncyParticle)
    θ[2:end-1] .-= (2*dot(∇ϕx[2:end-1], θ[2:end-1])/ZZB.normsq(∇ϕx[2:end-1]))*∇ϕx[2:end-1]
    θ
end
#Do not refresh the first and the last coefficient
function ZZB.refresh!(θ, F::BouncyParticle)
    for i in eachindex(θ)
        θ[i] = randn()
    end
    θ[1] = θ[end] = 0.0
    θ
end
ZZB.sλ(∇ϕx, θ, F) = ZZB.λ(∇ϕx, θ, F)
ZZB.sλ̄((a, b, c)::NTuple{3}, Δt) = a + ZZB.pos(b + c * Δt)


function ∇ϕ!(y, ξ, (L, T); K = 1)
    s = T * (rand())
    x = dotψ(ξ, s, L, T)
    bb = 2b(x) * b′(x) + b″(x)
    for i in eachindex(ξ)
        if i == (2 << L) + 1    # final point
            y[i] = 0.0
        elseif i == 1   # initial point
            y[i] = 0.0
        else
            l = lvl(i, L)
            k = (i - 1) ÷ (2 << l)
            δ = T / (1 << (L - l))
            if δ*k <= s <=   δ*(k + 1)
                y[i] = 0.5 * δ * Λ(s, L - l, T) * bb + ξ[i]
            else
                y[i] =  ξ[i]
            end
        end
    end
    y ξ[i]
end


const α = 1.0
T′ = 30000.0 # final clock of the pdmp
λref_bps = 1.0
Γ = sparse(1.0I, n, n)
B = BouncyParticle(Γ, x0*0, λref_bps)
out1, uT, acc = pdmp(∇ϕ!, 0.0, ξ0, θ0, T′, c_bps, B, (L, T); adapt=false, factor=0.0)

dt = 10.0
xx = ZZB.trajectory(discretize(out1, dt))
using Makie
using CairoMakie
S = T*(0:n)/(n+1)
p1 = lines(S, [dotψ(xx.x[end], s, L, T) for s in S], linewidth=0.3)
for ξ in xx.x[1:5:end]
    lines!(p1, S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.1, alpha = 0.1)
end
display(p1)


#########################################
# Mala for comparison with adaptation   #
#########################################

function get_info(ϕ, ∇ϕ, ξ, L, α, T)
    U = 0.5*dot(ξ, ξ) #gaussian part
    ∇U = deepcopy(ξ) #gaussian part
    dt = T/(2<<L)
    t = dt*0.5
    tt = 0:T/(2<<L):T
    N = Int(T/dt)
    for i in 1:N
        Xₜ = dotψ(ξ, t, L, T)
        U += ϕ(Xₜ, ξ, t, L, α, T)*dt
        for i in 2:(length(ξ)-1)
            l = lvl(i, L)
            k = (i - 1) ÷ (2 << l)
            δ = T / (1 << (L - l))
            if δ*k < t < δ*(k+1)
                ∇U[i] += ∇ϕ(Xₜ, l, t, ξ,  L, α, T)*dt
            end
        end
        t += dt
    end
    ∇U[1] = ∇U[end] = 0.0
    return ∇U, U
end

function mala_sampler(ϕ, ∇ϕ, ξ₀::Vector{Float64}, niter::Int64, args...)
        ξtrace = Array{Float64}(undef, length(ξ₀), niter)
        ξtrace[:,1] = ξ₀
        ∇U₀, U₀ = get_info(ϕ, ∇ϕ, ξ₀, args...)
        ξ₁ = deepcopy(ξ₀)
        τ = 0.015
        count = 0
        for i in 2:niter
                # initial and final point fixed
                for j in 2:length(ξ₀)-1
                    ξ₁[j]  = ξ₀[j] - τ*∇U₀[j] + sqrt(2*τ)*randn()
                end
                ∇U₁, U₁ = get_info(ϕ, ∇ϕ, ξ₁, args...)
                acc_rej =  U₀ - U₁ + (norm(ξ₁ - ξ₀ + τ*∇U₀)^2 - norm(ξ₀ - ξ₁ + τ*∇U₁)^2)/(4τ)
                # @show acc_rej
                # @assert acc_rej == 0.0
                if acc_rej > log(rand())
                        ∇U₀, U₀, ξ₀ = deepcopy(∇U₁), deepcopy(U₁), deepcopy(ξ₁)
                        count += 1
                end
                if i % 100 == 0
                    if count/100 <= 0.7
                        τ = max(0.0001, τ - 0.0001)
                    else
                        τ = min(1.0, τ + 0.0001)
                    end
                    #println("Adaptive step: ar: $(count/100), new tau: $τ")
                    count = 0
                end
                ξtrace[:,i] = ξ₀
        end
        ξtrace
end

# sin application
function ϕ(Xₜ, ξ, t, L, α, T)
    0.5*α*(α*sin(Xₜ)^2 + cos(Xₜ))
end

function ∇ϕ(Xₜ, l, t, ξ,  L, α, T)
    Λ(t, L - l, T)*0.5*(α^2*sin(2.0*Xₜ) - α*sin(Xₜ))
end
using Makie
using CairoMakie

function runall()
    α = 1.0
    L = 6
    T = 50.0
    n = (2 << L) + 1
    ξ₀ =  randn(n)
    u, v = -π, 3π  # initial and fianl point
    ξ₀[1] = u / sqrt(T)
    ξ₀[end] = v / sqrt(T)
    niter = 135000
    dt = 1/(2 << L)
    xx = mala_sampler(ϕ, ∇ϕ, ξ₀, niter, L, α, T)
    S = T*(0:n)/(n+1)
    p1 = lines(S, [dotψ(xx[:,end], s, L, T) for s in S], linewidth=0.3)
    for i in 1:100:size(xx)[2]
        lines!(p1, S, [dotψ(xx[:, i], s, L, T) for s in S], linewidth=0.1, alpha = 0.1)
    end
    display(p1)
end

runall()
