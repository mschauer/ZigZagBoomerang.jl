#################################################################################
# Comparison of Zig-Zag for diffusion bridges with tailored Poisson rates       #
# and with adapted Poisson rate. Reference: https://arxiv.org/abs/2001.05889.   #
#################################################################################

using ZigZagBoomerang, SparseArrays, LinearAlgebra
#using CairoMakie
const ZZB = ZigZagBoomerang
# Drift
const α = 1.5
const L = 7
const T = 50.0
b(x) = α * sin(x)
# First derivative
b′(x) = α * cos(x)
# Second derivative
b″(x) = -α * sin(x)
# Firt Faber Schauder Basis evaluated at time `t`
Λ(t, T::Float64) = sqrt(T) * 0.5 - abs((t % T) / sqrt(T) - sqrt(T) * 0.5)

# Rescaled Faber Schauder Basis evaluated at time `t`
Λ(t, l⁻::Int64, T::Float64) = Λ(t * (1 << l⁻), T) / sqrt(1 << l⁻)

# Linear function for final and initial value of the Bridge
#Λbar(t, T::Float64, final::Val{true})  =  t/T
#Λbar(t, T::Float64, final::Val{false})  = 1 - t/T
#Λ(2, 2T)*sqrt(2/T)

"""
    dotψ(ξ, s, L, T, u, v)
Given the truncated FS expansion with truncation level `L` and
coefficients `ξ`, output the value of the diffuion bridge at time `s` (`r`)
with initial value `u` at time 0 and final value `v` at `T`.
"""
function dotψ(ξ, s, L, T)
    0 <= s < T || error("out of bounds")
    r = s / sqrt(T) * ξ[end] + sqrt(T) * (1 - s / T) * ξ[1]
    for i = 0:L
        j = floor(Int, s / T * (1 << (L - i))) * (2 << i) + (1 << i) + 1
        r += ξ[j] * Λ(s, L - i, T)
    end
    r
end

"""
    dotψmoving(t, ξ, θ, t′, s, F, L, T, u, v)
Jointly updates the coefficeints (locally) and evaluates the diffuion bridge at time `s`.

Given the truncated FS expansion with truncation level `L` and
coefficients `ξ` and velocities `θ`, move first the coefficients required for
the evaluation of the diffuion bridge at time `s` up to time `t′` (according
to the dynamics of the sampler `F`) and output the
value of diffuion bridge at time `s` with initial value `u` at time 0 and final value `v` at `T`.
"""
function dotψmoving(t, ξ, θ, t′, s, F, L, T)
    0 <= s <= T || error("out of bounds")
    r = s / sqrt(T) * ξ[end] + sqrt(T) * (1 - s / T) * ξ[1]
    for i = 0:L
        j = floor(Int, s / T * (1 << (L - i))) * (2 << i) + (1 << i) + 1
        ZigZagBoomerang.smove_forward!(j, t, ξ, θ, t′, F)
        r += ξ[j] * Λ(s, L - i, T)
    end
    r
end

# find level of index i
function lvl_(i)
    l = 0
    while (i & 1) == 0
        l += 1
        i = i >> 1
    end
    l
end

function lvl(i, L)
    if i == 1 || i == (2 << L) + 1
        return L
    else
        return lvl_(i-1)
    end
end

# ↓ not used
"""
Unbiased estimate for the `i`th partial derivative of the potential function.
The variance of the estimate can be reduced by averaging over `K` independent realization.
`ξ` is the current position of the coefficients, `L` the truncation level.
The bridge has initial value `u` at time 0 and final value `v` at `T`.
"""
function ∇ϕ(ξ, i, K, L, T) # formula (17)
    if i == (2 << L) + 1    # final point
        s = T * (rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L, T)
        return 0.5 * sqrt(T) * s * (2b(x) * b′(x) + b″(x)) + ξ[i] - ξ[1]
    elseif i == 1   # initial point
        s = T * (rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L, T)
        return 0.5 * T^(1.5) * (1 - s / T) * (2b(x) * b′(x) + b″(x)) + ξ[i] - ξ[end]
    else
        l = lvl(i, L)
        k = (i - 1) ÷ (2 << l)
        δ = T / (1 << (L - l)) # T/(2^(L-l))
        r = 0.0
        for _ = 1:K
            s = δ * (k + rand())
            x = dotψ(ξ, s, L, T)
            r += 0.5 * δ * Λ(s, L - l, T) * (2b(x) * b′(x) + b″(x)) + ξ[i]
        end
        return r / K
    end
end


"""
    ∇ϕmoving(t, ξ, θ, i, t′, F, L, T, u, v)
Jointly updates the coefficeints (locally) and estimates
the `i`th partial derivative of the potential function.
The bridge has initial value `u` at time 0 and final value `v` at `T`.
"""
function ∇ϕmoving(t, ξ, θ, i, t′, F, L, T) # formula (17)
    if i == (2 << L) + 1 # final point
        s = T * (rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L, T)
        return 0.5 * sqrt(T) * s * (2b(x) * b′(x) + b″(x)) + ξ[i] - ξ[1]
    elseif i == 1 # initial point
        s = T * (rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L, T)
        return 0.5 * T^(1.5) * (1 - s / T) * (2b(x) * b′(x) + b″(x)) + ξ[i] - ξ[end]
    else
        l = lvl(i, L)
        k = (i - 1) ÷ (2 << l)
        δ = T / (1 << (L - l))
        s = δ * (k + rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L, T)
        return 0.5 * δ * Λ(s, L - l, T) * (2b(x) * b′(x) + b″(x)) + ξ[i]
    end
end



# ↓ not used
"""
    ∇ϕ!(y, ξ, k, L,  T, u, v)
In-place evaluation of the gradient of the potential function.
`ξ` is the current position, `k` is the number of MC realization,
`L` is the truncation level. The bridge has initial value `u` at time 0 and final value `v` at `T`.
"""
function ∇ϕ!(y, ξ, k, L, T)
    for i in eachindex(ξ)
        y[i] = ∇ϕ(ξ, i, k, L, T)
    end
    y
end
n = (2 << L) + 1
ξ0 = 0randn(n)
u, v = -π, 3π  # initial and fianl point
ξ0[1] = u / sqrt(T)
ξ0[end] = v / sqrt(T)
c = ones(n)
c[end] = c[1] = 0.0
θ0 = rand((-1.0, 1.0), n)
θ0[end] = θ0[1] = 0.0 # fix final point
T′ = 30000.0 # final clock of the pdmp

Γ = sparse(1.0I, n, n)
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ!, 0.0, ξ0, θ0, T, 10.0, Boomerang(Γ, ξ0*0, 0.1; ρ=0.9), 1, L, adapt=false);
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ, 0.0, ξ0, rand((-1.0, 1.0), n), T, 40.0*ones(n), ZigZag(Γ, ξ0*0), 5, L, adapt=false);
trace, (t, ξ, θ), (acc, num), c = @time spdmp(∇ϕmoving, 0.0, ξ0, θ0, T′, c, ZigZag(Γ, ξ0 * 0),
                        SelfMoving(), L, T, adapt = true);


@show acc/num
######################################################################################
##### Overloafing Poisson times in order to have tighter upperbounds
######################################################################################
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
            b2  θ[i]*(θ[i] - θ[end])
        elseif i == (2 << L) + 1
            a = c[i].c + T^(1.5)*0.5*(α^2 + α) * abs(θ[i])  # final point
            b1 = θ[i]*(x[i] - x[1])
            b2  θ[i]*(θ[i] - θ[1])
        else
            l = lvl(i, L)
            a = c[i].c + T^(1.5) / 2^((L - l) * 1.5 + 2) * (α^2 + α) * abs(θ[i]) # formula (22)
            b1 = x[i] * θ[i]
            b2 = θ[i] * θ[i]
        end
        return a, b1, b2
    end
end

c = [MyBound(0.0) for i in 1:n]
trace, (t, ξ, θ), (acc, num), c = @time spdmp(∇ϕmoving, 0.0, ξ0, θ0, T′, c, ZigZag(Γ, ξ0 * 0),
                        SelfMoving(), L, T, adapt = false);


@show acc/num
ts, ξs = splitpairs(discretize(trace, T′/n))
S = T*(0:n)/(n+1)

error("")

#using CairoMakie
p1 = lines(S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
for ξ in ξs[1:5:end]
    lines!(p1, S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
end
display(p1)


using Plots
using ColorSchemes

t = Vector(0.0:0.1:1.0)
ones(length(t))
color = :RdYlBu

plot(t, [i+ randn() for i in t], color = )

color
