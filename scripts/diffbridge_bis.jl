using Makie, ZigZagBoomerang, SparseArrays, LinearAlgebra
using CairoMakie
const ZZB = ZigZagBoomerang
# Drift
const α = 1.5
b(x) = α*sin(x)
# First derivative
b′(x) = α*cos(x)
# Second derivative
b″(x) = -α*sin(x)
# Firt Faber Schauder Basis evaluated at time `t`
Λ(t, T::Float64) = sqrt(T)*0.5 - abs((t % T)/sqrt(T) - sqrt(T)*0.5)

# Rescaled Faber Schauder Basis evaluated at time `t`
Λ(t, l⁻::Int64, T::Float64) = Λ(t*(1<<l⁻), T)/sqrt(1<<l⁻)

# Linear function for final and initial value of the Bridge
Λbar(t, T::Float64, final::Val{true})  =  t/T
Λbar(t, T::Float64, final::Val{false})  = 1 - t/T

"""
    dotψ(ξ, s, L, T, u, v)
Given the truncated FS expansion with truncation level `L` and
coefficients `ξ`, output the value of the diffuion bridge at time `s` (`r`)
with initial value `u` at time 0 and final value `v` at `T`.
"""
function dotψ(ξ, s, L, T, u, v)
    0 <= s <= T || error("out of bounds")
    r = Λbar(s, T, Val(false))*u + Λbar(s, T, Val(true))*v
    for i in 0:L
        j = floor(Int, s/T * (1 << (L - i)))*(2 << i) + (1 << i) #to change
        r += ξ[j]*Λ(s, L-i, T)
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
function dotψmoving(t, ξ, θ, t′, s, F, L, T, u, v)
    0 <= s < T || error("out of bounds")
    r = Λbar(s, T, Val(false))*u + Λbar(s, T, Val(true))*v
    for i in 0:L
        j = floor(Int, s/T*(1 << (L - i)))*(2 << i) + (1 << i) #to change
        ZigZagBoomerang.smove_forward!(j, t, ξ, θ, t′, F)
        r += ξ[j]*Λ(s, L-i, T)
    end
    r
end

# find level of index i
function lvl(i)
    l = 0
    while (i & 1) == 0
        l += 1
        i = i >> 1
    end
    l
end

# ↓ not used
"""
Unbiased estimate for the `i`th partial derivative of the potential function.
The variance of the estimate can be reduced by averaging over `K` independent realization.
`ξ` is the current position of the coefficients, `L` the truncation level.
The bridge has initial value `u` at time 0 and final value `v` at `T`.
"""
function ∇ϕ(ξ, i, K, L, T, u, v) # formula (17)
    l = lvl(i)
    k = i ÷ (2 << l)
    δ = T/(1 << (L-l)) # T/(2^(L-l))
    r = 0.0
    for _ in 1:K
        s = δ*(k + rand())
        x = dotψ(ξ, s, L,  T, u, v)
        r += 0.5*δ*Λ(s, L-l, T)*(2b(x)*b′(x) + b″(x)) + ξ[i]
    end
    r/K
end
"""
    ∇ϕmoving(t, ξ, θ, i, t′, F, L, T, u, v)
Jointly updates the coefficeints (locally) and estimates
the `i`th partial derivative of the potential function.
The bridge has initial value `u` at time 0 and final value `v` at `T`.
"""
function ∇ϕmoving(t, ξ, θ, i, t′, F, L, T, u, v) # formula (17)
    l = lvl(i)
    k = i ÷ (2 << l)
    δ = T/(1 << (L-l))
    s = δ*(k + rand())
    x = dotψmoving(t, ξ, θ, t′, s, F, L,  T, u, v)
    0.5*δ*Λ(s, L-l, T)*(2b(x)*b′(x) + b″(x)) + ξ[i]
end

# ↓ not used
"""
    ∇ϕ!(y, ξ, k, L,  T, u, v)
In-place evaluation of the gradient of the potential function.
`ξ` is the current position, `k` is the number of MC realization,
`L` is the truncation level. The bridge has initial value `u` at time 0 and final value `v` at `T`.
"""
function ∇ϕ!(y, ξ, k, L, T, u, v)
    for i in eachindex(ξ)
        y[i] = ∇ϕ(ξ, i, k, L, T, u, v)
    end
    y
end

######################################################################################
##### Modifying the doubly local algorithm in order to have tighter upperbounds
######################################################################################
"""
    poisson_time(a, b, c, u)
Obtaining waiting time for inhomogeneous Poisson Process
with rate of the form λ(t) = a + (b + c*t)^+, where `c`,`a`> 0 ,`b` ∈ R, `u` uniform random variable
"""
function poisson_time(a, b, c, u) # formula (22)
    if b>0
        return (-(b+a) + sqrt((b + a)^2 - 2.0*c*log(u)))/c # positive solution of quadratic equation c*0.5 x^2 + (b + a) x + log(u) = 0
    elseif a*b/c <= log(u)
        return -log(u)/a
    else
        return (-(a + b) + sqrt((a + b)^2 - 2.0*c*(b*b*0.5/c + log(u))))/c    # # positive solution of quadratic equation c*0.5 x^2 + (b + a) x + log(u) + b*b*0.5/c = 0
    end
end


"""
    abc(G, i, x, θ, c, Flow)

Returns the constant term `a` and linear term `b` when computing the Poisson times
from the upper upper bounding rates λᵢ(t) = max(a + b*t)^2. The factors `a` and `b`
can be function of the current position `x`, velocity `θ`, tuning parameter `c` and
the Graph `G`
"""
function abc(G, i, x, θ, Z::ZigZag, L::Int64, T::Float64)
    l = lvl(i)
    a = T^(1.5)/2^((L-l)*1.5 + 2)*(α^2 + α)*abs(θ[i]) # formula (22)
    b = x[i]*θ[i]
    c = θ[i]*θ[i]
    a, b, c
end

function spdmp_inner_specific!(Ξ, G, G2, ∇ϕ, t, x, θ, Q, a, b, c, t_old, (acc, num),
     F::Union{ZigZag,FactBoomerang}, args...; adaptscale = false)
    n = length(x)
    while true
        ii, t′ = ZZB.peek(Q)
        refresh = ii > n
        i = ii - refresh*n
        t, x, θ = ZZB.smove_forward!(G, i, t, x, θ, t′, F)
        if refresh
            t, x, θ = ZZB.smove_forward!(G2, i, t, x, θ, t′, F)
            if adaptscale
                effi = (1 + 2*F.ρ/(1 - F.ρ))
                τ = effi/(t[i]*F.λref)
                if τ < 0.2
                    F.σ[i] = F.σ[i]*exp(((0.3t[i]/acc[i] > 1.66) - (0.3t[i]/acc[i] < 0.6))*0.03*min(1.0, sqrt(τ/F.λref)))
                end
            end
            if F isa ZigZag
                θ[i] = F.σ[i]*rand((-1,1))
            else
                θ[i] = F.ρ*θ[i] + F.ρ̄*F.σ[i]*randn()
            end
            #renew refreshment
            Q[(n + i)] = t[i] + ZZB.waiting_time_ref(F)
            #update reflections
            for j in ZZB.neighbours(G, i)
                a[j], b[j], c[j] = abc(G, j, x, θ, F, L, t)
                t_old[j] = t[j]
                Q[j] = t[j] + poisson_time(a[j], b[j], c[j], rand())
            end
        else
            l, lb = ZZB.sλ(∇ϕ, i, t, x, θ, t′, F, args...), a[i] + ZZB.pos(b[i] + c[i]*(t[i] - t_old[i]))
            num += 1
            if rand()*lb < l
                acc += 1
                if l >= lb
                    println("a : $(a[i]), b : $(b[i]), c : $(c[i])")
                    println("lb : $(lb) and l : $(l)")
                    error("out of bounds")
                end
                t, x, θ = ZZB.smove_forward!(G2, i, t, x, θ, t′, F)
                θ = ZZB.reflect!(i, x, θ, F)
                for j in ZZB.neighbours(G, i)
                    a[j], b[j], c[j] = abc(G, j, x, θ, F, L, T)
                    t_old[j] = t[j]
                    Q[j] = t[j] + poisson_time(a[j], b[j], c[j], rand())
                end
            else
                a[i], b[i], c[i] = abc(G, i, x, θ, F, L, T)
                t_old[i] = t[i]
                Q[i] = t[i] + poisson_time(a[i], b[i], c[i], rand())
                continue
            end
        end
        push!(Ξ, ZZB.event(i, t, x, θ, F))
        return t, x, θ, t′, (acc, num), a, b, c, t_old
    end
end

function spdmp_specific(∇ϕ, t0, x0, θ0, T, F::Union{ZigZag,FactBoomerang}, args...; adaptscale=false)
    #sparsity graph
    a = zero(x0)
    b = zero(x0)
    c = zero(x0)
    t_old = zero(x0)
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    G2 = [i => setdiff(union((G[j].second for j in G[i].second)...), G[i].second) for i in eachindex(G)]
    x, θ = copy(x0), copy(θ0)
    num = acc = 0
    Q = ZZB.SPriorityQueue{Int,Float64}()
    for i in eachindex(θ)
        a[i], b[i], c[i] = abc(G, i, x, θ, F, L, T)
        t_old[i] = t[i]
        ZZB.enqueue!(Q, i =>poisson_time(a[i], b[i], c[i], rand()))
    end
    if ZZB.hasrefresh(F)
        for i in eachindex(θ)
            ZZB.enqueue!(Q, (n + i)=>waiting_time_ref(F))
        end
    end
    Ξ = ZZB.Trace(t0, x0, θ0, F)
    while t′ < T
        t, x, θ, t′, (acc, num), a, b, c, t_old = spdmp_inner_specific!(Ξ, G, G2, ∇ϕ, t, x, θ, Q,
                     a, b, c, t_old, (acc, num), F, args...; adaptscale=adaptscale)
    end
    #t, x, θ = smove_forward!(t, x, θ, T, F)
    Ξ, (t, x, θ), (acc, num)
end




L = 6
n = (2 << L) - 1
u = -float(π)
v = float(3π)
T = 50.0 # length diffusion bridge
ξ0 = 0randn(n)
θ0 = randn(n)
T′ = 20000.0 # final clock of the pdmp

Γ = sparse(1.0I, n, n)
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ!, 0.0, ξ0, θ0, T, 10.0, Boomerang(Γ, ξ0*0, 0.1; ρ=0.9), 1, L, adapt=false);
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ, 0.0, ξ0, rand((-1.0, 1.0), n), T, 40.0*ones(n), ZigZag(Γ, ξ0*0), 5, L, adapt=false);
trace, (t, ξ, θ), (acc, num) = @time spdmp_specific(∇ϕmoving, 0.0, ξ0,
    rand((-1.0, 1.0), n), T′, ZigZag(Γ, ξ0*0), SelfMoving(), L, T, u, v);
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ, 0.0, ξ0, rand((-1.0,1.0), n), T, 100.0*ones(n), FactBoomerang(Γ, ξ0*0, 0.1), 5, L, adapt=false);

ts, ξs = splitpairs(discretize(trace, 1.0))
S = T*(0:n)/(n+1)
error("")


p1 = lines(S, [dotψ(ξ, s, L, T, u, v) for s in S],  color=(:purple, 0.1))
for ξ in ξs[1:10:end]
    lines!(p1, S, [dotψ(ξ, s, L, T, u, v) for s in S], linewidth=0.3,  color=(:purple, 0.1))
end
display(p1)

p2 = surface([dotψ(ξ, s, L, T, u, v) for s in S, ξ in ξs], shading=false, show_axis=false, colormap = :deep)
scale!(p2, 1.0, 1.0, 100.)

p3 = hbox([lines(ts, getindex.(ξs, i)) for i in [1,2,4,8,16,(n+1)÷2]]...)

save("diffbridges.png", p1)
vbox(p1, p2, p3)
