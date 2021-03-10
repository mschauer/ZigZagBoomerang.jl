function smove_forward!(G, i, t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})
    nhd = neighbours(G, i)
    for i in nhd
        t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
    end
    t, x, θ
end
function smove_forward!(i::Int, t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})
    t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
    return t, x, θ
end

function smove_forward!(t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})
    for i in eachindex(x)
        t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
    end
    t, x, θ
end
function smove_forward!(G, i, t, x, θ, t′, B::Union{Boomerang, FactBoomerang})
    nhd = neighbours(G, i)
    for i in nhd
        τ = t′ - t[i]
        s, c = sincos(τ)
        t[i], x[i], θ[i] = t′, (x[i] - B.μ[i])*c + θ[i]*s + B.μ[i],
                    -(x[i] - B.μ[i])*s + θ[i]*c
    end
    t, x, θ
end

function event(i, t::Vector, x, θ, Z::Union{ZigZag,FactBoomerang})
    t[i], i, x[i], θ[i]
end

"""
    SelfMoving()

Indicates as `args[1]` that `∇ϕ` depends only on few coeffients
and takes responsibility to call `smove_forward!`
"""
struct SelfMoving
end
export SelfMoving
∇ϕ_(∇ϕ, t, x, θ, i, t′, Z, args...) = ∇ϕ(x, i, args...)
∇ϕ_(∇ϕ, t, x, θ, i, t′, Z, S::SelfMoving, args...) = ∇ϕ(t, x, θ, i, t′, Z, args...)
sλ(∇ϕi, i, x, θ, Z::Union{ZigZag,FactBoomerang}) = λ(∇ϕi, i, x, θ, Z)
sλ̄((a,b), Δt) = pos(a + b*Δt)

function spdmp_inner!(Ξ, G, G2, ∇ϕ, t, x, θ, Q, c, b, t_old, (acc, num),
     F::Union{ZigZag,FactBoomerang}, args...; structured=true, factor=1.5, adapt=false, adaptscale=false)
    n = length(x)
    while true
        ii, t′ = peek(Q)
        refresh = ii > n
        i = ii - refresh*n
        t, x, θ = smove_forward!(G, i, t, x, θ, t′, F)
        if refresh
            t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
            if adaptscale
                effi = (1 + 2*F.ρ/(1 - F.ρ))
                τ = effi/(t[i]*F.λref)
                if τ < 0.2
                    F.σ[i] = F.σ[i]*exp(((0.3t[i]/acc[i] > 1.66) - (0.3t[i]/acc[i] < 0.6))*0.03*min(1.0, sqrt(τ/F.λref)))
                end
            end
            if F isa ZigZag && eltype(θ) <: Number
                θ[i] = F.σ[i]*rand((-1,1))
            else
                θ[i] = F.ρ*θ[i] + F.ρ̄*F.σ[i]*randn(eltype(θ))
            end
            #renew refreshment
            Q[(n + i)] = t[i] + waiting_time_ref(F)
            #update reflections
            for j in neighbours(G, i)
                b[j] = ab(G, j, x, θ, c, F)
                t_old[j] = t[j]
                Q[j] = t[j] + poisson_time(b[j], rand())
            end
        else
            if structured || (length(args) > 1  && args[1] isa SelfMoving)
                # neighbours have moved (and all others are not our responsibility)
            else
                t, x, θ = smove_forward!(t, x, θ, t′, F)
            end

            ∇ϕi = ∇ϕ_(∇ϕ, t, x, θ, i, t′, F, args...)
            l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i], t[i] - t_old[i])
            num += 1
            if rand()*lb < l
                acc += 1
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    acc = num = 0
                    adapt!(c, i, factor)
                end
                t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
                θ = reflect!(i, ∇ϕi, x, θ, F)
                for j in neighbours(G, i)
                    b[j] = ab(G, j, x, θ, c, F)
                    t_old[j] = t[j]
                    Q[j] = t[j] + poisson_time(b[j], rand())
                end
            else
                b[i] = ab(G, i, x, θ, c, F)
                t_old[i] = t[i]
                Q[i] = t[i] + poisson_time(b[i], rand())
                continue
            end
        end
        push!(Ξ, event(i, t, x, θ, F))
        return t, x, θ, t′, (acc, num), c, b, t_old
    end
end

"""
    spdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, args...;
        factor=1.5, adapt=false)
        = Ξ, (t, x, θ), (acc, num), c

Version of spdmp which assumes that `i` only depends on coordinates
`x[j] for j in neighbours(G, i)`.

It returns a `FactTrace` (see [`Trace`](@ref)) object `Ξ`, which can be collected
into pairs `t => x` of times and locations and discretized with `discretize`.
Also returns the `num`ber of total and `acc`epted Poisson events and updated bounds
`c` (in case of `adapt==true` the bounds are multiplied by `factor` if they turn
out to be too small.) The final time, location and momentum at `T` can be obtained
with `smove_forward!(t, x, θ, T, F)`.
"""
function spdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, args...;
        factor=1.8, structured=false, adapt=false, adaptscale=false)
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    t_old = copy(t)
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    G2 = [i => setdiff(union((G[j].second for j in G[i].second)...), G[i].second) for i in eachindex(G)]
    x, θ = copy(x0), copy(θ0)
    num = acc = 0
    Q = SPriorityQueue{Int,Float64}()
    b = [ab(G, i, x, θ, c, F) for i in eachindex(θ)]
    for i in eachindex(θ)
        enqueue!(Q, i => poisson_time(b[i], rand()))
    end
    if hasrefresh(F)
        for i in eachindex(θ)
            enqueue!(Q, (n + i) => waiting_time_ref(F))
        end
    end
    Ξ = Trace(t0, x0, θ0, F)
    while t′ < T
        t, x, θ, t′, (acc, num), c,  b, t_old = spdmp_inner!(Ξ, G, G2, ∇ϕ, t, x, θ, Q,
                    c, b, t_old, (acc, num), F, args...; structured=structured, factor=factor, adapt=adapt, adaptscale=adaptscale)
    end
    #t, x, θ = smove_forward!(t, x, θ, T, F)
    Ξ, (t, x, θ), (acc, num), c
end
