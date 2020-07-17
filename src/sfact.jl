function smove_forward!(G, i, t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})
    nhd = neighbours(G, i)
    for i in nhd
        t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
    end
    t, x, θ
end
function smove_forward!(i::Int, t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})
    t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
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
        t[i], x[i], θ[i] = t′, (x[i] - B.μ[i])*cos(τ) + θ[i]*sin(τ) + B.μ[i],
                    -(x[i] - B.μ[i])*sin(τ) + θ[i]*cos(τ)
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
sλ(∇ϕ, i, t, x, θ, t′, Z::Union{ZigZag,FactBoomerang}, args...) = λ(∇ϕ, i, x, θ, Z, args...)
function sλ(∇ϕ, i, t, x, θ, t′, Z::Union{ZigZag,FactBoomerang}, ::SelfMoving, args...)
    pos(∇ϕ(t, x, θ, i, t′, Z, args...)*θ[i]) # needs to call smove_forward
end

function spdmp_inner!(Ξ, G, G2, ∇ϕ, t, x, θ, Q, c, a, b, t_old, (acc, num),
     F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false, adaptscale=false)
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
            if F isa ZigZag
                θ[i] = F.σ[i]*rand((-1,1))
            else
                θ[i] = F.ρ*θ[i] + F.ρ̄*F.σ[i]*randn()
            end
            #renew refreshment
            Q[(n + i)] = t[i] + waiting_time_ref(F)
            #update reflections
            for j in neighbours(G, i)
                a[j], b[j] = ab(G, j, x, θ, c, F)
                t_old[j] = t[j]
                Q[j] = t[j] + poisson_time(a[j], b[j], rand())
            end
        else
            l, lb = sλ(∇ϕ, i, t, x, θ, t′, F, args...), pos(a[i] + b[i]*(t[i] - t_old[i]))
            num += 1
            if rand()*lb < l
                acc += 1
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    acc = num = 0
                    c[i] *= factor
                end
                t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
                θ = reflect!(i, x, θ, F)
                for j in neighbours(G, i)
                    a[j], b[j] = ab(G, j, x, θ, c, F)
                    t_old[j] = t[j]
                    Q[j] = t[j] + poisson_time(a[j], b[j], rand())
                end
            else
                a[i], b[i] = ab(G, i, x, θ, c, F)
                t_old[i] = t[i]
                Q[i] = t[i] + poisson_time(a[i], b[i], rand())
                continue
            end
        end
        push!(Ξ, event(i, t, x, θ, F))
        return t, x, θ, t′, (acc, num), c,  a, b, t_old
    end
end

"""
    spdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, args...;
        factor=1.5, adapt=false)
        = Ξ, (t, x, θ), (acc, num), c, a, b, t_old

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
        factor=1.8, adapt=false, adaptscale=false)
    #sparsity graph
    a = zero(x0)
    b = zero(x0)
    t_old = zero(x0)
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    G2 = [i => setdiff(union((G[j].second for j in G[i].second)...), G[i].second) for i in eachindex(G)]
    x, θ = copy(x0), copy(θ0)
    num = acc = 0
    Q = SPriorityQueue{Int,Float64}()
    for i in eachindex(θ)
        a[i], b[i] = ab(G, i, x, θ, c, F)
        t_old[i] = t[i]
        enqueue!(Q, i =>poisson_time(a[i], b[i] , rand()))
    end
    if hasrefresh(F)
        for i in eachindex(θ)
            enqueue!(Q, (n + i)=>waiting_time_ref(F))
        end
    end
    Ξ = Trace(t0, x0, θ0, F)
    while t′ < T
        t, x, θ, t′, (acc, num), c,  a, b, t_old = spdmp_inner!(Ξ, G, G2, ∇ϕ, t, x, θ, Q,
                    c, a, b, t_old, (acc, num), F, args...; factor=factor, adapt=adapt, adaptscale=adaptscale)
    end
    #t, x, θ = smove_forward!(t, x, θ, T, F)
    Ξ, (t, x, θ), (acc, num), c
end
