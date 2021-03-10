using SparseArrays
const SA = SparseArrays

"""
    τ = freezing_time(x, θ)

computes the hitting time of a 1d particle with
constant velocity `θ` to hit 0 given the position `x`
"""
function freezing_time(x, θ, F::Union{BouncyParticle, ZigZag})
    if θ*x >= 0 # sic!
        return Inf
    else
        return -x/θ
    end
end



"""
    t, x, θ = ssmove_forward!(t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})

moves forward only the non_frozen particles
"""
function ssmove_forward!(t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})
    for i in eachindex(x)
        if θ[i] != 0.0
            t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
        end
    end
    t, x, θ
end
"""
    t, x, θ = ssmove_forward!(G, i, t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})


moves forward only the non_frozen particles neighbours of i
"""
function ssmove_forward!(G, i, t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})
    nhd = neighbours(G, i)
    for i in nhd
        if θ[i] != 0.0
            t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
        end
    end
    t, x, θ
end

"""
    queue_time!(Q, t, x, θ, i, b, f, Z::ZigZag)

Computes the (proposed) reflection time and the freezing time of the
ith coordinate and enqueue the first one. `f[i] = true` if the next
time is a freezing time.
"""
function queue_time!(Q, t, x, θ, i, b, f, Z::ZigZag)
    trefl = poisson_time(b[i], rand())
    tfreeze = freezing_time(x[i], θ[i], Z)
    if tfreeze <= trefl
        f[i] = true
        Q[i] = t[i] + tfreeze
    else
        f[i] = false
        Q[i] = t[i] + trefl
    end
    return Q
end

"""
    sspdmp_inner!(Ξ, G, G2, ∇ϕ, t, x, θ, Q, c, b, t_old, f, θf, (acc, num),
            F::ZigZag, κ, args...; strong_upperbounds = false, factor=1.5, adapt=false)

Inner loop of the sticky ZigZag sampler. `G[i]` is the set of indices used to derive the
bounding rate λbar_i and `G2` are the indices k in A_j for all j : i in Aj (neighbours of neighbours)

If ∇ϕ is not self moving, then it is assumed that ∇ϕ[x, i] is function of x_i
with i in G[i].
"""
function sspdmp_inner!(Ξ, G, G2, ∇ϕ, t, x, θ, Q, c, b, t_old, f, θf, (acc, num),
        F::ZigZag, κ, args...; structured=true, reversible=false,strong_upperbounds = false, factor=1.5, adapt=false)
    n = length(x)
    # f[i] is true if the next event will be a freeze
    while true
        ii, t′ = peek(Q)
        refresh = ii > n
        i = ii - refresh*n
        refresh && error("refreshment not implemented")
        if f[i] # case 1) to be frozen
            t, x, θ = smove_forward!(i, t, x, θ, t′, F) # move only coordinate i
            if abs(x[i]) > 1e-8
                error("x[i] = $(x[i]) !≈ 0")
            end
            x[i] = -0*θ[i]
            θf[i], θ[i] = θ[i], 0.0 # stop and save speed
            t_old[i] = t[i]
            f[i] = false
            Q[i] = t[i] - log(rand())/κ[i]
            if !strong_upperbounds
                t, x, θ = ssmove_forward!(G, i, t, x, θ, t′, F)
                t, x, θ = ssmove_forward!(G2, i, t, x, θ, t′, F)
                for j in neighbours(G, i)
                    if θ[j] != 0 # only non-frozen, especially not i
                        b[j] = ab(G, j, x, θ, c, F, args...)
                        t_old[j] = t[j]
                        Q = queue_time!(Q, t, x, θ, j, b, f, F)
                    end
                end
            end
        elseif x[i] == 0 && θ[i] == 0 # case 2) was frozen
            t[i] = t′ # equivalent to t, x, θ = smove_forward!(i, t, x, θ, t′, F) # move only coordinate i
            θ[i], θf[i] = θf[i], 0.0 # unfreeze, restore speed
            if reversible
                θ[i] *= rand((-1,1))
            end
            t_old[i] = t[i]
            t, x, θ = ssmove_forward!(G, i, t, x, θ, t′, F) # neighbours
            t, x, θ = ssmove_forward!(G2, i, t, x, θ, t′, F) # neighbours of neightbours \ neighbours
            for j in neighbours(G, i)
                if θ[j] != 0 # only non-frozen, including i # check!
                    b[j] = ab(G, j, x, θ, c, F, args...)
                    t_old[j] = t[j]
                    Q = queue_time!(Q, t, x, θ, j, b, f, F)
                end
            end
        else # was either a reflection time or an event time from the upper bound
            if structured
                t, x, θ = ssmove_forward!(G, i, t, x, θ, t′, F) # neighbours
            elseif length(args) > 1  && args[1] <: SelfMoving
                t, x, θ = ssmove_forward!(G, i, t, x, θ, t′, F) # neighbours
            else
                t, x, θ = ssmove_forward!(t, x, θ, t′, F) # all
            end
            # do it here so ∇ϕ is right event without self moving
            ∇ϕi = ∇ϕ_(∇ϕ, t, x, θ, i, t′, F, args...)
            l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i], t[i] - t_old[i])
            num += 1
            if rand()*lb < l # was a reflection time
                acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small. l/lb = $(l/lb)")
                    acc = num = 0
                    adapt!(c, i, factor)
                end
                # already done above t, x, θ = smove_forward!(G, i, t, x, θ, t′, F) # neighbours
                t, x, θ = ssmove_forward!(G2, i, t, x, θ, t′, F) # neighbours of neightbours \ neighbours
                θ = reflect!(i, ∇ϕi, x, θ, F)
                for j in neighbours(G, i)
                    if θ[j] != 0
                        b[j] = ab(G, j, x, θ, c, F, args...)
                        t_old[j] = t[j]
                        queue_time!(Q, t, x, θ, j, b, f, F)
                    end
                end
            else # was an event time from upperbound -> nothing happens
                b[i] = ab(G, i, x, θ, c, F, args...)
                t_old[i] = t[i]
                queue_time!(Q, t, x, θ, i, b, f, F)
                continue
            end
        end
        push!(Ξ, event(i, t, x, θ, F))
        return t, x, θ, t′, (acc, num), c, b, t_old
    end
end

function sspdmp(∇ϕ, t0, x0, θ0, T, c, F::ZigZag, κ, args...; structured=false, reversible=false,strong_upperbounds = false,
        factor=1.5, adapt=false)
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    t_old = copy(t)
    f = zeros(Bool, n)
    Γ = sparse(F.Γ)
    G = [i => rowvals(Γ)[nzrange(Γ, i)] for i in eachindex(θ0)]
    G2 = [i => setdiff(union((G[j].second for j in G[i].second)...), G[i].second) for i in eachindex(G)]
    x, θ = copy(x0), copy(θ0)
    θf = zero(θ)
    num = acc = 0
    Q = SPriorityQueue{Int,Float64}()
    b = [ab(G, i, x, θ, c, F, args...) for i in eachindex(θ)]
    for i in eachindex(θ)
        trefl = poisson_time(b[i], rand())
        tfreez = freezing_time(x[i], θ[i], F)
        if trefl > tfreez
            f[i] = true
            enqueue!(Q, i => t0 + tfreez)
        else
            f[i] = false
            enqueue!(Q, i => t0 + trefl)
        end
    end
    if hasrefresh(F)
        for i in eachindex(θ)
            enqueue!(Q, (n + i) => waiting_time_ref(F))
        end
    end
    Ξ = Trace(t0, x0, θ0, F)
    while t′ < T
        t, x, θ, t′, (acc, num), c,  b, t_old = sspdmp_inner!(Ξ, G, G2, ∇ϕ, t, x, θ, Q,
                    c, b, t_old, f, θf, (acc, num), F, κ, args...; structured=structured, reversible=reversible,
                    strong_upperbounds = strong_upperbounds , factor=factor,
                    adapt=adapt)
    end
    #t, x, θ = ssmove_forward!(t, x, θ, T, F)
    Ξ, (t, x, θ), (acc, num), c
end
