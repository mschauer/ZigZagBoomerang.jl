
function freezing_time(x, θ)
    if θ*x >= 0 # sic!
        return Inf
    else
        return -x/θ
    end
end

function queue_time!(Q, t, x, θ, i, b, f, κ)
    trefl = poisson_time(b[i], rand())
    tfreeze = freezing_time(x[i], θ[i])
    if tfreeze <= trefl
        f[i] = true
        Q[i] = t[i] + tfreeze
    else
        f[i] = false
        Q[i] = t[i] + trefl
    end
    return
end



function sspdmp_inner!(Ξ, G, G2, ∇ϕ, t, x, θ, Q, c, b, t_old, f, θf, κ, (acc, num),
     F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)
    n = length(x)
    while true
        ii, t′ = peek(Q)
        refresh = ii > n
        i = ii - refresh*n
        refresh && error("refreshment not implemented")
        t, x, θ = smove_forward!(G, i, t, x, θ, t′, F)
        if f[i] # to be frozen
            if abs(x[i]) > 1e-8
                error("x[i] = $(x[i]) !≈ 0")
            end
            x[i] = -0*θ[i]
            θf[i], θ[i] = θ[i], 0.0 # stop and save speed
            t_old[i] = t[i]
            f[i] = false
            Q[i] = t[i] - log(rand())/κ
            t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
            for j in neighbours(G, i)
                if θ[j] != 0 # only non-frozen, especially not i
                    b[j] = ab(G, j, x, θ, c, F)
                    t_old[j] = t[j]
                    queue_time!(Q, t, x, θ, j, b, f, κ)
                end
            end
        elseif x[i] == 0 && θ[i] == 0 # was frozen
            θ[i], θf[i] = θf[i], 0.0 # unfreeze, restore speed
            t_old[i] = t[i]
            t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
            for j in neighbours(G, i)
                if θ[j] != 0 # only non-frozen
                    b[j] = ab(G, j, x, θ, c, F)
                    t_old[j] = t[j]
                    queue_time!(Q, t, x, θ, j, b, f, κ)
                end
            end
        else
            l, lb = sλ(∇ϕ, i, t, x, θ, t′, F, args...), sλ̄(b[i], t[i] - t_old[i])
            num += 1
            if rand()*lb < l
                acc += 1
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    acc = num = 0
                    adapt!(c, i, factor)
                end
                t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
                θ = reflect!(i, x, θ, F)
                for j in neighbours(G, i)
                    if θ[j] != 0
                        b[j] = ab(G, j, x, θ, c, F)
                        t_old[j] = t[j]
                        queue_time!(Q, t, x, θ, j, b, f, κ)
                    end
                end
            else
                b[i] = ab(G, i, x, θ, c, F)
                t_old[i] = t[i]
                queue_time!(Q, t, x, θ, i, b, f, κ)
                continue
            end
        end
        push!(Ξ, event(i, t, x, θ, F))
        return t, x, θ, t′, (acc, num), c, b, t_old
    end
end

function sspdmp(∇ϕ, t0, x0, θ0, T, c, κ, F::Union{ZigZag,FactBoomerang}, args...;
        factor=1.8, adapt=false)
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    t_old = copy(t)
    f = zeros(Bool, n)
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    G2 = [i => setdiff(union((G[j].second for j in G[i].second)...), G[i].second) for i in eachindex(G)]
    x, θ = copy(x0), copy(θ0)
    θf = zero(θ)
    num = acc = 0
    Q = SPriorityQueue{Int,Float64}()
    b = [ab(G, i, x, θ, c, F) for i in eachindex(θ)]
    for i in eachindex(θ)
        trefl = poisson_time(b[i], rand())
        tfreez = freezing_time(x[i], θ[i])
        if trefl > tfreez
            f[i] = true
            enqueue!(Q, i => tfreez)
        else
            f[i] = false
            enqueue!(Q, i => trefl)
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
                    c, b, t_old, f, θf, κ, (acc, num), F, args...; factor=factor, adapt=adapt)
    end
    #t, x, θ = ssmove_forward!(t, x, θ, T, F)
    Ξ, (t, x, θ), (acc, num), c
end
