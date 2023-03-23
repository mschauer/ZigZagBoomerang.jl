

function sspdmp_inner2!(Ξ, G, G1, G2, ∇ϕ, t, x, θ, Q, c, b, t_old, f, θf, (acc, num),
        F::ZigZag, κ::Vector{Float64}, mode, args...; strong_upperbounds = false, factor=1.5, adapt=false)
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
                for j in neighbours(G1, i)
                    if θ[j] != 0 # only non-frozen, especially not i
                        b[j] = ab(G1, j, x, θ, c, F, args...)
                        t_old[j] = t[j]
                        Q = queue_time!(Q, t, x, θ, j, b, f, F)
                    end
                end
            end
        elseif x[i] == 0 && θ[i] == 0 # case 2) was frozen
            t[i] = t′ # equivalent to t, x, θ = smove_forward!(i, t, x, θ, t′, F) # move only coordinate i
            θ[i], θf[i] = θf[i], 0.0 # unfreeze, restore speed
            if mode[i] == :restore
                continue
            elseif mode[i] == :refresh
                θ[i] *= rand((-1,1))
            elseif mode[i] == :reflect
                θ[i] *= -1.0
            else
                error("the mode = $(mode[i]) selected is not implemented")
            end
            t_old[i] = t[i]
            t, x, θ = ssmove_forward!(G, i, t, x, θ, t′, F) # neighbours
            t, x, θ = ssmove_forward!(G2, i, t, x, θ, t′, F) # neighbours of neightbours \ neighbours
            for j in neighbours(G1, i)
                if θ[j] != 0 # only non-frozen, including i # check!
                    b[j] = ab(G1, j, x, θ, c, F, args...)
                    t_old[j] = t[j]
                    Q = queue_time!(Q, t, x, θ, j, b, f, F)
                end
            end
        else # was either a reflection time or an event time from the upper bound
            t, x, θ = ssmove_forward!(G, i, t, x, θ, t′, F) # neighbours
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
                # have not moved yet
                t, x, θ = ssmove_forward!(G2, i, t, x, θ, t′, F) # neighbours of neightbours \ neighbours
                θ = reflect!(i, ∇ϕi, x, θ, F)
                for j in neighbours(G1, i)
                    if θ[j] != 0
                        b[j] = ab(G, j, x, θ, c, F, args...)
                        t_old[j] = t[j]
                        queue_time!(Q, t, x, θ, j, b, f, F)
                    end
                end
            else # was an event time from upperbound -> nothing happens
                b[i] = ab(G1, i, x, θ, c, F, args...)
                t_old[i] = t[i]
                queue_time!(Q, t, x, θ, i, b, f, F)
                continue
            end
        end
        push!(Ξ, event(i, t, x, θ, F))
        return t, x, θ, t′, (acc, num), c, b, t_old
    end
end

function sspdmp2(∇ϕ, t0, x0, θ0, T, c, G, F::ZigZag, κ, mode, args...;strong_upperbounds = false,
        factor=1.5, adapt=false, progress=false, progress_stops = 20)
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    t_old = copy(t)
    f = zeros(Bool, n)
    Γ = sparse(F.Γ)
    G1 = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    if G === nothing
        G = G1
    end
    @assert all(a.second ⊇ b.second for (a,b) in zip(G, G1))
    G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]
    x, θ = copy(x0), copy(θ0)
    θf = zero(θ)
    num = acc = 0
    Q = SPriorityQueue{Int,Float64}()
    b = [ab(G1, i, x, θ, c, F, args...) for i in eachindex(θ)]
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
    if progress
        prg = Progress(progress_stops, 1)
    else
        prg = missing
    end
    stops = ismissing(prg) ? 0 : max(prg.n - 1, 0) # allow one stop for cleanup
    tstop = T/stops
    while t′ < T
        t, x, θ, t′, (acc, num), c,  b, t_old = sspdmp_inner2!(Ξ, G, G1, G2, ∇ϕ, t, x, θ, Q,
                    c, b, t_old, f, θf, (acc, num), F, κ, mode, args...; 
                    strong_upperbounds = strong_upperbounds , factor=factor,
                    adapt=adapt)
        if t′ > tstop
            tstop += T/stops
            next!(prg) 
        end  
    end
    ismissing(prg) || ProgressMeter.finish!(prg)
    #t, x, θ = ssmove_forward!(t, x, θ, T, F)
    Ξ, (t, x, θ), (acc, num), c
end

sspdmp2(∇ϕ, t0, x0, θ0, T, c, F, κ, args...; kwargs...) = sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, F, κ, args...;  kwargs...)




