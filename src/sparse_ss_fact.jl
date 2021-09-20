
function new_queue_time!(Q, t, x, θ, i, b, f, Z::ZigZag)
    trefl = poisson_time(b[i], rand())
    tfreeze = freezing_time(x[i], θ[i], Z)
    if tfreeze <= trefl
        f[i] = true
        enqueue!(Q, i => t[i] + tfreeze)
    else
        f[i] = false
        enqueue!(Q, i => t[i] + trefl)
    end
    return Q
end


function sspdmp_inner!(Ξ, G, G1, G2, ∇ϕ, t, x::SparseVector{Float64, Int64}, θ, Q, c, b, t_old, f, θf, s, ns, (acc, num),
        F::ZigZag, κ::Float64, args...; reversible=false, strong_upperbounds = false, factor=1.5, adapt=false)
    n = length(x)
    # f[i] is true if the next event will be a freeze
    while true
        ii, t′ = dequeue_pair!(Q)
        # println("time is : $(t′)")
        refresh = n < ii <= 2n
        i = ii - refresh*n
        refresh && error("refreshment not implemented")
        if i == 0  # unfreezing time
            # println("unfreezing time")
            i0 = rand(eachindex(s)[s]) # select randomly an index 
            if !(x[i0] == 0 && θ[i0] == 0 && s[i0]  == 1)
                println("x[i] = $(x[i]) !≈ 0")
                dump(x[i0])
                dump(θ[i0])
                dump(s[i0])
                # error("x[i0] = $(x[i0]) and θ[i0] = $(θ[i0]) and s[i0] == $(s[i0])") # check that the particle was frozen
            end
            t[i0] = t′ # equivalent to t, x, θ = smove_forward!(i, t, x, θ, t′, F) # move only coordinate i
            θ[i0], θf[i0] = θf[i0], 0.0 # unfreeze, restore speed
            if reversible
                θ[i0] *= rand((-1,1))
            end
            t, x, θ = ssmove_forward!(G, i0, t, x, θ, t′, F) # neighbours
            t, x, θ = ssmove_forward!(G2, i0, t, x, θ, t′, F) # neighbours of neightbours \ neighbours
            for j in neighbours(G1, i0)
                if θ[j] != 0 # only non-frozen, without including i # check!
                    if j == i0 
                        b[j] = ab(G1, j, x, θ, c, F, args...)
                        t_old[j] = t[j]
                        enqueue!(Q, j => t[j] + poisson_time(b[j], rand())) # cannot be hitting time
                        f[j] = false
                    else
                        b[j] = ab(G1, j, x, θ, c, F, args...)
                        t_old[j] = t[j]
                        Q = queue_time!(Q, t, x, θ, j, b, f, F)
                    end
                end
            end
            s[i0] = 0 # i0 is not stuck anymore
            ns -= 1 # one coordinate less frozen at 0
            @assert ns >= 0
            if ns == 0 # no coordinate frozen
                enqueue!(Q, 0 => Inf)
                # Q[0] = Inf
            else
                enqueue!(Q, 0 => t′ - log(rand())/(κ*ns))
                # Q[0] = t′ - log(rand())/(κ*ns)
            end
        elseif f[i] # case 1) to be frozen
            # println("new particle to be frozen")
            # delete!(Q, i) 
            t, x, θ = smove_forward!(i, t, x, θ, t′, F) # move only coordinate i
            if abs(x[i]) > 1e-8
                error("x[i] = $(x[i]) !≈ 0")
            end
            SparseArrays.dropstored!(x,i)
            θf[i], θ[i] = θ[i], 0.0 # stop and save speed
            t_old[i] = t[i]
            f[i] = false
            ns += 1
            s[i] = 1
            Q[0] = t′ - log(rand())/(κ*ns) # renew sticky time
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
        else # was either a reflection time or an event time from the upper bound
            # println("proposed reflection time")
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
                    if j == i
                        b[j] = ab(G, j, x, θ, c, F, args...)
                        t_old[j] = t[j]
                        new_queue_time!(Q, t, x, θ, j, b, f, F)
                    elseif θ[j] != 0
                        b[j] = ab(G, j, x, θ, c, F, args...)
                        t_old[j] = t[j]
                        queue_time!(Q, t, x, θ, j, b, f, F)
                    end
                end
            else # was an event time from upperbound -> nothing happens
                b[i] = ab(G1, i, x, θ, c, F, args...)
                t_old[i] = t[i]
                new_queue_time!(Q, t, x, θ, i, b, f, F)
                continue
            end
        end
        if i == 0
            push!(Ξ, event(i0, t, x, θ, F))
        else
            push!(Ξ, event(i, t, x, θ, F))
        end 
        return t, x, θ, t′, s, ns, (acc, num), c, b, t_old
    end
end

function sspdmp(∇ϕ, t0, x0::SparseVector{Float64, Int64}, θ0, θf, T, c, G, F::ZigZag, κ::Float64, args...; reversible=false,strong_upperbounds = false,
            factor=1.5, adapt=false, progress=false, progress_stops = 20)
    n = length(x0)

    [@assert θ0[i] == 0.0  for i in findall(iszero, x0)]
    [@assert θf[i] != 0.0 for i in eachindex(x0)]
    t′ = t0
    t = fill(t′, size(θ0)...)
    t_old = copy(t)
    f = zeros(Bool, n) # to be frozen
    s = ones(Bool,  n) # s[i] == 1 if x[i] == 0 θ[i] == 0 (stuck at 0)
    [s[i] = 0  for i in rowvals(x0)]
    [@assert s[i] == 0 for i in rowvals(x0)]
    ns = sum(s)
    Γ = sparse(F.Γ)
    G1 = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    if G === nothing
        G = G1
    end
    @assert all(a.second ⊇ b.second for (a,b) in zip(G, G1))
    G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]
    x, θ = copy(x0), copy(θ0)
    num = acc = 0
    Q = PriorityQueue{Int,Float64}()
    b = [ab(G1, i, x, θ, c, F, args...) for i in eachindex(θ)]
    for i in eachindex(θ)
        if s[i] == 0
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
    end
    enqueue!(Q, 0 => t0 - log(rand())/(κ*ns) )
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
        t, x, θ, t′, s, ns, (acc, num), c,  b, t_old = sspdmp_inner!(Ξ, G, G1, G2, ∇ϕ, t, x, θ, Q,
                    c, b, t_old, f, θf, s, ns, (acc, num), F, κ, args...; reversible=reversible,
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

