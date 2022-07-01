
function ab(G, i, x, θ, C::LocalBound, ∇ϕi, vi, Z::ZigZag, args...)
    a = C.c[i] + ∇ϕi'*θ[i] 
    b = C.c[i]/100 + vi
    a, b, 2.0/C.c[i]/abs(θ[i])
end
#(idot(Z.Γ, i, x)  - idot(Z.Γ, i, Z.μ))'*θ[i]),  θ[i]'*idot(Z.Γ, i, θ)


function spdmp_inner!(rng, G, G2, ∇ϕ, t, x, θ, Q, c::LocalBound, (b, renew), t_old, (acc, num),
 F::Union{ZigZag,FactBoomerang,JointFlow}, args...; factor=1.5, adapt=false, adaptscale=false)
    n = length(x)
    while true
        i, t′ = peek(Q)
     #   @show i
        refresh = i > n
        if refresh
            i = rand(1:n)
        end
        if refresh && F isa JointFlow
            t, x, θ = smove_forward!(G, i, t, x, θ, t′, F)
            t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
            # FIXME: `refresh` is not defined
            x[i], θ[i] = ZigZagBoomerang.refresh(rng, x[i], θ[i], F[i])
            Q[(n + 1)] = t′ + waiting_time_ref(rng, F) #renew refreshment time
            for j in neighbours(G, i)
                ∇ϕj, vj = ∇ϕ(t, x, θ, j, t′, F, args...)
                b[j] = ab(G, j, x, θ, c, ∇ϕj, vj, F)
                t_old[j] = t[j]
                τ, renew[j] = next_time(t[j], b[j], rand(rng))
                Q[j] = τ
            end
         #   @show x, θ, i, t′
        elseif renew[i]
            t, x, θ = smove_forward!(G, i, t, x, θ, t′, F)
            ∇ϕi, vi = ∇ϕ(t, x, θ, i, t′, F, args...)
            b[i] = ab(G, i, x, θ, c, ∇ϕi, vi, F)
            t_old[i] = t[i]
            τ, renew[i] = next_time(t[i], b[i], rand(rng))
            Q[i] = τ
            continue
        else
            t, x, θ = smove_forward!(G, i, t, x, θ, t′, F)
            ∇ϕi, vi = ∇ϕ(t, x, θ, i, t′, F, args...)
            l, lb = sλ(∇ϕi, i, x, θ, F), pos(b[i][1] + b[i][2]*(t[i] - t_old[i]))
            #@show i, θ, l, lb
            num += 1
            if rand(rng)*lb < l
                acc[i] += 1
                if l >= lb
                    if !adapt
                        @show t[i] - t_old[i]
                        @show ∇ϕi, vi, l, lb, b
                        error("Tuning parameter `c = $(c.c[i])` too small.")
                    end
                    c.c[i] = c.c[i]*factor
                end
                t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
                θ = reflect!(i, ∇ϕi, x, θ, F)
                #∇ϕi, vi = ∇ϕ(t, x, θ, i, t′, F, args...)
                for j in neighbours(G, i)
                    ∇ϕj, vj = ∇ϕ(t, x, θ, j, t′, F, args...)
                    b[j] = ab(G, j, x, θ, c, ∇ϕj, vj, F)
                    t_old[j] = t[j]
                    τ, renew[j] = next_time(t[j], b[j], rand(rng))
                    Q[j] = τ
                end
            else
                b[i] = ab(G, i, x, θ, c, ∇ϕi, vi, F)
                t_old[i] = t[i]
                τ, renew[i] = next_time(t[i], b[i], rand(rng))
                Q[i] = τ
                continue
            end
        end
        return event(i, t, x, θ, F), t, x, θ, t′, (acc, num), c, (b, renew), t_old
    end
end

"""
spdmp(∇ϕ, t0, x0, θ0, T, c, [G,] F::Union{ZigZag,FactBoomerang}, args...;
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
function spdmp(∇ϕ, t0, x0, θ0, T, C::LocalBound, G_, F::Union{ZigZag,FactBoomerang,JointFlow}, args...;
    factor=1.8, adapt=false, adaptscale=false, progress=false, progress_stops = 20, seed=Seed())
    n = length(x0)

    rng = Rng(seed)
    t′ = t0
    t = fill(t′, size(θ0)...)
    t_old = copy(t)
    if G_ == All()
        G = [i=>collect(1:n) for i in 1:n]
        G2 = nothing
    else
        G = G_
        G2 = [i => setdiff(union((G[j].second for j in G[i].second)...), G[i].second) for i in eachindex(G)]
    end
    x, θ = copy(x0), copy(θ0)
    num = 0
    acc = zeros(Int, length(θ))
    Q = SPriorityQueue{Int,Float64}()
    ∇ϕi, vi = ∇ϕ(t, x, θ, 1, t′, F, args...)
    b = fill(ab(G, 1, x, θ, C, ∇ϕi, vi, F), n)
    renew = zeros(Bool, n)
    for i in eachindex(θ)
        ∇ϕi, vi = ∇ϕ(t, x, θ, i, t′, F, args...)
        b[i] = ab(G, i, x, θ, C, ∇ϕi, vi, F)
        τ, renew[i] = next_time(t[i], b[i], rand(rng))
        enqueue!(Q, i => τ)
    end
    if hasrefresh(F)
        enqueue!(Q, (n + 1) => waiting_time_ref(F))
    end
    if progress
        prg = Progress(progress_stops, 1)
    else
        prg = missing
    end
    stops = ismissing(prg) ? 0 : max(prg.n - 1, 0) # allow one stop for cleanup
    tstop = T/stops
    Ξ = Trace(t0, x0, θ0, F)
    while t′ < T
        ev, t, x, θ, t′, (acc, num), C, (b, renew), t_old = spdmp_inner!(rng, G, G2, ∇ϕ, t, x, θ, Q,
                    C, (b, renew), t_old, (acc, num), F, args...; factor=factor, adapt=adapt, adaptscale=adaptscale)
        push!(Ξ, ev)
        if t′ > tstop
            tstop += T/stops
            next!(prg) 
        end  
    end
    ismissing(prg) || ProgressMeter.finish!(prg)
    #t, x, θ = smove_forward!(t, x, θ, T, F)
    Ξ, (t, x, θ), (acc, num), C
end

spdmp(∇ϕ, t0, x0, θ0, T, C::LocalBound, F::Union{ZigZag,FactBoomerang,JointFlow}, args...; kargs...) = spdmp(∇ϕ, t0, x0, θ0, T, C, [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)], F, args...; kargs...) 
pdmp(∇ϕ, t0, x0, θ0, T, C::LocalBound, F::Union{ZigZag,FactBoomerang,JointFlow}, args...; kargs...) = spdmp(∇ϕ, t0, x0, θ0, T, C, All(), F, args...; kargs...) 
