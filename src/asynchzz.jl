#
struct StrongUpperBounds{TG,Tc,Tmult}
    G::TG
    adapt::Bool
    c::Tc
    multiplier::Tmult
end

function ab(rng, su::StrongUpperBounds, flow, target::StructuredTarget, t′, u, j)
    t, x, v = u
    ssmove_forward!(target.G, j, t, x, v, t′, flow.old) 
    ∇ϕi = target(t′, u, j, flow) 
    ti, xi, vi = geti(u, j)
    a = su.c[j] + ∇ϕi'*vi
    b = 0.0
    s = su.adapt ? (1.0 - rand(rng)^2) : 1.0
    ti, a, b, ti + s/su.c[j]
end

function ab(rng, su::StrongUpperBounds, flow, ∇ϕi, t′, u, j)
    ti, xi, θi = geti(u, j)
    a = su.c[j] + ∇ϕi'*θi
    b = 0.0
    s = su.adapt ? (1.0 - rand(rng)^2) : 1.0
    ti, a, b, ti + s/su.c[j]
end






function queue_time!(rng, q::PartialQueue, u, i, b, action, barriers::Vector, flow::StickyFlow; enqueue=false)
    t, x, v = u
    trefl = poisson_time(t[i], b[i], rand(rng)) 
    thit = t[i] + hitting_time(barriers[i], geti(u, i), flow)
    trefresh = b[i][end]
    τ, k = findmin((trefresh, trefl, thit))
    q[i] = τ
    action[i] = (renew, reflect, hit)[k]
    return q
end




function asynchzz(u0, target::StructuredTarget, flow::StickyFlow, upper_bounds::StrongUpperBounds, barriers::Vector{<:StickyBarriers}, end_condition;  progress=false, progress_stops = 20, rng=Rng(Seed()))
    u = deepcopy(u0)
    # Initialize
    (t, x, v) = u
    d = length(v)
    t′ = maximum(t)
    v_old = copy(v)

    G = target.G
   # G1 = upper_bounds.G1
   # G2 = upper_bounds.G2
    # priority queue
    q = PartialQueue(G, copy(t))
    dequeue!(q)
    # Skeleton
    Ξ = Trace(t′, u0[2], u0[3], flow.old) # TODO use trace
    # Diagnostics
    acc = AcceptanceDiagnostics()
    ## create bounds ab 
    b = [ab(rng, upper_bounds, flow, target, t′, u, 1)][1:0]

    action = fill(Action(0), d)
   
    # fill priorityqueue
    for i in eachindex(v)          
        push!(b, ab(rng, upper_bounds, flow, target, t′, u, i))
        di = dir(geti(u, i))
        if x[i] != barriers[i].x[di]
            enqueue_time!(rng, q, u, i, b, action, barriers, flow)
        else
            v_old[i], v[i] = v[i], 0.0 # stop and save speed
            action[i] = unfreeze
            q[i] = t′ - log(rand(rng))/barriers[i].κ[di]
        end
    end
    if progress
        prg = Progress(progress_stops, 1)
    else
        prg = missing
    end
    
    println("Run main, run total")

    t′ = asynchzz_main(rng, prg, q, Ξ, t′, u, v_old, b, action, target, flow, upper_bounds, barriers, end_condition, acc)

    return Ξ, t′, u, acc
end

function asynchzz_main(rng, prg, q, Ξ, t′, u, v_old, b, action, target, flow, upper_bounds, barriers, end_condition, acc)
    (t, x, v) = u
    T = endtime(end_condition)
    stops = ismissing(prg) ? 0 : max(prg.n - 1, 0) # allow one stop for cleanup
    tstop = t′ + (T-t′)/stops
    while finished(end_condition, t′) 
        t′ = asynchzz_inner!(rng, q, Ξ, t′, u, v_old, b, action, target, flow, upper_bounds, barriers, acc)     
        if t′ > tstop
            tstop += T/stops
            next!(prg, showvalues = () -> [(:batches, round(acc.batchsize/acc.batches, digits=3)), (:empty, round(acc.empty/acc.batches, digits=3)), (:acc, round(acc.acc/acc.num,digits=3))]) 
        end  
    end
    ismissing(prg) || ProgressMeter.finish!(prg)
    return t′
end
function asynchzz_inner!(rng, q, Ξ, tmin, u, v_old, b, action, target, flow, upper_bounds, barriers, acc)

    t, x, v = u
    acc.batchsize += length(q.minima)
    acc.batches += 1
    if isempty(q.minima)
        #rand() < 0.001 && 
        error("empty")
        collectmin(q)
        acc.empty += 1
    end
    
    minima = dequeue!(q) 
    #println(length(minima), " ", tmin)
    tmin = Inf
    for (i, t′) in minima
        tmin = min(tmin, t′)
        @assert q.ripes[i] == localmin(q, i)
        if !q.ripes[i] 
            error()
        end
        t′ = q.vals[i]

        G = target.G
        if action[i] == hit # case 1) to be frozen or to reflect from boundary
            di = dir(geti(u, i))
            x[i] = barriers[i].x[di]
            t[i] = t′
            v_old[i], v[i] = v[i], 0.0 # stop and save speed
            action[i] = unfreeze
            q[i] = t′ - log(rand(rng))/barriers[i].κ[di]
        elseif action[i] == renew
            t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
            ∇ϕi = target(t′, u, i, flow) 
            l, lb = λ(i, u, ∇ϕi, b, flow)           
            if l > lb
                !upper_bounds.adapt && error("Tuning parameter `c` too small in check. l/lb = $(l/lb)")
                reset!(acc)
                adapt!(upper_bounds.c, i, upper_bounds.multiplier)
            end
            b[i] = ab(rng, upper_bounds, flow, ∇ϕi, t′, u, i) 
            queue_time!(rng, q, u, i, b, action, barriers, flow)
            # don't save
            continue
        elseif v[i] == 0 && v_old[i] != 0# case 2) was frozen
            t[i] = t′ # particle does not move, only time
            v[i], v_old[i] = v_old[i], 0.0 # unfreeze, restore speed
            di = dir(geti(u, i))
            if barriers[i].rule[di] == :reversible
                v[i] *= rand((-1,1))
            elseif barriers[i].rule[di] == :reflect
                v[i] = -v[i]
            end
            t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
            b[i] = ab(rng, upper_bounds, flow, target, t′, u, i)
            queue_time!(rng, q, u, i, b, action, barriers, flow)
        else    # was either a reflection 
                #time or an event time from the upper bound  
            t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
            ∇ϕi = target(t′, u, i, flow) 
            l, lb = λ(i, u, ∇ϕi, b, flow)                 
            if rand(rng)*lb < l # was a reflection time
                accept!(acc, lb, l)
                if l > lb
                    !upper_bounds.adapt && error("Tuning parameter `c` too small. l/lb = $(l/lb)")
                    reset!(acc)
                    adapt!(upper_bounds.c, i, upper_bounds.multiplier)
                end
                v = reflect!(i, ∇ϕi, x, v, flow.old) # reflect!
                b[i] = ab(rng, upper_bounds, flow, target, t′, u, i)
                queue_time!(rng, q, u, i, b, action, barriers, flow)
            else # was an event time from upperbound -> nothing happens
                not_accept!(acc, lb, l)
                b[i] = ab(rng, upper_bounds, flow, ∇ϕi, t′, u, i) 
                queue_time!(rng, q, u, i, b, action, barriers, flow)
                # don't save
                continue
            end    
        end    
        push!(Ξ, event(i, t′, x, v, flow.old))  
    end
    return tmin
end


export sspdmp4
function sspdmp4(∇ϕ2, t, x0, v0, T, c, ::Nothing, Z, κ, args...; progress=false, adapt = false, factor = 1.5)
    ∇ϕ(x, i) = ∇ϕ2(x, i, args...)
    Γ = Z.Γ
    d = length(x0)
    t0 = fill(t, d)
    u0 = (t0, x0, v0) 
    target = StructuredTarget(Γ, ∇ϕ)
    barriers = [StickyBarriers((0.0, 0.0), (:sticky, :sticky), (κ[i], κ[i])) for i in 1:d]
    flow = StickyFlow(Z)
    multiplier = factor
    G = G1 = target.G
    upper_bounds = StrongUpperBounds(G, adapt, c, multiplier)
    end_time = EndTime(T)
    trace, _, _, acc = @time asynchzz(u0, target, flow, upper_bounds, barriers, end_time; progress=progress)
    println("acc ", acc.acc/acc.num)
    return trace, acc
end
