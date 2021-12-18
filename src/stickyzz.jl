#=
# Inventory

Boundaries, Reference #Q

State space
# u = (t, x, v)

Flow # 

Gradient of negative log-density ϕ    

Graph

Skeleton 
=#
using Test

struct StickyBarriers{Tx,Trule,Tκ}
    x::Tx # Intervals
    rule::Trule # set velocity to 0 and change label from free to frozen
    κ::Tκ 
end    

StickyBarriers() = StickyBarriers((-Inf, Inf), (:reflect, :reflect), (Inf, Inf))

struct StickyFlow{T}
    old::T
end

struct StickyUpperBounds{TG,TG2,TΓ,Tstrong,Tc,Tmult}
    G1::TG
    G2::TG2
    Γ::TΓ
    strong::Tstrong
    adapt::Bool
    c::Tc
    multiplier::Tmult
end
StickyUpperBounds(G, G1, Γ, c; adapt=false, strong=false, multiplier=1.5) = 
StickyUpperBounds(G1, 
                [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)],
                Γ,
                strong,
                adapt,
                c,
                multiplier)


struct LocalUpperBounds{TG,TG2,TΓ,Tstrong,Tc,Tmult}
    G1::TG
    G2::TG2
    Γ::TΓ
    strong::Tstrong
    adapt::Bool
    c::Tc
    multiplier::Tmult
end

struct StructuredTarget{TG,T∇ϕ}
    G::TG
    ∇ϕ::T∇ϕ
    #selfmoving ? 
end
(target::StructuredTarget)(t′, u, i, F) = target.∇ϕ(u[2], i)

StructuredTarget(Γ::SparseMatrixCSC, ∇ϕ) = StructuredTarget([i => rowvals(Γ)[nzrange(Γ, i)] for i in axes(Γ, 1)], ∇ϕ)

struct StructuredTarget2nd{TG,Tderivs}
    G::TG
    derivs::Tderivs
end
(target::StructuredTarget2nd)(t′, u, i, F) = target.derivs(u[2], i)




mutable struct AcceptanceDiagnostics
    acc::Int
    num::Int
end

function accept!(acc::AcceptanceDiagnostics, args...) 
    acc.acc += 1
    acc.num += 1
    acc
end
function not_accept!(acc::AcceptanceDiagnostics, args...) 
    acc.num += 1
    acc
end
function reset!(acc::AcceptanceDiagnostics)
    acc.num = acc.acc = 0
    acc
end
function stickystate(rng, x0)
    d = length(x0)
    v0 = rand(rng, (-1.0, 1.0), d)
    t0 = zeros(d)
    (t0, x0, v0) 
end
stickystate(x0) = stickystate(Random.GLOBAL_RNG, x0)

struct EndTime
    T::Float64
end
finished(end_time::EndTime, t) = t < end_time.T
endtime(end_time::EndTime) = end_time.T

dir(ui) = ui[3] > 0 ? 2 : (ui[3] < 0 ? 1 : 0)

geti(u, i) = (u[1][i], u[2][i], u[3][i]) 

function hitting_time(barrier, ui, flow)
    t,x,v = ui
    di = dir(ui) # fix me: if above a reflecting barrier, reflect
    if  v*(x - barrier.x[di]) >= 0 # sic!
        return Inf
    else
        return -(x - barrier.x[di])/v
    end
end

function λ(i, u::Tuple, ∇ϕi, b, ::StickyFlow) 
    t, x, θ = u
    ti, xi, θi = t[i], x[i], θ[i]
    abc = b[i]
    @assert ti <= abc[4]
    pos(∇ϕi'*θi), pos(abc[2] + abc[3]*(ti - abc[1]))
end


function ab(rng, su::StickyUpperBounds, flow, targetorgrad, t′, u, j)
    (t, x, v) = u
    a, b = ab(su.G1, j, x, v, su.c, flow.old) #TODO
    t[j], a, b, Inf
end

function queue_time!(rng, Q, u, i, b, action, barriers::Vector, flow::StickyFlow; enqueue=false)
    t, x, v = u
    @assert b[i][end] == Inf
    trefl = poisson_time(t[i], b[i], rand(rng)) - t[i]
    thit = hitting_time(barriers[i], geti(u, i), flow)
    if thit <= trefl
        action[i] = hit
        if enqueue
            enqueue!(Q, i => t[i] + thit)
        else
            Q[i] = t[i] + thit
        end     
    else
        action[i] = reflect
        if enqueue 
            enqueue!(Q, i => t[i] + trefl)
        else
            Q[i] = t[i] + trefl
        end
    end
    return Q
end

enqueue_time!(rng, Q, u, i, b, action, barriers::Vector, flow::StickyFlow) = 
    queue_time!(rng, Q, u, i, b, action, barriers::Vector, flow::StickyFlow, enqueue=true)
   
@enum Action begin
    hit
    reflect
    unfreeze
    renew  
end
function stickyzz(u0, target::StructuredTarget, flow::StickyFlow, upper_bounds::StickyUpperBounds, barriers::Vector{<:StickyBarriers}, end_condition;  progress=false, progress_stops = 20, rng=Rng(Seed()))
    u = deepcopy(u0)
    # Initialize
    (t0, x0, v0) = u
    d = length(v0)
    t′ = maximum(t0)
    v0_old = copy(v0)

    u_old = (copy(t0), x0, v0_old)

    # priority queue
    Q = SPriorityQueue{Int,Float64}()
    # Skeleton
    Ξ = Trace(t′, u0[2], u0[3], flow.old) # TODO use trace
    # Diagnostics
    acc = AcceptanceDiagnostics(0, 0)
    ## create bounds ab 
    b = [ab(rng, upper_bounds, flow, target, t′, u, 1)][1:0]

    action = fill(Action(0), d)
    
    # fill priorityqueue
    for i in eachindex(v0)          
        push!(b, ab(rng, upper_bounds, flow, target, t′, u, i))
        di = dir(geti(u, i))
        if x0[i] != barriers[i].x[di]
            enqueue_time!(rng, Q, u, i, b, action, barriers, flow)
        else
            v0_old[i], v0[i] = v0[i], 0.0 # stop and save speed
            action[i] = unfreeze
            enqueue!(Q, i => t′ - log(rand(rng))/barriers[i].κ[di])
        end
    end
    if progress
        prg = Progress(progress_stops, 1)
    else
        prg = missing
    end
    
    t′ = sticky_main(rng, prg, Q, Ξ, t′, u_old, u, b, action, target, flow, upper_bounds, barriers, end_condition, acc)

    return Ξ, t′, u, acc
end

function sticky_main(rng, prg, Q::SPriorityQueue, Ξ, t′, u_old, u, b, action, target, flow, upper_bounds, barriers, end_condition, acc)
    (t, x, v) = u
    T = endtime(end_condition)
    stops = ismissing(prg) ? 0 : max(prg.n - 1, 0) # allow one stop for cleanup
    tstop = t′ + (T-t′)/stops
    
    while finished(end_condition, t′) 
        t, x, v = u
        i, t′ = stickyzz_inner!(rng, Q, Ξ, t′, u, u_old, b, action, target, flow, upper_bounds, barriers, acc)
        t, x, v = u
        push!(Ξ, event(i, t, x, v, flow.old))
        if t′ > tstop
            tstop += T/stops
            next!(prg) 
        end  
    end
    ismissing(prg) || ProgressMeter.finish!(prg)
    return t′
end
function stickyzz_inner!(rng, Q, Ξ, t′, u, u_old, b, action, target, flow, upper_bounds, barriers, acc)
    while true
        t, x, v = u
        t_old, _, v_old = u_old
        i, t′ = peek(Q)
        G = target.G
        G1 = upper_bounds.G1
        G2 = upper_bounds.G2
        if action[i] == hit # case 1) to be frozen or to reflect from boundary
            di = dir(geti(u, i))
            x[i] = barriers[i].x[di]
            t_old[i] = t[i] = t′
            v_old[i], v[i] = v[i], 0.0 # stop and save speed
            action[i] = unfreeze
            Q[i] = t′ - log(rand(rng))/barriers[i].κ[di]
            if upper_bounds.strong == false
                t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
                t, x, v = ssmove_forward!(G2, i, t, x, v, t′, flow.old)
                for j in neighbours(G1, i)
                    if v[j] != 0 # only non-frozen, especially not i
                        b[j] = ab(rng, upper_bounds, flow, target, t′, u, j)
                        t_old[j] = t[j]
                        Q = queue_time!(rng, Q, u, j, b, action, barriers, flow)
                    end
                end
            end
            return i, t′
        elseif v[i] == 0 && v_old[i] != 0# case 2) was frozen
            t_old[i] = t[i] = t′ # particle does not move, only time
            v[i], v_old[i] = v_old[i], 0.0 # unfreeze, restore speed
            di = dir(geti(u, i))
            if barriers[i].rule[di] == :reversible
                v[i] *= rand((-1,1))
            elseif barriers[i].rule[di] == :reflect
                v[i] = -v[i]
            end
            t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
            t, x, v = ssmove_forward!(G2, i, t, x, v, t′, flow.old)
            for j in neighbours(G1, i)
                if v[j] != 0 # only non-frozen, including i # check!   
                    b[j] = ab(rng, upper_bounds, flow, target, t′, u, j)
                    t_old[j] = t[j]
                    Q = queue_time!(rng, Q, u, j, b, action, barriers, flow)
                end
            end
            return i, t′ 
        else    # was either a reflection 
                #time or an event time from the upper bound  
            t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
            ∇ϕi = target(t′, u, i, flow) 
            l, lb = λ(i, u, ∇ϕi, b, flow)           
            #l, lb = sλ(∇ϕi, i, x, v, flow.old), sλ̄(b[i], t[i] - t_old[i])
            
            if rand(rng)*lb < l # was a reflection time
                accept!(acc, lb, l)
                if l > lb
                    !upper_bounds.adapt && error("Tuning parameter `c` too small. l/lb = $(l/lb)")
                    reset!(acc)
                    adapt!(upper_bounds.c, i, upper_bounds.multiplier)
                end
                v = reflect!(i, ∇ϕi, x, v, flow.old) # reflect!
                t, x, v = ssmove_forward!(G2, i, t, x, v, t′, flow.old)  # neighbours of neightbours \ neighbours
                for j in neighbours(G1, i)
                    if v[j] != 0
                        b[j] = ab(rng, upper_bounds, flow, target, t′, u, j) 
                        t_old[j] = t[j]
                        queue_time!(rng, Q, u, j, b, action, barriers, flow)
                    end
                end
                return i, t′ 
            else # was an event time from upperbound -> nothing happens
                not_accept!(acc, lb, l)
                b[i] = ab(rng, upper_bounds, flow, ∇ϕi, t′, u, i) 
                t_old[i] = t[i]
                queue_time!(rng, Q, u, i, b, action, barriers, flow)
                # don't save
                continue
            end               
        end
    end
end


export sspdmp2
function sspdmp2(∇ϕ2, t, x0, v0, T, c, ::Nothing, Z, κ, args...; strong_upperbounds = false, progress=false, adapt = false, factor = 1.5)
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
    upper_bounds = StickyUpperBounds(G, G1, Γ, c; adapt=adapt, strong = strong_upperbounds, multiplier= multiplier)
    end_time = EndTime(T)
    trace, _, _, acc = @time stickyzz(u0, target, flow, upper_bounds, barriers, end_time; progress=progress)
    println("acc ", acc.acc/acc.num)
    return trace, acc
end