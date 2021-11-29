#=
# Inventory

Boundaries, Reference #Q

State space
# u = (x, v, f)

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

struct StickyUpperBounds{TG,TG2,TΓ,Tstrong,Tc,Tfact}
    G1::TG
    G2::TG2
    Γ::TΓ
    strong::Tstrong
    adapt::Bool
    c::Tc
    factor::Tfact
end

struct StructuredTarget{TG,T∇ϕ}
    G::TG
    ∇ϕ::T∇ϕ
    #selfmoving ? 
end

(target::StructuredTarget)(t′, u, i, F) = target.∇ϕ(u[2], i)

StructuredTarget(Γ::SparseMatrixCSC, ∇ϕ) = StructuredTarget([i => rowvals(Γ)[nzrange(Γ, i)] for i in axes(Γ, 1)], ∇ϕ)

function ab(su::StickyUpperBounds, flow, i, u)
    (t, x, v) = u
    ab(su.G1, i, x, v, su.c, flow.old) #TODO
end

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

function freezing_time(barrier, ui, flow)
    t,x,v = ui
    di = dir(ui) # fix me: if above a reflecting barrier, reflect
    if  v*(x - barrier.x[di]) >= 0 # sic!
        return Inf
    else
        return -(x - barrier.x[di])/v
    end
end

function queue_time!(rng, Q, u, i, b, f, barriers::Vector, flow::StickyFlow)
    t, x, v = u
    trefl = poisson_time(b[i], rand(rng))
    tfreeze = freezing_time(barriers[i], geti(u, i), flow)
    if tfreeze <= trefl
        f[i] = true
        Q[i] = t[i] + tfreeze
    else
        f[i] = false
        Q[i] = t[i] + trefl
    end
    return Q
end
function stickyzz(u0, target::StructuredTarget, flow::StickyFlow, upper_bounds::StickyUpperBounds, barriers::Vector{<:StickyBarriers}, end_condition;  progress=false, progress_stops = 20, rng=Rng(Seed()))
    u = deepcopy(u0)
    # Initialize
    (t0, x0, v0) = u
    d = length(v0)
    t′ = maximum(t0)
    # priority queue
    Q = SPriorityQueue{Int,Float64}()
    # Skeleton
    Ξ = Trace(t′, u0[2], u0[3], flow.old) # TODO use trace
    # Diagnostics
    acc = AcceptanceDiagnostics(0, 0)
    ## create bounds ab
    b = [ab(upper_bounds, flow, i, u) for i in eachindex(v0)]
    f = zeros(Bool, d)
    # fill priorityqueue
    for i in eachindex(v0)
        trefl = poisson_time(b[i], rand(rng)) #TODO
        tfreez = freezing_time(barriers[i], geti(u, i), flow) #TODO
        if tfreez <= trefl
            f[i] = true
            enqueue!(Q, i => t0[i] + tfreez)
        else
            f[i] = false
            enqueue!(Q, i => t0[i] + trefl)
        end
    end
    if progress
        prg = Progress(progress_stops, 1)
    else
        prg = missing
    end

    println("Run main, run total")

    t′ = @time @inferred sticky_main(rng, prg, Q, Ξ, t′, u, b, f, target, flow, upper_bounds, barriers, end_condition, acc)

    return Ξ, t′, u, acc
end

function sticky_main(rng, prg, Q::SPriorityQueue, Ξ, t′, u, b, f, target, flow, upper_bounds, barriers, end_condition, acc)
    (t, x, v) = u
    T = endtime(end_condition)
    stops = ismissing(prg) ? 0 : max(prg.n - 1, 0) # allow one stop for cleanup
    tstop = t′ + (T-t′)/stops
    u_old = (copy(t), x, copy(v))
    while finished(end_condition, t′) 
        t, x, v = u
        i, t′ = stickyzz_inner!(rng, Q, Ξ, t′, u, u_old, b, f, target, flow, upper_bounds, barriers, acc)
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
function stickyzz_inner!(rng, Q, Ξ, t′, u, u_old, b, f, target, flow, upper_bounds, barriers, acc)
    while true
        t, x, v = u
        t_old, _, v_old = u_old
        i, t′ = peek(Q)
        G = target.G
        G1 =  upper_bounds.G1
        G2 =  upper_bounds.G2
        if f[i] # case 1) to be frozen or to reflect
            di = dir(geti(u, i))
            x[i] = barriers[i].x[di] # a bit dangerous
            t_old[i] = t[i] = t′
            v_old[i], v[i] = v[i], 0.0 # stop and save speed
            f[i] = false

            Q[i] = t′ - log(rand(rng))/barriers[i].κ[di]
            if upper_bounds.strong == false
                t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
                t, x, v = ssmove_forward!(G2, i, t, x, v, t′, flow.old)
                for j in neighbours(G1, i)
                    if v[j] != 0 # only non-frozen, especially not i
                        b[j] = ab(upper_bounds, flow, j, u)
                        t_old[j] = t[j]
                        Q = queue_time!(rng, Q, u, j, b, f, barriers, flow)
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
                    b[j] = ab(upper_bounds, flow, j, u)
                    t_old[j] = t[j]
                    Q = queue_time!(rng, Q, u, j, b, f, barriers, flow)
                end
            end
            return i, t′ 
        else    # was either a reflection 
                #time or an event time from the upper bound  
            t, x, v = ssmove_forward!(G, i, t, x, v, t′, flow.old) 
            ∇ϕi = target(t′, u, i, flow)           
            l, lb = sλ(∇ϕi, i, x, v, flow.old), sλ̄(b[i], t[i] - t_old[i])
            
            if rand(rng)*lb < l # was a reflection time
                accept!(acc, lb, l)
                if l > lb
                    !upper_bounds.adapt && error("Tuning parameter `c` too small. l/lb = $(l/lb)")
                    reset!(acc)
                    adapt!(upper_bounds.c, i, upper_bounds.factor)
                end
                v = reflect!(i, ∇ϕi, x, v, flow.old) # reflect!
                t, x, v = ssmove_forward!(G2, i, t, x, v, t′, flow.old)  # neighbours of neightbours \ neighbours
                for j in neighbours(G1, i)
                    if v[j] != 0
                        b[j] = ab(upper_bounds, flow, j, u)
                        t_old[j] = t[j]
                        queue_time!(rng, Q, u, j, b, f, barriers, flow)
                    end
                end
                return i, t′ 
            else # was an event time from upperbound -> nothing happens
                not_accept!(acc, lb, l)
                b[i] = ab(upper_bounds, flow, i, u)
                t_old[i] = t[i]
                queue_time!(rng, Q, u, i, b, f, barriers, flow)
                # don't save
                continue
            end               
        end
    end
end
