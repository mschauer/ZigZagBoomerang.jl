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

struct StickyFlow{T}
    old::T
end

struct StickyUpperBounds{TG,TΓ,Tstrong,Tc,Tfact}
    G::TG
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

StructuredTarget(Γ::SparseMatrixCSC, ∇ϕ) = StructuredTarget([i => rowvals(Γ)[nzrange(Γ, i)] for i in axes(Γ, 1)], ∇ϕ)

function ab(su::StickyUpperBounds, flow, i, u)
    (t, x, v) = u
    ab(su.G, i, x, v, su.c, flow.old) #TODO
end

struct AcceptanceDiagnostics
    acc::Int
    num::Int
end

function stickystate(x0)
    d = length(x0)
    v0 = rand((-1.0, 1.0), d)
    t0 = zeros(d)
    u0 = (t0, x0, v0) 
end

struct EndTime
    T::Float64
end
finished(end_time::EndTime, t) = t < end_time.T

dir(ui) = dir = ui[3] > 0 ? 1 : 2
geti(u, i) = (u[1][i], u[2][i], u[3][i]) 

function freezing_time(barrier, ui)
    t,x,v = ui
    di = dir(ui)
    if  v*(x-barrier.x[di]) >= 0 # sic!
        return Inf
    else
        return -(x - barrier.x[di])/v
    end
end

function stickyzz(u0, target::StructuredTarget, flow::StickyFlow, upper_bounds::StickyUpperBounds, barriers::Vector{<:StickyBarriers}, end_condition)
    # Initialize
    (t0, x0, v0) = u0
    d = length(v0)
    t′ = maximum(t0)
    # priority queue
    Q = SPriorityQueue{Int,Float64}()
    # Skeleton
    Ξ = [] # TODO use trace
    # Diagnostics
    acc = AcceptanceDiagnostics(0, 0)
    ## create bounds ab
    b = [ab(upper_bounds, flow, i, u0) for i in eachindex(v0)]
    f = zeros(Bool, d)
    # fill priorityqueue
    for i in eachindex(v0)
        trefl = poisson_time(b[i], rand()) #TODO
        tfreez = freezing_time(barriers[i], geti(u0, i)) #TODO
        if trefl > tfreez
            f[i] = true
            enqueue!(Q, i => t0[i] + tfreez)
        else
            f[i] = false
            enqueue!(Q, i => t0[i] + trefl)
        end
    end
    rng = Random.GLOBAL_RNG
    println("Run main, run total")

    Ξ = @time @inferred sticky_main(rng, Q, Ξ, t′, u0, b, f, target, flow, upper_bounds, barriers, end_condition, acc)

    return Ξ
end

function sticky_main(rng, Q::SPriorityQueue, Ξ, t′, u, b, f, target, flow, upper_bounds, barriers, end_condition, acc)
    (t, x, v) = u
    u_old = (copy(t), x, copy(v))
    while finished(end_condition, t′) 
        (t, x, v) = u
        d = length(x)
       
        i, t′ = stickyzz_inner!(rng, Q, Ξ, t′, u, u_old, b, f, target, flow, upper_bounds, barriers, acc)
        t, x, θ = u
        push!(Ξ, event(i, t, x, θ, flow.old))
    end
    return Ξ
end
function stickyzz_inner!(rng, Q, Ξ, t′, u, u_old, b, f, target, flow, upper_bounds, barriers, acc)
    while true
        t, x, v = u
        t_old, _, v_old = u_old
        i, t′ = peek(Q)
        
        
        if f[i] # case 1) to be frozen
            t_old[i] = t[i]
            t[i] = t′
            f[i] = false
            di = dir(geti(u, i))
            Q[i] = t′ - log(rand(rng))/barriers[i].κ[di]
            return i, t′ 
        elseif x[i] == 0 && θ[i] == 0 # case 2) was frozen
            queue_time!(Q, u..., i, b, f, flow.old)
            return i, t′ 
        else    # was either a reflection 
            t_old[i] = t[i]
            t[i] = t′ 
            queue_time!(Q, u..., i, b, f, flow.old)
                #time or an event time from the upper bound  
            if rand() < 0.5 # was a reflection time 
                return i, t′ 
            else # was an event time from upperbound -> nothing happens
                continue
            end               
        end

    end
end


