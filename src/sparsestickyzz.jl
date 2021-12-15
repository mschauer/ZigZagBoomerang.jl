
using Dictionaries
mutable struct SparseState
    d::Int
    u::Dictionary{Int64,Tuple{Float64, Float64, Float64}}
    t′::Float64
end
Base.length(u::SparseState) = u.d
nz(u::SparseState) = u.d - nnz(u)
function sparsestickystate(xs)
    SparseState(length(xs), dictionary(i=> (0.0, x, rand((-1.0,1.0))) for (i,x) in enumerate(xs) if x ≠ 0), 0.0)
end
function sparsestickystate(xs::SparseVector)
    SparseState(length(xs), dictionary(i=> (0.0, x, rand((-1.0,1.0))) for (i,x) in zip(SparseArrays.nonzeroinds(xs), nonzeros(xs)) if x ≠ 0), 0.0)
end


geti(u::SparseState, i::Int) = u[i]
function Base.getindex(u::SparseState, i::Int)
    has, ι = gettoken(u.u, i)
    if !has
        return (u.t′, 0.0, 0.0)
    end
    return gettokenvalue(u.u, ι)
end
function Base.setindex!(u::SparseState, ui, i::Int)
    u.u[i] = ui
    if u.t′ < ui[1]
        u.t′ = ui[1]
    end
    return ui
end
function Base.insert!(u::SparseState, i, ui)
    insert!(u.u, i, ui)
    #sortkeys!(u.u)
    if u.t′ < ui[1]
        u.t′ = ui[1]
    end
    return ui
end
function idot(A::SparseMatrixCSC, j, u::SparseState)
    rows = rowvals(A)
    vals = nonzeros(A)
    s = 0.0
    @inbounds for i in nzrange(A, j)
        s += vals[i]'*u[rows[i]][2]
    end
    s
end

function midot(A::SparseMatrixCSC, j, u::SparseState)
    rows = rowvals(A)
    vals = nonzeros(A)
    s = 0.0
    U = pairs(sortkeys(u.u))
   # @assert issorted(keys(u.u))
    U = pairs(u.u)
    ϕ = iterate(U)
    ϕ === nothing && return s
    (k, uk), state = ϕ
    @inbounds for i in nzrange(A, j)
        while rows[i] > k
            ϕ = iterate(U, state)
            ϕ === nothing && return s
            kold = k
            (k, uk), state = ϕ
            if !haskey(u.u, k)
                display(u.u)
                error("$k")
            end
            @assert u[k] == uk
            @assert kold < k     
        end
        if rows[i] == k
            s += vals[i]'*uk[2]
        end
    end
    s
end

Dictionaries.gettoken(u, i) = gettoken(u.u, i)
Dictionaries.gettokenvalue(u, ι) = gettokenvalue(u.u, ι)
Dictionaries.settokenvalue!(u, ι, v) = settokenvalue!(u.u, ι, v)

SparseArrays.nnz(u::SparseState) = length(u.u)
Base.keys(u::SparseState) = keys(u.u)

function stickystate(u::SparseState)
    @assert u.t′ == 0
    t = spzeros(u.d)
    x = spzeros(u.d)
    v = spzeros(u.d)

    for (i, u) in pairs(u.u)
        t[i] = u[1]
        x[i] = u[2]
        v[i] = u[3]
    end
    (t, x, v)
end

@enum Rules begin
    REFLECT
    REVERSIBLE
    STICKY
end

mutable struct SparseStickyUpperBounds{Tc,Tmult}
    adapt::Bool
    c::Tc
    multiplier::Tmult
end
SparseStickyUpperBounds(c; adapt=false, strong=false, multiplier=1.5) = 
    SparseStickyUpperBounds(adapt, c, multiplier) 



function hitting_time(barrier::StickyBarriers{<:Number}, ui, flow)
    t, x, v = ui
    if  v*(x - barrier.x) >= 0 # sic!
        return Inf
    else
        return t - (x - barrier.x)/v
    end
end


function λ(i, u, ∇ϕi, clocks, ::StickyFlow) 
    ti, xi, θi = u[i]
    abc = clocks[i][2]
    pos(∇ϕi'*θi), pos(abc[2] + abc[3]*(ti - abc[1]))
end

function ab(rng, su::SparseStickyUpperBounds, i, u, ∇ϕi, flow)
    ti, xi, θi = u[i]
    a = su.c + ∇ϕi'*θi
    b = 0.0
    s = su.adapt ? (1.0 - rand(rng)^2) : 1.0
    ti, a, b, ti + s/su.c 
end
function poisson_time(t, b::Tuple, r)
    @assert t <= b[end] # check bound validity 
    Δt = t - b[1]
    a = b[2] + Δt*b[3]
    b = b[3]
    t + poisson_time((a, b, 0.01), r) # guarantee minimum rate
end
function queue_time!(rng, Q, i, u, t′, b, clocks, barriers, flow::StickyFlow)
    t, x, v = ui = u[i]
    if t′ != t
        display(u.u)
        error("$i: $t ≠ $t′")
    end
    trefresh = b[end]
    trefl = poisson_time(t, b, rand(rng))
    thit = hitting_time(barriers, ui, flow)
    τ = min(trefresh, trefl, thit)
    Q[i] = τ
    if thit == τ
        action = hit
    elseif trefl == τ
        action = reflect
    else
        action = renew
    end
    clocks[i] = (action, b)
    return Q
end

function enqueue_time!(rng, Q, i, u, t′, b, clocks, barriers, flow::StickyFlow)
    t, x, v = ui = u[i]
    if t′ != t
        display(u.u)
        error("$i: $t ≠ $t′")
    end
    trefresh = b[end]
    trefl = poisson_time(t, b, rand(rng))
    thit = hitting_time(barriers, ui, flow)
    τ = min(trefresh, trefl, thit)
    enqueue!(Q, i => τ)
    if thit == τ
        action = hit
    elseif trefl == τ
        action = reflect
    else
        action = renew
    end
    set!(clocks, i, (action, b))
    return Q
end

@enum StickClock begin
    thaw = 0
end
function event(i, u::SparseState, Z)
    ti, xi, θi = u[i]
    ti, i, xi, θi
end



# FIXME
(target::StructuredTarget)(t′, u::SparseState, i, F) = target.∇ϕ(u, i)


function sparsestickyzz(u, target, flow::StickyFlow, upper_bounds, barriers::StickyBarriers, end_condition;  progress=false, progress_stops = 20, clusterα = 1.0, rng=Rng(Seed()))
    @assert barriers.rule == :reversible
    u0 = stickystate(u)
    # Initialize
    d = u.d
    t′ = u.t′

    #sortkeys!(u.u)
#    insert!(clocks, unfreeze) # add clock

    # priority queue
    Q = PriorityQueue{Int,Float64}()
    # Skeleton
    Ξ = Trace(t′, u0[2], u0[3], flow.old) # TODO use trace
    # Diagnostics
    acc = AcceptanceDiagnostics(0, 0)

    enqueue!(Q, Int(thaw) => t′ + randexp(rng)/(barriers.κ*(u.d-nnz(u)))) 
   
    # action and bound per clock
    clocks = similar(copy(keys(u)), Tuple{Action, Tuple{Float64, Float64, Float64, Float64}})
    
    # fill priorityqueue
    for i in keys(clocks)
        ∇ϕi = target(t′, u, i, flow)     
        b = ab(rng, upper_bounds, i, u, ∇ϕi, flow)
        enqueue_time!(rng, Q, i, u, t′, b, clocks, barriers, flow)
    end
    if progress
        prg = Progress(progress_stops, 1)
    else
        prg = missing
    end

    t′ = sparsesticky_main(rng, prg, clocks, Q, Ξ, t′, u, target, flow, upper_bounds, barriers, end_condition, acc, clusterα)

    return Ξ, t′, u, acc
end

function sparsesticky_main(rng, prg, clocks, Q, Ξ, t′, u, target, flow, upper_bounds, barriers, end_condition, acc, clusterα)
    T = endtime(end_condition)
    stops = ismissing(prg) ? 0 : max(prg.n - 1, 0) # allow one stop for cleanup
    tstop = t′ + (T-t′)/stops
    while finished(end_condition, t′) 
        i, t′ = sparsestickyzz_inner!(rng, clocks, Q, Ξ, t′, u,  target, flow, upper_bounds, barriers, acc, clusterα, T/3)
        @assert u.t′ == t′
        push!(Ξ, event(i, u, flow))
        if t′ > tstop
            tstop += T/stops
            next!(prg, showvalues = ()-> [(:active, round(nnz(u)/length(u), digits=3)), (:bound, upper_bounds.c), (:acc, round(acc.acc/acc.num,digits=3))]) 
        end  
    end
    ismissing(prg) || ProgressMeter.finish!(prg)
    return t′
end

function move_forward!(G, j, u, t′, ::StickyFlow) # check inferred
    for i in neighbours(G, j)
        has, ι = gettoken(u, i)
        has || continue
        ti, xi, θi = gettokenvalue(u, ι)
        ti, xi = t′, xi + θi*(t′ - ti)
        settokenvalue!(u, ι, (ti, xi, θi))
    end
    if u.t′ < t′
        u.t′ = t′
    end
    return
end
function reflect!(i, u, ∇ϕi, ::StickyFlow)
    ti, xi, θi = u[i]
    u[i] = ti, xi, -θi
    return
end



function sparsestickyzz_inner!(rng, clocks, Q, Ξ, t′, u, target, flow, upper_bounds, barriers, acc, clusterα, Tα)
    while true
        told = t′
        i, t′ = peek(Q)
        @assert t′ >= told
        if i != 0 && !haskey(u.u, i)
            display(u.u)
        end
        #@show i, t′
        G = target.G
        if i == Int(thaw)
            @assert nz(u) > 0
            while true    
                i = rand(1:length(u))
                !haskey(u.u, i) && break
            end
            if clusterα < 1 && t′ > Tα
                cost = 0
                for j in neighbours(G, i)
                    j == i && continue
                    cost += !haskey(u.u, j)
                end
                accepted = rand(rng) < clusterα^cost  
            else 
                accepted = true  
            end

            if accepted
                insert!(u, i, (t′, barriers.x, rand((-1.0, 1.0))))

                ∇ϕi = target(t′, u, i, flow)    
                b = ab(rng, upper_bounds, i, u, ∇ϕi, flow)
                enqueue_time!(rng, Q, i, u, t′, b, clocks, barriers, flow)
                Q[Int(thaw)] = t′ + randexp(rng)/(barriers.κ*(nz(u)))  
                return i, t′
            else continue # don't save
            end
        elseif clocks[i][1] == renew
            move_forward!(G, i, u, t′, flow) 
            ∇ϕi = target(t′, u, i, flow)
            l, lb = λ(i, u, ∇ϕi, clocks, flow)    # check boundary validity at expiry
            if l > lb
                !upper_bounds.adapt && error("Tuning parameter `c` too small. l/lb = $(l/lb)")
                reset!(acc)
                upper_bounds.c *= upper_bounds.multiplier
            end
            b = ab(rng, upper_bounds, i, u, ∇ϕi, flow)
            queue_time!(rng, Q, i, u, t′, b, clocks, barriers, flow)
            continue # don't save
        elseif clocks[i][1] == hit # case 1) to be frozen or to reflect from boundary
            move_forward!(G, i, u, t′, flow) 
            ti, xi, vi = geti(u, i)
            @assert ti == t′
            if !≈(xi, barriers.x, atol=1e-7)
                error("$i $xi")
            end
            if clusterα < 1 && t′ > Tα 
                cost = 0
                for j in neighbours(G, i)
                    j == i && continue
                    cost += haskey(u.u, j)
                end
                accepted = rand(rng) < clusterα^cost  
            else 
                accepted = true  
            end
            if accepted
                delete!(u.u, i)
                delete!(clocks, i)
                delete!(Q, i)
                Q[Int(thaw)] = t′ + randexp(rng)/(barriers.κ*(nz(u)))
                return i, t′
            else
                ∇ϕi = target(t′, u, i, flow)    
                b = ab(rng, upper_bounds, i, u, ∇ϕi, flow)
                queue_time!(rng, Q, i, u, t′, b, clocks, barriers, flow)
                Q[Int(thaw)] = t′ + randexp(rng)/(barriers.κ*(nz(u)))  
                return i, t′
            end
        else    # was either a reflection 
                #time or an event time from the upper bound  
            move_forward!(G, i, u, t′, flow) 
            ∇ϕi = target(t′, u, i, flow)        
            l, lb = λ(i, u, ∇ϕi, clocks, flow)
            
            if rand(rng)*lb < l # was a reflection time
                accept!(acc, lb, l)
                if l > lb
                    !upper_bounds.adapt && error("Tuning parameter `c` too small. l/lb = $(l/lb)")
                    reset!(acc)
                    upper_bounds.c *= upper_bounds.multiplier
                end
                reflect!(i, u, ∇ϕi, flow) 
                ∇ϕi = target(t′, u, i, flow)
                b = ab(rng, upper_bounds, i, u, ∇ϕi, flow)
                queue_time!(rng, Q, i, u, t′, b, clocks, barriers, flow)
                return i, t′ 
            else # was an event time from upperbound -> nothing happens
                not_accept!(acc, lb, l)
                b = ab(rng, upper_bounds, i, u, ∇ϕi, flow)
                queue_time!(rng, Q, i, u, t′, b, clocks, barriers, flow)
                continue # don't save
            end               
        end
    end
end



function sspdmp3(∇ϕ2, u0, T, c, ::Nothing, Z, κ, args...; adapt = false, factor = 1.5, progress=true, progress_stops = 20, clusterα=1.0)
    ∇ϕ(x, i) = ∇ϕ2(x, i, args...)
    Γ = Z.Γ
    d = length(u0)
    target = StructuredTarget(Γ, ∇ϕ)
    barrier = StickyBarriers(0.0, :reversible, κ)
    flow = StickyFlow(ZigZag(nothing, nothing, nothing))

    multiplier = factor
    upper_bounds = SparseStickyUpperBounds(c; adapt=adapt, multiplier= multiplier)
  
    end_time = EndTime(T)
    trace, _, uT, acc = @time sparsestickyzz(u0, target, flow, upper_bounds, barrier, end_time; progress=progress, progress_stops = progress_stops, clusterα=clusterα)
 
    println("acc ", acc.acc/acc.num)
    return trace, acc, uT

end
