using ZigZagBoomerang: SPriorityQueue, enqueue!
using FunctionWranglers
using StructArrays
using StructArrays: components
using Random, LinearAlgebra
include("wranglers.jl")
include("switch.jl")
function reset!(i, t′, u, args...)
    false, i
end

function next_reset(j, i, t′, u, args...)
    Inf
end


struct Handler{T1,T2,T3,T4,T5}
    action!::T1
    next_action::T2
    state::T3
    T::T4
    args::T5
end
Base.eltype(::Type{Handler{T1,T2,T3,T4,T5}}) where {T1,T2,T3,T4,T5} = Tuple{Float64, Int, eltype(T3), Int}
function Base.iterate(handler::Handler)
    d = length(handler.state)
    action = ones(Int, d) # reset! events to trigger next_action event computations
    Q = SPriorityQueue(zeros(d))
    u = deepcopy(handler.state)
    iterate(handler, (u, action, Q))
end
Base.IteratorSize(::Type{<:Handler}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:Handler}) = Base.HasEltype()

function Base.iterate(handler, (u, action, Q))
    action! = handler.action!
    next_action = handler.next_action
    ev = handle!(u, action!, next_action, action, Q, handler.args...)
    ev[1] > handler.T && return nothing
    ev, (u, action, Q)
end

function handle(handler)
     u = deepcopy(handler.state)
     action! = handler.action!
     next_action = handler.next_action
     d = length(handler.state)
     action = ones(Int, d) # resets
     Q = SPriorityQueue(zeros(d))
     ev = handle!(u, action!, next_action, action, Q, handler.args...)
     while ev[1] < handler.T
        ev = handle!(u, action!, next_action, action, Q, handler.args...)
     end
     return ev
end

function handle!(u, action!, next_action, action, Q, args...) 
    # Who is (i) next_action, when (t′) and what (j) happens?
    done = false
    local e, t′, i
    while !done
        i, t′ = peek(Q)
        e = action[i]
        #action_nextaction(action!, next_action, Q, action, e, i, t′, u, args...)
        done = switch(e, action!, next_action, (Q, action), i, t′, u, args...)
    end
    #= Equivalent to
    # Trigger state change
    affected = sindex(action!, e, i, t′, u, args...)::Union{Vector{Int},Int}
    # What happens next_action now has changed actionor each j, trigger updates
    for j in affected # make typestable?
        τ, e = sfindmin(next_action, j, i, t′, u, args...)
        Q[j] = τ
        action[j] = e
    end
     =#    

    (t′, i, u[i], action[i])
end
function lastiterate(itr) 
    ϕ  = iterate(itr)
    if ϕ === nothing
        error("empty")
    end
    x, state = ϕ
    while true
        ϕ = iterate(itr, state)
        if ϕ === nothing 
            return x
        end
        x, state = ϕ
    end
end


#r = range(-1, 1, length=500)
#image(Float64[(x + 1/(2sqrt(3)) > 0) && ((x + 1/(2sqrt(3)))*4/3 + abs(y)*2 < 1) for x in r, y in r]')
