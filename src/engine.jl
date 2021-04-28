using FunctionWranglers
using StructArrays
using StructArrays: components
using Random, LinearAlgebra
include("wranglers.jl")
include("switch.jl")


struct Schedule{T1,T2,T3,T4,T5}
    action!::T1
    next_action::T2
    state::T3
    T::T4
    args::T5
end
Base.eltype(::Type{Schedule{T1,T2,T3,T4,T5}}) where {T1,T2,T3,T4,T5} = Tuple{Float64, Int, eltype(T3), Int}
function Base.iterate(handler::Schedule)
    d = length(handler.state)
    action = ones(Int, d) # reset! events to trigger next_action event computations
    Q = SPriorityQueue(zeros(d))
    u = deepcopy(handler.state)
    iterate(handler, (u, action, Q))
end
Base.IteratorSize(::Type{<:Schedule}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:Schedule}) = Base.HasEltype()

function Base.iterate(handler, (u, action, Q))
    action! = handler.action!
    next_action = handler.next_action
    num, ev = handle!(u, action!, next_action, action, Q, handler.args...)
    ev[1] > handler.T && return nothing
    ev, (u, action, Q)
end

@inline function queue_next!(J, next_action, Q, action, i, t′, u, args...)
    for j in J # make typestable?
        τ, e = _sfindmin(next_action, Inf, 0, 1, j, i, t′, u, args...)
        Q[j] = τ
        action[j] = e
    end
end

function simulate(handler)
     u = deepcopy(handler.state)
     action! = handler.action!
     next_action = handler.next_action
     d = length(handler.state)
     action = ones(Int, d) # resets
     
     Q = SPriorityQueue(zeros(d))
     
     total, lastev = handle!(u, action!, next_action, action, Q, handler.args...)
     while true
        num, ev = handle!(u, action!, next_action, action, Q, handler.args...)
        total += num
        ev[1] > handler.T && break
        lastev = ev
     end
    return total, lastev
end

#=
function handle!(u, action!, next_action, action, Q, args...) 
    # Who is (i) next_action, when (t′) and what (j) happens?
    done = false
    local e, t′, i
    while !done
        i, t′ = peek(Q)
        e = action[i]

        if e == 1
            done, affected1 = action![1](i, t′, u, args...)
            queue_next!(affected1, next_action, Q, action, i, t′, u, args...) 
        elseif e == 2
            done, affected2 = action![2](i, t′, u, args...)
            queue_next!(affected2, next_action, Q, action, i, t′, u, args...) 
        elseif e == 3
            done, affected3 = action![3](i, t′, u, args...)
            queue_next!(affected3, next_action, Q, action, i, t′, u, args...) 
        elseif e == 4
            done, affected4 = action![4](i, t′, u, args...)
            queue_next!(affected4, next_action, Q, action, i, t′, u, args...) 
        elseif e == 5
            done, affected5 = action![5](i, t′, u, args...)
            queue_next!(affected5, next_action, Q, action, i, t′, u, args...) 
        elseif e == 6
            done, affected6 = action![6](i, t′, u, args...)
            queue_next!(affected6, next_action, Q, action, i, t′, u, args...) 
        end
    end

    (t′, i, u[i], action[i])
end
=#

function handle!(u, action!, next_action, action, Q, args::Vararg{Any, N}) where {N}
    # Who is (i) next_action, when (t′) and what (j) happens?
    done = false
    local e, t′, i
    num = 0
    while !done
        num += 1
        i, t′ = peek(Q)
        e = action[i]

        #done = action_nextaction(action!, next_action, Q, action, e, i, t′, u, args...)
        done = switch(e, action!, next_action, (Q, action), i, t′, u, args...)
    end
    num, (t′, i, u[i], action[i])
end



