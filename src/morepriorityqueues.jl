using DataStructures
using Graphs

function rcat(r1::AbstractUnitRange, r2::AbstractUnitRange)
    r1[end] + 1 == r2[1] || throw(ArgumentError("Arguments not contiguous"))
    return r1[1]:r2[end]
end
function rcat(r1, r2)
    (r1, r2)
end

struct LinearQueue{I,T}
    inds::I
    vals::T
end
function Base.peek(q::LinearQueue)
    t, i = findmin(q.vals)
    return q.inds[i], t
end
Base.keys(q::LinearQueue) = q.inds
function Base.getindex(q::LinearQueue{<:AbstractUnitRange}, key)
    q.vals[1 + key - first(q.inds)]
end
function Base.setindex!(q::LinearQueue{<:AbstractUnitRange}, value, key)
    q.vals[1 + key - first(q.inds)] = value
end
    
struct PriorityQueues{S,T}
    head::S
    tail::T
end
Base.keys(q::PriorityQueues) = rcat(keys(q.head), keys(q.tail))
function Base.peek(q::PriorityQueues)
    i1, t1 = peek(q.head)
    isempty(q.tail) && return i1, t1
    i, t = peek(q.tail)
    if t1 < t
        return i1, t1
    else
        return i, t
    end
end

function Base.getindex(q::PriorityQueues, key)
    key in keys(q.head) && return q.head[key]
    return q.tail[key]
end
function Base.setindex!(q::PriorityQueues, value, key)
    key in keys(q.head) && return q.head[key] = value
    return q.tail[key] = value
end


struct PartialQueue{U,T,S,R}
    G::U
    vals::T
    ripes::S
    minima::R
end
function PartialQueue(G, vals)
    ripes = falses(length(vals))
    minima = Pair{Int64, Float64}[]
    for i in eachindex(vals)
        ripes[i] = localmin(G, vals, i)
        ripes[i] && push!(minima, i=>vals[i])
    end            
    PartialQueue(G, vals, ripes, minima)
end
function build!(q::PartialQueue)
    reset!(q.minima[])
    for i in eachindex(q.vals)
        q.ripes[i] = localmin(q, i)
    end
    return
end
function check(q::PartialQueue)
    vals = q.vals
    ripes = falses(length(vals))
    for i in eachindex(vals)
        ripes[i] = localmin(q, i)
    end
    ripes == q.ripes || error("Internal error")
end
function collectmin(q::PartialQueue)
    isempty(q.minima) || error("Full queue")
    m = findall(q.ripes)
    for (i,t) in zip(m, q.vals[m])
        push!(q.minima, i=>t)
    end
    if isempty(q.minima) 
        #check(q)
        error("No minimum in queue")
    end
    q.minima
end

function localmin(q, i)
    localmin(q.G, q.vals, i)
end
function localmin(G, vals, i)
    val = vals[i]
    for j in neighbours(G, i)
        i == j && continue
        val > vals[j] && return false
    end
    return true
end

Base.peek(q::PartialQueue) = q.minima
function DataStructures.dequeue!(q::PartialQueue) 
    m = copy(q.minima)
    resize!(q.minima, 0)
    m
end

Base.getindex(q::PartialQueue, key) = q.vals[key]
function Base.setindex!(q::PartialQueue, value, key) 
    q.vals[key] < value || throw(ArgumentError("Can't decrease key $(q.vals[key]) to $value"))
    q.vals[key] = value
    (q.ripes[key] = localmin(q, key)) && push!(q.minima, key => value)
    for i in neighbours(q.G, key)
        i == key && continue
        ripe = localmin(q, i)
        (!q.ripes[i] && ripe) && push!(q.minima, i => q.vals[i])
        q.ripes[i] = ripe
        (q.ripes[i] && !ripe) && i != key && error("lost minimum")
    end
#    check(q)
    value
end
