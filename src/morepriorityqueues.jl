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
