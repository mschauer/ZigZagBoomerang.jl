using DataStructures
using Graphs
const CHECKPQ = false
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
    nregions::Int
end
nv_(G::Vector) = length(G)
nv_(G::AbstractGraph) = nv(G)

div1(a,b) = (a-1)÷b + 1
rkey(q, key) = div1(q.nregions*key, nv_(q.G))
rkey((nregions, nv)::Tuple, key) = div1(nregions*key, nv)

function PartialQueue(G, vals, nregions=1)
    ripes = falses(length(vals))
    minima = [Pair{Int64, Float64}[] for _ in 1:nregions]
    for i in 1:length(vals)
        ripes[i] = localmin(G, vals, i)
        ripes[i] && push!(minima[div1(nregions*i, nv_(G))], i=>vals[i])
    end
    PartialQueue(G, vals, ripes, minima, nregions)
end
#=function build!(q::PartialQueue)
    resize!(q.minima)
    for i in eachindex(q.vals)
        q.ripes[i] = localmin(q, i)
    end
    return
end=#
function check(q::PartialQueue)
    for i in eachindex(q.vals)
        q.ripes[i] == localmin(q, i) || error("Internal error")
    end
end

function checkqueue(q::PartialQueue)
    minima = [Pair{Int64, Float64}[] for _ in 1:q.nregions]
    for i in 1:length(q.vals)
        q.ripes[i] == localmin(q, i) || error("Internal error")   
        q.ripes[i] && push!(minima[div1(q.nregions*i, nv_(q.G))], i=>q.vals[i])
    end
    CHECKPQ && for i in 1:q.nregions
        if Set(first.(q.minima[i])) != Set(first.(minima[i]))
            println(setdiff(minima[i], q.minima[i]))
            println(setdiff(q.minima[i], minima[i]))
            
            error("corrupted")
        end
    end
end




function collectmin(q::PartialQueue)
    all(isempty.(q.minima)) || error("Full queue")
    for i in findall(q.ripes)
        push!(q.minima[div1(q.nregions*i, nv_(q.G))], i=>q.vals[i])
    end
    if all(isempty.(q.minima))
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
    for _ in 1:q.nregions
        push!(q.minima, Pair{Int64, Float64}[])
    end
    m
end

Base.getindex(q::PartialQueue, key) = q.vals[key]
function Base.setindex!(q::PartialQueue, value, key) 
    q.vals[key] < value || throw(ArgumentError("Can't decrease key $(q.vals[key]) to $value"))
    q.vals[key] = value
    rkey = div1(q.nregions*key, nv_(q.G))
    (q.ripes[key] = localmin(q, key)) && push!(q.minima[rkey], key => value)
    for i in neighbours(q.G, key)
        i == key && continue
        ripe = localmin(q, i)
        ri = div1(q.nregions*i, nv_(q.G))
        (!q.ripes[i] && ripe) && push!(q.minima[ri], i => q.vals[i])
        q.ripes[i] = ripe
        (q.ripes[i] && !ripe) && i != key && error("lost minimum")
    end
    #check(q)
    value
end
