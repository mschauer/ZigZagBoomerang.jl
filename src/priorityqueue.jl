# This code was adapted from MIT licenced code from JuliaCollections/DataStructures.jl
# See https://github.com/JuliaCollections/DataStructures.jl/commits/master/src/priorityqueue.jl
# `index` was made a `Vector{Int}`

using DataStructures
using DataStructures: heapparent, heapleft, heapright, lt
import DataStructures.enqueue!
using Base: Ordering
struct SPriorityQueue{K,V,O<:Ordering} <: AbstractDict{K,V}
    # Binary heap of (element, priority) pairs.
    xs::Array{Pair{K,V},1}
    o::O

    # Map elements to their index in xs
    index::Vector{Int}

    function SPriorityQueue{K,V}() where {K,V}
        new{K,V,Base.Order.ForwardOrdering}(Vector{Pair{K,V}}(), Base.Order.ForwardOrdering(), Int[])
    end

    SPriorityQueue{K, V, O}(xs::Array{Pair{K,V}, 1}, o::O, index) where {K,V,O<:Ordering} = new(xs, o, index)

end
function SPriorityQueue(τ)
    Q = SPriorityQueue{Int,eltype(τ)}()
    for i in eachindex(τ)
        enqueue!(Q, i => τ[i])
    end
    Q
end
Base.length(pq::SPriorityQueue) = length(pq.xs)
Base.isempty(pq::SPriorityQueue) = isempty(pq.xs)

Base.peek(pq::SPriorityQueue) = pq.xs[1]

function percolate_down!(pq::SPriorityQueue, i::Integer)
    x = pq.xs[i]
    L = length(pq)
    @inbounds while (l = heapleft(i)) <= L
        r = heapright(i)
        j = r > L || lt(pq.o, pq.xs[l].second, pq.xs[r].second) ? l : r
        if lt(pq.o, pq.xs[j].second, x.second)
            pq.index[pq.xs[j].first] = i
            pq.xs[i] = pq.xs[j]
            i = j
        else
            break
        end
    end
    pq.index[x.first] = i
    pq.xs[i] = x
end

function percolate_up!(pq::SPriorityQueue, i::Integer)
    x = pq.xs[i]
    @inbounds while i > 1
        j = heapparent(i)
        if lt(pq.o, x.second, pq.xs[j].second)
            pq.index[pq.xs[j].first] = i
            pq.xs[i] = pq.xs[j]
            i = j
        else
            break
        end
    end
    pq.index[x.first] = i
    pq.xs[i] = x
end

# Equivalent to percolate_up! with an element having lower priority than any other
function force_up!(pq::SPriorityQueue, i::Integer)
    x = pq.xs[i]
    @inbounds while i > 1
        j = heapparent(i)
        pq.index[pq.xs[j].first] = i
        pq.xs[i] = pq.xs[j]
        i = j
    end
    pq.index[x.first] = i
    pq.xs[i] = x
end

Base.getindex(pq::SPriorityQueue, key) = pq.xs[pq.index[key]].second

# Change the priority of an existing element, or equeue it if it isn't present.
function Base.setindex!(pq::SPriorityQueue{K, V}, value, key) where {K,V}
    i = pq.index[key]
    oldvalue = pq.xs[i].second
    pq.xs[i] = Pair{K,V}(key, value)
    if lt(pq.o, oldvalue, value)
        percolate_down!(pq, i)
    else
        percolate_up!(pq, i)
    end
    return value
end

function enqueue!(pq::SPriorityQueue{K,V}, pair::Pair{K,V}) where {K,V}
    key = pair.first
    if length(pq) + 1 != key
        throw(ArgumentError("Elements must be enqueue! in order"))
    end
    push!(pq.xs, pair)
    push!(pq.index, length(pq))
    percolate_up!(pq, length(pq))

    return pq
end


# Unordered iteration through key value pairs in a SPriorityQueue
# O(n) iteration.
function _iterate(pq::SPriorityQueue, state)
    (k, idx), i = state
    return (pq.xs[idx], i)
end
_iterate(pq::SPriorityQueue, ::Nothing) = nothing

Base.iterate(pq::SPriorityQueue, ::Nothing) = nothing

function Base.iterate(pq::SPriorityQueue, ordered::Bool=true)
    _iterate(pq, iterate(pq.index))
end

Base.iterate(pq::SPriorityQueue, i) = _iterate(pq, iterate(pq.index, i))
