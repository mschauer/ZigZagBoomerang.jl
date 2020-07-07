# implementation of factorised samplers

using DataStructures
using Statistics
using SparseArrays
using LinearAlgebra

"""
    neighbours(G::Vector{<:Pair}, i) = G[i].second

Return extended neighbourhood of `i` including `i`.
`G`: graphs of neightbourhoods
"""
neighbours(G::Vector{<:Pair}, i) = G[i].second
#need refreshments
hasrefresh(::FactBoomerang) = true
hasrefresh(::ZigZag) = false

#Ugly
isZigZag(::ZigZag) = true
isZigZag(::FactBoomerang) = false
normsq(x::Real) = abs2(x)
normsq(x) = dot(x,x)



"""
    λ(∇ϕ, i, x, θ, Z::ZigZag)
`i`th Poisson rate of the `ZigZag` sampler
"""
function λ(∇ϕ, i, x, θ, Z::ZigZag, args...)
    pos(∇ϕ(x, i, args...)*θ[i])
end


"""
    λ(∇ϕ, i, x, θ, Z::FactBoomerang)
`i`th Poisson rate of the `FactBoomerang` sampler
"""
function λ(∇ϕ, i, x, θ, B::FactBoomerang, args...)
    pos((∇ϕ(x, i, args...) - (x[i] - B.μ[i]))*θ[i])
end


"""
    ab(G, i, x, θ, c, Flow)

Returns the constant term `a` and linear term `b` when computing the Poisson times
from the upper upper bounding rates λᵢ(t) = max(a + b*t)^2. The factors `a` and `b`
can be function of the current position `x`, velocity `θ`, tuning parameter `c` and
the Graph `G`
"""
function ab(G, i, x, θ, c, Z::ZigZag)
    a = c[i] + θ[i]*(idot(Z.Γ, i, x)  - idot(Z.Γ, i, Z.μ))
    b = θ[i]*idot(Z.Γ, i, θ)
    a, b
end



function ab(G, i, x, θ, c, Z::FactBoomerang)
    nhd = neighbours(G, i)
    a = c[i]*sqrt(normsq(x[nhd] - Z.μ[nhd]) + normsq(θ[nhd]))
    b = 0.0
    a, b
end



"""
    λ_bar(G, i, x, θ, c, Z)

Computes the bounding rate `λ_bar` at position `x` and velocity `θ`.
"""
λ_bar(G, i, x, θ, c, Z::ZigZag) = pos(ab(G, i, x, θ, c, Z)[1])
λ_bar(G, i, x, θ, c, Z::FactBoomerang) = pos(ab(G, i, x, θ, c, Z)[1])


function event(i, t, x, θ, Z::Union{ZigZag,FactBoomerang})
    t, i, x[i], θ[i]
end



"""
    pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc),
        F::Union{ZigZag, FactBoomerang}; factor=1.5, adapt=false)

Inner loop of the factorised samplers: the Factorise Boomerand algorithm and the Zig-Zag sampler.
Input: a dependency graph `G`, gradient `∇ϕ`,
current position `x`, velocity `θ`, Queue of events `Q`, time `t`, and tuning parameter `c`.

The sampler 1) extracts from the queue the first event time. 2) moves deterministically
according to its dynamics until event time. 3) Evaluates whether the event
time is a reflection time or not. 4) If it is a reflection time, the velocity reflects
according its reflection rule and updates `Q` according to the
dependency graph `G`. `(num, acc)` counts how many event times occour and how many of
those are real reflection times.
"""
function pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc),
     F::Union{FactBoomerang}, args...; factor=1.5, adapt=false)

    (refresh, i), t′ = dequeue_pair!(Q)
    if t′ - t < 0
        error("negative time")
    end
    t, x, θ = move_forward!(t′ - t, t, x, θ, F)
    if refresh
        θ[i] = randn()
        #renew refreshment
        enqueue!(Q, (true, i)=> t + poisson_time(F.λref, 0.0, rand()))
        #update reflections
        Q[(false, i)] = t + poisson_time(ab(G, i, x, θ, c, F)..., rand())
        for j in neighbours(G, i)
            j == i && continue
            Q[(false, j)] = t + poisson_time(ab(G, j, x, θ, c, F)..., rand())
        end
        push!(Ξ, event(i, t, x, θ, F))
    else
        l, lb = λ(∇ϕ, i, x, θ, F, args...), λ_bar(G, i, x, θ, c, F)
        num += 1
        if rand()*lb < l
            acc += 1
            if l >= lb
                !adapt && error("Tuning parameter `c` too small.")
                c[i] *= factor
            end
            θ = reflect!(i, θ, x, F)
            for j in neighbours(G, i)
                j == i && continue
                Q[(false, j)] = t + poisson_time(ab(G, j, x, θ, c, F)..., rand())
            end
            push!(Ξ, event(i, t, x, θ, F))
        end
        enqueue!(Q, (false, i)=>t + poisson_time(ab(G, i, x, θ, c, F)..., rand()))
    end
    t, x, θ, (num, acc)
end

function pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), Z::ZigZag, args...;
        factor=1.5, adapt=false)

    i, t′ = dequeue_pair!(Q)
    if t′ - t < 0
        error("negative time")
    end
    t, x, θ = move_forward!(t′ - t, t, x, θ, Z)

    l, lb = λ(∇ϕ, i, x, θ, Z, args...), λ_bar(G, i, x, θ, c, Z)
    num += 1

    if rand()*lb < l
        acc += 1
        if l >= lb
            !adapt && error("Tuning parameter `c` too small.")
            c[i] *= factor
        end
        θ = reflect!(i, θ, x, Z)
        for j in neighbours(G, i)
            j == i && continue
            Q[j] = t + poisson_time(ab(G, j, x, θ, c, Z)..., rand())
        end
        push!(Ξ, event(i, t, x, θ, Z))
    end
    enqueue!(Q, i=>t + poisson_time(ab(G, i, x, θ, c, Z)..., rand()))
    t, x, θ, (num, acc), c
end

"""
    pdmp(∇ϕ, t0, x0, θ0, T, c, Z::LocalZigZag; factor=1.5, adapt=false) = Ξ, (t, x, θ), (acc, num)

`Local` algorithm.
Input: Gradient of negative log density `∇ϕ`, initial time `t0`,
initial position `x0`, initial velocity `θ0`, final clock `T`, tuning parameter `c`.

The process moves at to time `T` with invariant mesure μ(dx) ∝ exp(-ϕ(x))dx and outputs
a collection of reflection points `Ξ` which, together with the initial triple `x`
`θ` and `t` are sufficient for reconstructuing continuously the continuous path
"""
function pdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{FactBoomerang}, args...;
        factor=1.5, adapt=false)
    #sparsity graph
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    t, x, θ = t0, copy(x0), copy(θ0)
    num = acc = 0
    Q = PriorityQueue{Tuple{Bool, Int64},Float64}()
    for i in eachindex(θ)
        enqueue!(Q, (false, i)=>poisson_time(ab(G, i, x, θ, c, F)..., rand()))
        if hasrefresh(F)
            enqueue!(Q, (true, i)=>poisson_time(F.λref, 0.0, rand()))
        end
    end
    #TO CHANGE
    if isZigZag(F)
        Ξ = ZigZagTrace(t0, x0, θ0)
    else
        Ξ = FactBoomTrace(t0, x0, θ0)
    end
    while t < T
        t, x, θ, (num, acc) = pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), F, args...; factor=factor, adapt=adapt)
    end
    Ξ, (t, x, θ), (acc, num)
end

function pdmp(∇ϕ, t0, x0, θ0, T, c, Z::ZigZag, args...; factor=1.5, adapt=false)

    #sparsity graph
    G = [i => rowvals(Z.Γ)[nzrange(Z.Γ, i)] for i in eachindex(θ0)]

    t, x, θ = t0, copy(x0), copy(θ0)
    num = acc = 0

    Q = PriorityQueue{Int,Float64}()

    for i in eachindex(θ)
        enqueue!(Q, i=>poisson_time(ab(G, i, x, θ, c, Z)..., rand()))
    end

    Ξ = ZigZagTrace(t0, x0, θ0)
    while t < T
        t, x, θ, (num, acc), c = pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), Z, args...; factor=factor, adapt=adapt)
    end
    Ξ, (t, x, θ), (acc, num), c
end
