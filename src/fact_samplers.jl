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
hasrefresh(Z::ZigZag) = Z.λref > 0




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
    pos((∇ϕ(x, i, args...) - (x[i] - B.μ[i])*B.Γ[i,i])*θ[i])
end

loosen(c, x) = c + x #+ log(c+1)*abs(x)/100
"""
    ab(G, i, x, θ, c, Flow)

Returns the constant term `a` and linear term `b` when computing the Poisson times
from the upper upper bounding rates λᵢ(t) = max(a + b*t)^2. The factors `a` and `b`
can be function of the current position `x`, velocity `θ`, tuning parameter `c` and
the Graph `G`
"""
function ab(G, i, x, θ, c, Z::ZigZag)
    a = loosen(c[i], θ[i]*(idot(Z.Γ, i, x)  - idot(Z.Γ, i, Z.μ)))
    b = loosen(c[i]/100, θ[i]*idot(Z.Γ, i, θ))
    a, b
end



function ab(G, i, x, θ, c, Z::FactBoomerang)
    nhd = neighbours(G, i)
    z = sqrt(sum((x[j] - Z.μ[j])^2 + θ[j]^2 for j in nhd))
    z2 = (x[i]^2 + θ[i]^2)
    a = c[i]*sqrt(z2)*z + z2*Z.Γ[i,i]
    b = 0.0
    a, b
end

function adapt!(c, i, factor)
    c[i] *= factor
    c
end

"""
    λ_bar(G, i, x, θ, c, Z)

Computes the bounding rate `λ_bar` at position `x` and velocity `θ`.
"""
#λ_bar(G, i, x, θ, c, Z::ZigZag) = pos(ab(G, i, x, θ, c, Z)[1])
#λ_bar(G, i, x, θ, c, Z::FactBoomerang) = pos(ab(G, i, x, θ, c, Z)[1])


function event(i, t, x, θ, Z::Union{ZigZag,FactBoomerang})
    t, i, x[i], θ[i]
end



"""
    pdmp_inner!(Ξ, G, ∇ϕ, t, x, θ, Q, c, a, b, t_old, (acc, num),
        F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)
        = t, x, θ, (acc, num), c, a, b, t_old

Inner loop of the factorised samplers: the factorised Boomerang algorithm and
the Zig-Zag sampler. Given a dependency graph `G`, gradient `∇ϕ`,
current position `x`, velocity `θ`, Queue of events `Q`, time `t`, tuning parameter `c`,
terms of the affine bounds `a`,`b` and time when the upper bounds were computed `t_old`

The sampler 1) extracts from the queue the first event time. 2) moves deterministically
according to its dynamics until event time. 3) Evaluates whether the event
time is a accepted reflection or refreshment time or shadow time. 4) If it is a
reflection time, the velocity reflects according its reflection rule, if it is a
refreshment time, the sampler updates the velocity from its prior distribution (Gaussian).
In both cases, updates `Q` according to the dependency graph `G`. The sampler proceeds
until the next accepted reflection time or refreshment time. `(num, acc)`
incrementally counts how many event times occour and how many of those are real reflection times.
"""
function pdmp_inner!(Ξ, G, ∇ϕ, t, x, θ, Q, c, a, b, t_old, (acc, num),
     F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)

    while true
        (refresh, i), t′ = dequeue_pair!(Q)
        if t′ - t < 0
            error("negative time")
        end
        t, x, θ = move_forward!(t′ - t, t, x, θ, F)
        if refresh
            θ[i] = F.ρ*θ[i] + sqrt(1-F.ρ^2)*F.σ[i]*randn()
            #renew refreshment
            enqueue!(Q, (true, i) => t + waiting_time_ref(F))
            #update reflections
            for j in neighbours(G, i)
                a[j], b[j] = ab(G, j, x, θ, c, F)
                t_old[j] = t
                Q[(false, j)] = t + poisson_time(a[j], b[j], rand())
            end
        else
            l, lb = λ(∇ϕ, i, x, θ, F, args...), pos(a[i] + b[i]*(t - t_old[i]))
            num += 1
            if rand()*lb < l
                acc += 1
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c[i] *= factor
                end
                θ = reflect!(i, x, θ, F)
                for j in neighbours(G, i)
                    a[j], b[j] = ab(G, j, x, θ, c, F)
                    t_old[j] = t
                    Q[(false, j)] = t + poisson_time(a[j], b[j], rand())
                end
            else
                # Move a, b, t_old inside the queue as auxiliary variables
                a[i], b[i] = ab(G, i, x, θ, c, F)
                t_old[i] = t
                enqueue!(Q, (false, i) => t + poisson_time(a[i], b[i], rand()))
                continue
            end
        end
        push!(Ξ, event(i, t, x, θ, F))
        return t, x, θ, (acc, num), c, a, b, t_old
    end
end



"""
    pdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag, FactBoomerang}, args...; factor=1.5, adapt=false) = Ξ, (t, x, θ), (acc, num), c

Outer loop of the factorised samplers, the Factorised Boomerang algorithm
and the Zig-Zag sampler. Inputs are a function `∇ϕ` giving `i`th element of gradient
of negative log target density `∇ϕ(x, i, args...)`, starting time and position `t0, x0`,
velocities `θ0`, and tuning vector `c` for rejection bounds and final clock `T`.

The process moves to time `T` with invariant mesure μ(dx) ∝ exp(-ϕ(x))dx and outputs
a collection of reflection points which, together with the initial triple `t`, `x`
`θ` are sufficient for reconstructuing continuously the continuous path.
It returns a `FactTrace` (see [`Trace`](@ref)) object `Ξ`, which can be collected
into pairs `t => x` of times and locations and discretized with `discretize`.
Also returns the `num`ber of total and `acc`epted Poisson events and updated bounds
`c` (in case of `adapt==true` the bounds are multiplied by `factor` if they turn
out to be too small.)

This version does not assume that `∇ϕ` has sparse conditional dependencies,
see [`spdmp`](@ref).
"""
function pdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, args...;
        factor=1.5, adapt=false)
    a = zero(x0)
    b = zero(x0)
    t_old = zero(x0)
    #sparsity graph
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    t, x, θ = t0, copy(x0), copy(θ0)
    num = acc = 0
    Q = PriorityQueue{Tuple{Bool, Int64},Float64}()
    for i in eachindex(θ)
        a[i], b[i] = ab(G, i, x, θ, c, F)
        t_old[i] = t
        enqueue!(Q, (false, i) => poisson_time(a[i], b[i], rand()))
        if hasrefresh(F)
            enqueue!(Q, (true, i)=>waiting_time_ref(F))
        end
    end
    Ξ = Trace(t0, x0, θ0, F)
    while t < T
        t, x, θ, (acc, num), c, a, b, t_old = pdmp_inner!(Ξ, G, ∇ϕ, t, x, θ, Q, c, a, b, t_old, (acc, num), F, args...; factor=factor, adapt=adapt)
    end
    Ξ, (t, x, θ), (acc, num), c
end
