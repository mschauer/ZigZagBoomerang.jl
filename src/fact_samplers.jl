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
function λ(∇ϕi, i, x, θ, Z::ZigZag)
    pos(∇ϕi'*θ[i])
end


"""
    λ(∇ϕ, i, x, θ, Z::FactBoomerang)
`i`th Poisson rate of the `FactBoomerang` sampler
"""
function λ(∇ϕi, i, x, θ, B::FactBoomerang)
    pos((∇ϕi - (x[i] - B.μ[i])*B.Γ[i,i])*θ[i])
end

loosen(c, x) = c + x #+ log(c+1)*abs(x)/100
"""
    ab(G, i, x, θ, c, Flow)

Returns the constant term `a` and linear term `b` when computing the Poisson times
from the upper upper bounding rates λᵢ(t) = max(a + b*t)^2. The factors `a` and `b`
can be function of the current position `x`, velocity `θ`, tuning parameter `c` and
the Graph `G`
"""
function ab(G, i, x, θ, c, Z::ZigZag, args...)
    a = loosen(c[i], (idot(Z.Γ, i, x)  - idot(Z.Γ, i, Z.μ))'*θ[i])
    b = loosen(c[i]/100, θ[i]'*idot(Z.Γ, i, θ))
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
function spdmp_inner!(rng, G, G1, G2, ∇ϕ, t, x, θ, Q, c, b, t_old, (acc, num),
    F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false, adaptscale=false)
   n = length(x)
   while true
       i, t′ = peek(Q)
       refresh = i > n
       if refresh
           i = rand(1:n)
       end
       t, x, θ = smove_forward!(G, i, t, x, θ, t′, F)
       if refresh
           i = rand(1:n)
           t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
           if adaptscale && F isa ZigZag
               adapt_γ = 0.01; adapt_t0 = 15.; adapt_κ = 0.75
               pre = log(2.0) - sqrt(1.0 + t′)/(adapt_γ*(1.0 + t′ + adapt_t0))*log((1+acc[i])/(1.0 + 0.3*t′))
               η = (1 + t′)^(-adapt_κ)
               F.σ[i] = exp(η*pre + (1-η)*log(F.σ[i]))
               θ[i] = F.σ[i]*sign(θ[i])
           else
               if adaptscale
                   effi = (1 + 2*F.ρ/(1 - F.ρ))
                   τ = effi/(t[i]*F.λref)
                   if τ < 0.2
                       F.σ[i] = F.σ[i]*exp(((0.3t[i]/acc[i] > 1.66) - (0.3t[i]/acc[i] < 0.6))*0.03*min(1.0, sqrt(τ/F.λref)))
                   end
               end
               if F isa ZigZag && eltype(θ) <: Number
                   θ[i] = F.σ[i]*rand((-1,1))
               else
                   θ[i] = F.ρ*θ[i] + F.ρ̄*F.σ[i]*randn(eltype(θ))
               end
           end
 
           #renew refreshment
           Q[(n + 1)] = t′ + waiting_time_ref(F)
           #update reflections
           for j in neighbours(G1, i)
               b[j] = ab(G1, j, x, θ, c, F)
               t_old[j] = t[j]
               Q[j] = t[j] + poisson_time(b[j], rand(rng))
           end
       else
           # G neighbours have moved (and all others are not our responsibility)

           ∇ϕi = ∇ϕ_(∇ϕ, t, x, θ, i, t′, F, args...)
           l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i], t[i] - t_old[i])
           num += 1
           if rand(rng)*lb < l
               acc[i] += 1
               if l >= lb
                   !adapt && error("Tuning parameter `c` too small.")
                   #acc .= 0 
                   #num = 0
                   adapt!(c, i, factor)
               end
               t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
               θ = reflect!(i, ∇ϕi, x, θ, F)
               for j in neighbours(G1, i)
                   b[j] = ab(G1, j, x, θ, c, F)
                   t_old[j] = t[j]
                   Q[j] = t[j] + poisson_time(b[j], rand(rng))
               end
           else
               b[i] = ab(G1, i, x, θ, c, F)
               t_old[i] = t[i]
               Q[i] = t[i] + poisson_time(b[i], rand(rng))
               continue
           end
       end
       return event(i, t, x, θ, F), t, x, θ, t′, (acc, num), c, b, t_old
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
    n = length(x0)
    a = zeros(n)
    b = zeros(n)
    t_old = zeros(n)
    #sparsity graph
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    t, x, θ = t0, copy(x0), copy(θ0)
    num = acc = 0
    rng = Rng()
    Q = PriorityQueue{Tuple{Bool, Int64},Float64}()
    for i in eachindex(θ)
        a[i], b[i] = ab(G, i, x, θ, c, F)
        t_old[i] = t
        enqueue!(Q, (false, i) => poisson_time(a[i], b[i], rand(rng)))
        if hasrefresh(F)
            enqueue!(Q, (true, i)=>waiting_time_ref(F))
        end
    end
    Ξ = Trace(t0, x0, θ0, F)
    while t < T
        t, x, θ, (acc, num), c, a, b, t_old = pdmp_inner!(rng, Ξ, G, ∇ϕ, t, x, θ, Q, c, a, b, t_old, (acc, num), F, args...; factor=factor, adapt=adapt)
    end
    Ξ, (t, x, θ), (acc, num), c
end
