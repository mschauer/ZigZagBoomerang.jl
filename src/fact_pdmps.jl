using DataStructures
using Statistics
using SparseArrays
using LinearAlgebra

"""
    struct ZigZag(Γ, μ) <: ContinuousDynamics

Flag for local implementation of the ZigZag which exploits
any conditional independence structure of the target measure,
in form the argument Γ, a sparse precision matrix approximating
target precision. μ is the approximate target mean.
"""
struct ZigZag{T,S} <: ContinuousDynamics
    Γ::T
    μ::S
end

"""
    FactBoomerang(μ, λ) <: ContinuousDynamics

Dynamics preserving the N(μ, 1) measure (Boomerang)
with refreshment time `λ`
"""
struct FactBoomerang{R, T, S} <: ContinuousDynamics
    Γ::R
    μ::T
    λref::S
end
FactBoomerang(Γ, λ) = FactBoomerang(Γ, 0.0, λ)

#TODO
# FactBoomerang(d, λ) = FactBoomerang(fullyconnecetdgraph(d), 0.0, λ)

"""
    neighbours(G::Vector{<:Pair}, i) = G[i].second

Return extended neighbourhood of `i` including `i`.
`G`: graphs of neightbourhoods
"""
neighbours(G::Vector{<:Pair}, i) = G[i].second

"""
    move_forward!(τ, t, x, θ, Z::ZigZag)
Updates the position `x`, velocity `θ` and time `t` of the
process after a time step equal to `τ` according to the deterministic
dynamics of the `ZigZag` sampler: (x(τ), θ(τ)) = (x(0) + θ(0)*t, θ(0)).
`x`: current location, `θ`: current velocity, `t`: current time,
"""
move_forward!(τ, t, x, θ, Z::ZigZag) = move_forward!(τ, t, x, θ, Z::Bps)


#same as Boomerang
move_forward!(τ, t, x, θ, Z::FactBoomerang) = move_forward!(τ, t, x, θ, Z::Boomerang)

"""
        reflect!(i, θ, x, Z)
Reflection rule of `ZigZag` sampler at reflection time.
`i`: coordinate which flips sign, `θ`: velocity, `x`: position (not used for
the `ZigZag`)
"""
function reflect!(i, θ, x, Z::ZigZag)
    θ[i] = -θ[i]
    θ
end

# Same as Zig-Zag
reflect!(i, θ, x, ::FactBoomerang) = reflect!(i, θ, x, ::ZigZag)


normsq(x::Real) = abs2(x)
normsq(x) = dot(x,x)
"""
    λ(∇ϕ, i, x, θ, Z::ZigZag)
`i`th Poisson rate of the `ZigZag` sampler
"""
function λ(∇ϕ, i, x, θ, Z::ZigZag)
    pos(∇ϕ(x, i)*θ[i])
end

"""
    λ(∇ϕ, i, x, θ, Z::FactBoomerang)
`i`th Poisson rate of the `FactBoomerang` sampler
"""
function λ(∇ϕ, i, x, θ, Z::FactBoomerang)
    pos((∇ϕ(x, i) - (x[i] - B.μ[i]))*θ[i])
end


"""
    ab(G, i, x, θ, c, Z::ZigZag)

Returns the constant term `a` and linear term `b` when computing the Poisson times
from the upper upper bounding rates λᵢ(t) = max(a + b*t)^2. The factors `a` and `b`
can be function of the current position `x`, velocity `θ`, tuning parameter `c` and
the Graph `G`
"""
function ab(G, i, x, θ, c, Z::ZigZag)
    a = c[i] + θ[i]*(dot(Z.Γ[:, i], x)  - dot(Z.Γ[:, i], Z.μ))
    b = θ[i]*dot(Z.Γ[:, i], θ)
    a, b
end


"""
    λ_bar(G, i, x, θ, c, Z::ZigZag)

Computes the bounding rate `λ_bar` at position `x` and velocity `θ`.
"""
λ_bar(G, i, x, θ, c, Z::ZigZag) = pos(ab(G, i, x, θ, c, Z::ZigZag)[1])
event(i, t, x, θ, Z::ZigZag) = (i, t, x[i])
event(i, t, x, θ, Z::FactBoomerang) = (i, t, x[i], θ[i])

"""
    pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), Z::ZigZag;
        factor=1.5, adapt=false)
Inner loop of the `ZigZag` algorithm.
Input: a dependency graph `G`, gradient `∇ϕ`,
current position `x`, velocity `θ`, Queue of events `Q`, time `t`, and tuning parameter `c`.

The sampler 1) extracts from the queue the first event time. 2) moves deterministically
according to the `ZigZag` dynamics until event time. 3) Evaluates whether the event
time is a reflection time or not. 4) If it is a reflection time, the velocity reflects
according the the `ZigZag` reflection rule and updates `Q` according to the
dependency graph `G`. `(num, acc)` counts how many event times occour and how many of
those are real reflection times.
"""
function pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), Z::ZigZag;
        factor=1.5, adapt=false)

    i, t′ = dequeue_pair!(Q)
    if t′ - t < 0
        error("negative time")
    end
    t, x, θ = move_forward!(t′ - t, t, x, θ, Z)

    l, lb = λ(∇ϕ, i, x, θ, Z), λ_bar(G, i, x, θ, c, Z)
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
    t, x, θ, (num, acc)
end

"""
    pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), FB::FactBoomerang;
        factor=1.5, adapt=false)
Inner loop of the Factorized Boomerand algorithm.
Input: a dependency graph `G`, gradient `∇ϕ`,
current position `x`, velocity `θ`, Queue of events `Q`, time `t`, and tuning parameter `c`.

The sampler 1) extracts from the queue the first event time. 2) moves deterministically
according to the `ZigZag` dynamics until event time. 3) Evaluates whether the event
time is a reflection time or not. 4) If it is a reflection time, the velocity reflects
according the the `ZigZag` reflection rule and updates `Q` according to the
dependency graph `G`. `(num, acc)` counts how many event times occour and how many of
those are real reflection times.
"""
function pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc),
     FB::FactBoomerang; factor=1.5, adapt=false)

    (refresh, i), t′ = dequeue_pair!(Q)
    if t′ - t < 0
        error("negative time")
    end
    t, x, θ = move_forward!(t′ - t, t, x, θ, FB)
    if refresh
        θ[i] = randn()
        enqueue!(Q, (true, i)=> t + poisson_time(FB.λref, 0.0, rand()))
        for j in neighbours(G, i)
            j == i && continue
            Q[(false, j)] = t + poisson_time(ab(G, j, x, θ, c, FB)..., rand())
        end
        push!(Ξ, event(i, t, x, θ, FB))
    else
        l, lb = λ(∇ϕ, i, x, θ, FB), λ_bar(G, i, x, θ, c, FB)
        num += 1
        if rand()*lb < l
            acc += 1
            if l >= lb
                !adapt && error("Tuning parameter `c` too small.")
                c[i] *= factor
            end
            θ = reflect!(i, θ, x, FB)
            for j in neighbours(G, i)
                j == i && continue
                Q[(false, j)] = t + poisson_time(ab(G, j, x, θ, c, FB)..., rand())
            end
            push!(Ξ, event(i, t, x, θ, FB))
        end
    end
    enqueue!(Q, (false, i)=>t + poisson_time(ab(G, i, x, θ, c, FB)..., rand()))
    t, x, θ, (num, acc)
end

"""
    pdmp(∇ϕ, t0, x0, θ0, T, c, Z::ZigZag; factor=1.5, adapt=false) = Ξ, (t, x, θ), (acc, num)

`ZigZag` algorithm.
Input: Gradient of negative log density `∇ϕ`, initial time `t0`,
initial position `x0`, initial velocity `θ0`, final clock `T`, tuning parameter `c`.

The process moves at to time `T` with invariant mesure μ(dx) ∝ exp(-ϕ(x))dx and outputs
a collection of reflection points `Ξ` which, together with the initial triple `x`
`θ` and `t` are sufficient for reconstructuing continuously the continuous path
"""
function pdmp(∇ϕ, t0, x0, θ0, T, c, Z::ZigZag; factor=1.5, adapt=false)

    #sparsity graph
    G = [i => rowvals(Z.Γ)[nzrange(Z.Γ, i)] for i in eachindex(θ0)]

    t, x, θ = t0, copy(x0), copy(θ0)
    num = acc = 0

    Q = PriorityQueue{Int,Float64}()

    for i in eachindex(θ)
        enqueue!(Q, i=>poisson_time(ab(G, i, x, θ, c, Z)..., rand()))
    end

    Ξ = [event(1, t, x, θ, Z)][1:0]
    while t < T
        t, x, θ, (num, acc) = pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), Z; factor=1.5)
    end
    Ξ, (t, x, θ), (acc, num)
end


"""
    pdmp(∇ϕ, t0, x0, θ0, T, c, FB::FactBoomerang; factor=1.5,
    adapt=false) = Ξ, (t, x, θ), (acc, num)

`ZigZag` algorithm.
Input: Gradient of negative log density `∇ϕ`, initial time `t0`,
initial position `x0`, initial velocity `θ0`, final clock `T`, tuning parameter `c`.

The process moves at to time `T` with invariant mesure μ(dx) ∝ exp(-ϕ(x))dx and outputs
a collection of reflection points `Ξ` which, together with the initial triple `x`
`θ` and `t` are sufficient for reconstructuing continuously the continuous path
"""
function pdmp(∇ϕ, t0, x0, θ0, T, c, F::FactBoomerang; factor=1.5, adapt=false)
    #sparsity graph
    G = [i => rowvals(FB.Γ)[nzrange(FB.Γ, i)] for i in eachindex(θ0)]
    t, x, θ = t0, copy(x0), copy(θ0)
    num = acc = 0

    Q = PriorityQueue{Tuple{Bool, Int64},Float64}()
    for i in eachindex(θ)
        enqueue!(Q, (false, i)=>poisson_time(ab(G, i, x, θ, c, FB)..., rand()))
        enqueue!(Q, (true, i)=>poisson_time(FB.λref, 0.0, rand()))
    end

    Ξ = [event(1, t, x, θ, FB)][1:0]
    while t < T
        t, x, θ, (num, acc) = pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), FB; factor=1.5)
    end
    Ξ, (t, x, θ), (acc, num)
end
