import ZigZagBoomerang: poisson_time # poisson_time(a, b, u), poisson_time((a, b, c), u)

struct Linear
end

normsq(q) = dot(q, q)

"""
Orthogonal subspace Crank-Nicolson step
"""
function oscn!(rng, v, ∇ψx, ρ)
    # Decompose v
    vₚ = (dot(v, ∇ψx)/normsq(∇ψx))*∇ψx
    v⊥ = (v - vₚ) * ρ
    if ρ == 1
        @. v = -vₚ + v⊥ 
    else
        # Sample and project
        z = randn!(rng, similar(v)) * √(1.0f0 - ρ^2)
        z -= (dot(z, ∇ψx)/dot(∇ψx, ∇ψx))*∇ψx
        @. v = (-vₚ + v⊥ + z) 
    end
    v
end

function Qbounce!(rng, u, ∇ψx)
    t, x, v = u
    oscn!(rng, v, ∇ψx, 0.99)
end
slope(x) = x[1][]
grad(x) = x[2][]
bound(x) = x[3][]


function subpdmp!(rng, Ξ, Ψ!, u, J, Δ, flow, Qbounce!, sgb)
    while true
        τ, action = next_event(rng, u, bound(sgb))
        τ > Δ && return false 
        move_forward!(flow, J, τ, u)
        if action == :expire
            Ψ!(rng, u, flow, sgb)
            continue
        elseif action == :bounce
            Ψ!(rng, u, flow, sgb)
            if rand(rng) <= max(0, slope(sgb))/λ̄(τ, bound(sgb))
                Qbounce!(rng, u, grad(sgb))
                Ψ!(rng, u, flow, sgb)
            else
                continue
            end
        elseif action == :refresh
            Qrefresh!(rng, u)
            Ψ!(rng, u, flow, sgb)    
        end
        push!(Ξ, deepcopy(u))
        return true
    end
end

function anisotropic(rng, Ψ!, u0, J, T, flow, Qbounce!)
    Ξ = [u0]
    u = deepcopy(u0)
    sgb = Ψ!(rng, u, flow)    
    changed = true
    while changed
         changed = subpdmp!(rng, Ξ, Ψ!, u, J, T, flow, Qbounce!, sgb)
         #@show u
    end
    return Ξ, u
end
function next_event(rng, u, ab)
    told, a, b, c, Δ = ab
    t, x, v = u
    @assert reduce(==, t)
    @assert told == t[1]

    # next event time
    τ = t[1] + poisson_time((a, b, c), rand(rng))
    τrefresh = t[1] + Inf

    # next event
    when, what = findmin((τ, Δ, τrefresh))
    return when, (:bounce, :expire, :refresh)[what]
end

function simplebound(rng, u, a)
    t, x, v = u
    @assert reduce(==, t)
    c = 0.1
    b = 0.0
    Δ = t[1] + 1.0
    (t[1], a, b, c, Δ)
end


λ̄(t, (told, a, b, c, Δ)) = max(a + b*(t - told), 0) + c
function move_forward!(flow::Linear, ::typeof(:), τ, u)
   t, x, v = u
   @. x = x + v*(τ - t)
   @. t = τ
   return u
end
using LinearAlgebra
using ForwardDiff
ψ(x) = normsq(x .- 1.0)/2

function Ψ!(rng, u, flow, (slope, grad, bound)=([0.0], [zero(u[3])], [simplebound(rng, u, 100.0)]))  
    t, x, v = u
    grad[] .= x .- 1.0 #ForwardDiff.gradient(ψ, x)
    slope[] = dot(v, grad[])
    bound[] = simplebound(rng, u, 100.0)
    slope, grad, bound
end
sep(Ξ) = map(x->x[1][1], Ξ), map(x->x[2], Ξ)


u0 = ([0.0, 0.0], [1.0f0, 0.5f0], [1.0f0, 1.0f0])
rng =  Random.default_rng()
anisotropic(rng, Ψ!, u0, :, 1000. , Linear(), Qbounce!)
Ξ, u = @time anisotropic(rng, Ψ!, u0, :, 1000., Linear(), Qbounce!)
ts, xs = sep(Ξ)
using GLMakie
lines(ts, getindex.(xs, 1))

lines(getindex.(xs, 1), getindex.(xs, 2), linewidth=0.1)

using ProfileView
ProfileView.@profview anisotropic(rng, Ψ!, u0, :, 1000. , Linear(), Qbounce!)