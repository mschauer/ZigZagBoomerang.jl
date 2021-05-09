using Random
using ConcreteStructs
@concrete struct SPDMP
    G
    G1
    G2
    ∇ϕ
    F
    rng
    adapt
    factor
end

function ZigZagBoomerang.smove_forward!(G, i, t, x, θ, m, t′, Z::Union{BouncyParticle, ZigZag})
    nhd = neighbours(G, i)
    for i in nhd
        if m != 1   #  not frozen
            t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
        else    # frozen 
            t[i] = t′
        end
    end
    t, x, θ
end
function ZigZagBoomerang.smove_forward!(G, i, t, x, θ, m, t′, B::Union{Boomerang, FactBoomerang})
    nhd = neighbours(G, i)
    for i in nhd
        τ = t′ - t[i]
        s, c = sincos(τ)
        if m[i] != 1 # not frozen
            t[i], x[i], θ[i] = t′, (x[i] - B.μ[i])*c + θ[i]*s + B.μ[i],
                    -(x[i] - B.μ[i])*s + θ[i]*c
        else    # frozen 
            t[i] = t′
        end
    end
    t, x, θ
end

function reset!(i, t′, u, P::SPDMP, args...)
    t, x, θ, θ_old, m = components(u)
    smove_forward!(P.G, i, t, x, θ, m, t′, P.F)
    smove_forward!(P.G2, i, t, x, θ, m, t′, P.F)

    false, P.G1[i].first
end

function never_reset(j, _, t′, u, P, args...)
    0, Inf
end


function rand_reflect!(i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    smove_forward!(G, i, t, x, θ, m, t′, F)
    ∇ϕi = P.∇ϕ(x, i, args...)
    l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i], t[i] - t_old[i])
    if rand(P.rng)*lb < l
        if l >= lb
            !P.adapt && error("Tuning parameter `c` too small.")
            adapt!(c, i, P.factor)
        end
        smove_forward!(G2, i, t, x, θ, m, t′, F)
        ZigZagBoomerang.reflect!(i, ∇ϕi, x, θ, F)
        return true, neighbours(G1, i)
    else
        return false, G1[i].first
    end
    
end

function freeze!(ξ, i, t′, u, P::SPDMP, args...)


    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
     
    @assert norm(x[i] + θ[i]*(t′ - t[i]) - ξ) < 1e-7
    smove_forward!(G, i, t, x, θ, m, t′, F)
    smove_forward!(G2, i, t, x, θ, m, t′, F)
    
    if m[i] == 0 # to freeze
        x[i] = ξ
        t[i] = t′
        m[i] = 1
        θ_old[i], θ[i] = θ[i], 0.0
    else # to unfreeze
        m[i] = 0
        t[i] = t′
        θ[i] = θ_old[i]
    end

    return true, neighbours(G1, i)

end

function discontinuity_at!(ξ, a, dir, i, t′, u, P::SPDMP, args...)


    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    
    @assert norm(x[i] + θ[i]*(t′ - t[i]) - ξ) < 1e-7
    smove_forward!(G, i, t, x, θ, m, t′, F)
  
    x[i] = ξ
    t[i] = t′
    if dir*θ[i] > 0 && rand(P.rng) < a
        θ[i] *= -1
        smove_forward!(G2, i, t, x, θ, m, t′, F)
        return true, neighbours(G1, i)
    else
        return false, G1[i].first
    end
end

function reflect!(ξ, dir, i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
     
    @assert dir*(x[i] + θ[i]*(t′ - t[i]) - ξ) < 1e-7 
    smove_forward!(G, i, t, x, θ, m, t′, F)
    smove_forward!(G2, i, t, x, θ, m, t′, F)
  
    x[i] = ξ
    θ[i] = dir*abs(θ[i])
    t[i] = t′
    return true, neighbours(G1, i)
end


function next_rand_reflect(j, i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    if m[j] == 1 
        return 0, Inf
    end
    b[j] = ab(G1, j, x, θ, c, F)
    t_old[j] = t′
    0, t[j] + poisson_time(b[j], rand(P.rng))
end

function next_reflect(ξ, dir,  j, i, t′, u, P::SPDMP, args...) 
    t, x, θ, θ_old, m = components(u)

    if dir*(x[j] - ξ) < 0
        return 0, t[j]
    end
    0, θ[j]*(x[j] - ξ) >= 0 ? Inf : t[j] - (x[j] - ξ)/θ[j]
end


function next_hit(ξ, j, i, t′, u, P::SPDMP, args...) 
    t, x, θ, θ_old, m = components(u)
    0, θ[j]*(x[j] - ξ) >= 0 ? Inf : t[j] - (x[j] - ξ)/θ[j]
end


function next_freezeunfreeze(ξ, κ, j, i, t′, u, P::SPDMP, args...) 
    t, x, θ, θ_old, m = components(u)
    if m[j] == 0
        0, θ[j]*(x[j] - ξ) >= 0 ? Inf : t[j] - (x[j] - ξ)/θ[j]
    else
        0, t[j] + poisson_time(κ, rand(P.rng))
    end
end
