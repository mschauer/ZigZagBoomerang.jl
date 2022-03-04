# This could work for  ZigZag1d as well
"""
    move_forward!(τ, t, x, θ, Z::Union{BouncyParticle, ZigZag})

Updates the position `x`, velocity `θ` and time `t` of the
process after a time step equal to `τ` according to the deterministic
dynamics of the Bouncy particle sampler (`BouncyParticle`) and `ZigZag`:
(x(τ), θ(τ)) = (x(0) + θ(0)*t, θ(0)).
`x`: current location, `θ`: current velocity, `t`: current time,
"""
function move_forward!(τ, t, x, θ, Z::Union{BouncyParticle, ZigZag, GenBouncyParticle})
    t += τ
    x .+= θ .* τ
    t, x, θ
end


# This one could work for Boomerang 1d as well
"""
    move_forward!(τ, t, x, θ, B::Boomerang)
Updates the position `x`, velocity `θ` and time `t` of the
process after a time step equal to `τ` according to the deterministic
dynamics of the `Boomerang` sampler which are the Hamiltonian
dynamics preserving the Gaussian measure:
: x_t = μ +(x_0 − μ)*cos(t) + v_0*sin(t), v_t = −(x_0 − μ)*sin(t) + v_0*cos(t)
`x`: current location, `θ`: current velocity, `t`: current time.
"""
function move_forward!(τ, t, x, θ, B::Union{Boomerang, FactBoomerang})
    s, c = sincos(τ)
    for i in eachindex(x)
        x[i], θ[i] = (x[i] - B.μ[i])*c + θ[i]*s + B.μ[i],
                    -(x[i] - B.μ[i])*s + θ[i]*c
    end
    t + τ, x, θ
end

"""
        reflect!(i, x, θ, F)

Reflection rule of sampler `F` at reflection time.
`i`: coordinate which flips sign, `x`: position, `θ`: velocity (position
not used for the `ZigZag` and `FactBoomerang`.)
"""
function reflect!(i, ∇ϕx::Number, x, θ, F::Union{ZigZag, FactBoomerang})
    θ[i] = -θ[i]
    θ
end
function reflect!(i, ∇ϕx, x, θ, F::Union{ZigZag, FactBoomerang})
    θ[i] = θ[i] - (2*dot(∇ϕx, θ[i])/normsq(∇ϕx))*∇ϕx
    θ
end




"""
    move_forward(τ, t, x, θ, ::ZigZag1d)

Updates the position `x`, velocity `θ` and time `t` of the
process after a time step equal to `τ` according to the deterministic
dynamics of the `ZigZag1d` sampler: (x(τ), θ(τ)) = (x(0) + θ(0)*t, θ(0)).
`x`: current location, `θ`: current velocity, `t`: current time,
"""
function move_forward(τ, t, x, θ, ::ZigZag1d)
    τ + t, x + θ*τ , θ
end


"""
    move_forward(τ, t, x, θ, B::Boomerang1d)
Updates the position `x`, velocity `θ` and time `t` of the
process after a time step equal to `τ` according to the deterministic
dynamics of the `Boomerang1d` sampler: x_t = μ +(x_0 − μ)*cos(t) + v_0*sin(t),
v_t = −(x_0 − μ)*sin(t) + v_0*cos(t)
`x`: current location, `θ`: current velocity, `t`: current time.
"""
function move_forward(τ, t, x, θ, B::Boomerang1d)
    s, c = sincos(τ)
    t + τ, (x - B.μ)*c + θ*s + B.μ, -(x - B.μ)*s + θ*c
end

"""
    reflect!(∇ϕx, θ, F::BouncyParticle, Boomerang, GenBouncyParticle)

Reflection rule of sampler `F` at reflection time.
x`: position, `θ`: velocity
"""
function reflect!(∇ϕx, x, θ, F::BouncyParticle)
    θ .-= (2*dot(∇ϕx, θ)/normsq(F.L\∇ϕx))*(F.L'\(F.L\∇ϕx))
    θ
end
function reflect!(∇ϕx, x, θ, F::Boomerang)
    θ .-= (2*dot(∇ϕx, θ)/normsq(F.L\∇ϕx))*(F.L'\(F.L\∇ϕx))
#    θ .-= (2*dot(∇ϕx, θ)/normsq(∇ϕx))*∇ϕx 
    θ
end
function reflect!(∇ϕx, x, θ, F::GenBouncyParticle)
    θp = (θ'∇ϕx / normsq(∇ϕx)) .* ∇ϕx
    θ⊥ = F.ρ .* (θ - θp)
    z = randn!(similar(θ)) .* √(1.0f0 - F.ρ^2)
    z -= (z'∇ϕx / normsq(∇ϕx)) .* ∇ϕx
    θ .= -θp + θ⊥ + z
    θ
end

waiting_time_ref(rng, F) = poisson_time(rng, F.λref)
waiting_time_ref(F) = poisson_time(F.λref)


function refresh!(rng, θ, F)
    ρ̄ = sqrt(1-F.ρ^2)
    @inbounds for i in eachindex(θ)
        θ[i] = F.ρ*θ[i] + ρ̄*randn(rng)
    end
    θ
end

function refresh!(rng, θ, F::BouncyParticle)
    ρ̄ = sqrt(1-F.ρ^2)
    θ .*= F.ρ
    u = ρ̄*(F.L'\randn(rng, length(θ)))
    θ .+= u
    θ
end
function refresh!(rng, θ, F::Boomerang)
    ρ̄ = sqrt(1-F.ρ^2)
    θ .*= F.ρ
    u = ρ̄*(F.L'\randn(rng, length(θ)))
    θ .+= u
    θ
end