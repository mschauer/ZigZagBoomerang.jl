# This could work for  ZigZag1d as well
"""
    move_forward!(τ, t, x, θ, Z::Union{BouncyParticle, ZigZag})

Updates the position `x`, velocity `θ` and time `t` of the
process after a time step equal to `τ` according to the deterministic
dynamics of the Bouncy particle sampler (`BouncyParticle`) and `ZigZag`:
(x(τ), θ(τ)) = (x(0) + θ(0)*t, θ(0)).
`x`: current location, `θ`: current velocity, `t`: current time,
"""
function move_forward!(τ, t, x, θ, Z::Union{BouncyParticle, ZigZag})
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
    for i in eachindex(x)
        x[i], θ[i] = (x[i] - B.μ[i])*cos(τ) + θ[i]*sin(τ) + B.μ[i],
                    -(x[i] - B.μ[i])*sin(τ) + θ[i]*cos(τ)
    end
    t + τ, x, θ
end

"""
        reflect!(i, x, θ, F)

Reflection rule of sampler `F` at reflection time.
`i`: coordinate which flips sign, `x`: position, `θ`: velocity (position
not used for the `ZigZag` and `FactBoomerang`.)
"""
function reflect!(i, x, θ, F::Union{ZigZag, FactBoomerang})
    θ[i] = -θ[i]
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
    x_new = sqrt(B.Σ)*((x - B.μ)/sqrt(B.Σ)*cos(τ) + θ*sin(τ)) + B.μ
    θ = -(x - B.μ)/sqrt(B.Σ)*sin(τ) + θ*cos(τ)
    t + τ, x_new, θ
end

"""
    reflect!(∇ϕx, θ, F::BouncyParticle, Boomerang)

Reflection rule of sampler `F` at reflection time.
x`: position, `θ`: velocity
"""
reflect!(∇ϕx, θ, x, ::Union{BouncyParticle, Boomerang}) = θ .-= 2*dot(∇ϕx, θ)/normsq(∇ϕx)*∇ϕx
