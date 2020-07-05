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
    x_new = (x - B.μ)*cos(τ) + θ*sin(τ) + B.μ
    θ = -(x - B.μ)*sin(τ) + θ*cos(τ)
    t + τ, x_new, θ
end

# This could work for  ZigZag1d as well
"""
    move_forward!(τ, t, x, θ, Z::Union{Bps, ZigZag})
Updates the position `x`, velocity `θ` and time `t` of the
process after a time step equal to `τ` according to the deterministic
dynamics of the Buoncy particle sampler (`Bps`) and `ZigZag`:
(x(τ), θ(τ)) = (x(0) + θ(0)*t, θ(0)).
`x`: current location, `θ`: current velocity, `t`: current time,
"""
function move_forward!(τ, t, x, θ, Z::Union{Bps, ZigZag})
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
    x_new = (x .- B.μ)*cos(τ) .+ θ*sin(τ) .+ B.μ
    θ .= -(x .- B.μ)*sin(τ) .+ θ*cos(τ)
    t + τ, x_new, θ
end


"""
        reflect!(i, θ, x, Z)
Reflection rule of `ZigZag` sampler at reflection time.
`i`: coordinate which flips sign, `θ`: velocity, `x`: position (not used for
the `ZigZag`)
"""
function reflect!(i, θ, x, F::Union{ZigZag, Bps})
    θ[i] = -θ[i]
    θ
end
