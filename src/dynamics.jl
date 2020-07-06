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
