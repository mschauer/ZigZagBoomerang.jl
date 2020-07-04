"""
    ContinuousDynamics

Abstract type for the deterministic dynamics of PDMPs
"""
abstract type ContinuousDynamics end

eventtime(x) = x[1]
eventposition(x) = x[2]
"""
    Boomerang(μ, λ) <: ContinuousDynamics

Dynamics preserving the N(μ, 1) measure (Boomerang)
with refreshment time `λ`
"""
struct Boomerang{T, S} <: ContinuousDynamics
    μ::T
    λref::S
end
Boomerang(λ) = Boomerang(0.0, λ)

"""
    Bps{T} <: ContinuousDynamics
λref::T : refreshment rate which has to be strivtly positive
Flag for the Bouncy particle sampler
"""
struct Bps{T} <: ContinuousDynamics
    λref::T
end

"""
    move_forward(τ, t, x, θ, ::Bps)
Updates the position `x`, velocity `θ` and time `t` of the
process after a time step equal to `τ` according to the deterministic
dynamics of the Buoncy particle sampler (`Bps`): (x(τ), θ(τ)) = (x(0) + θ(0)*t, θ(0)).
`x`: current location, `θ`: current velocity, `t`: current time,
"""
move_forward

move_forward!(τ, t, x, θ, Z::Bps) = linear_move_forward!(τ, t, x, θ)
const move_forward = move_forward!

function linear_move_forward!(τ, t, x, θ)
    t += τ
    x .+= θ .* τ
    t, x, θ
end
linear_move_forward!(τ, t, x::Float64, θ) = (t + τ, x + θ*τ, θ)


"""
    move_forward(τ, t, x, θ, B::Boomerang)
Updates the position `x`, velocity `θ` and time `t` of the
process after a time step equal to `τ` according to the deterministic
dynamics of the `Boomerang` sampler which are the Hamiltonian
dynamics preserving the Gaussian measure:
: x_t = μ +(x_0 − μ)*cos(t) + v_0*sin(t), v_t = −(x_0 − μ)*sin(t) + v_0*cos(t)
`x`: current location, `θ`: current velocity, `t`: current time.
"""
move_forward!(τ, t, x, θ, B::Boomerang) = circular_move_forward!(τ, t, x, θ, B)
function circular_move_forward!(τ, t, x, θ, B)
    x_new = (x .- B.μ)*cos(τ) .+ θ*sin(τ) .+ B.μ
    θ .= -(x .- B.μ)*sin(τ) .+ θ*cos(τ)
    t + τ, x_new, θ
end
function circular_move_forward!(τ, t, x::Number, θ, B::Boomerang)
    x_new = (x - B.μ)*cos(τ) + θ*sin(τ) + B.μ
    θ = -(x - B.μ)*sin(τ) + θ*cos(τ)
    t + τ, x_new, θ
end

#Poisson rates which determine the first reflection time
λ(∇ϕ, x, θ, F::Bps) = max(0.0, dot(θ, ∇ϕ(x)))
λ(∇ϕ, x, θ, B::Boomerang) = max(0.0, dot(θ, ∇ϕ(x) .- (x .- B.μ)))

# affine bounds for Bps
λ_bar(x, θ, c, ::Bps) = max(0.0, c + dot(θ,x))


# constant bound for Boomerang with global bounded |∇ϕ(x)|
# suppose |∇ϕ(x, :Boomerang)| ≤ C. Then λ(x(t),θ(t)) ≤ C*sqrt(x(0)^2 + θ(0)^2)
λ_bar(x, θ, c, B::Boomerang) = sqrt(norm(θ)^2 + norm(x .- B.μ)^2)*c #Global bound


# waiting times
ab(x, θ, c, ::Bps) = (c + dot(θ,x), 1.0)
ab(x, θ, c, B::Boomerang) = (sqrt(norm(θ)^2 + norm(x - B.μ)^2)*c, 0.0)

# waiting_time
waiting_time_ref(::Bps) = poisson_time(B.λref, 0.0, rand())
waiting_time_ref(B::Boomerang) = poisson_time(B.λref, 0.0, rand())

reflect!(∇ϕ, θ, x, ::Bps) = θ - 2*dot(∇ϕ, θ)/dot(∇ϕ,∇ϕ)*∇ϕ
reflect!(∇ϕ, θ, x, ::Boomerang) = θ - 2*dot(∇ϕ, θ)/dot(∇ϕ,∇ϕ)*∇ϕ


# Algorithm for one dimensional pdmp (ZigZag or Boomerang)
"""
    pdmp(∇ϕ, x, θ, T, Flow::ContinuousDynamics; adapt=true,  factor=2.0)

Run a piecewise deterministic process from location and velocity `x, θ` until time
`T`. `c` is a tuning parameter for the upper bound of the Poisson rate.
If `adapt = false`, `c = c*factor` is tried, otherwise an error is thrown.

Returns vector of tuples `(t, x, θ)` (time, location, velocity) of
direction change events.
"""
function pdmp(∇ϕ, x, θ, T, c, Flow::ContinuousDynamics; adapt=false, factor=2.0)
    scaleT = Flow isa Boomerang ? 1.25 : 1.0
    T = T*scaleT
    t = zero(T)
    Ξ = [(t, x, θ)]
    τref = waiting_time_ref(Flow)
    τ =  poisson_time(ab(x, θ, c, Flow)..., rand())
    while t<T
        if τref < τ
            t, x, θ = move_forward(τref, t, x, θ, Flow)
            θ = randn()
            τref = waiting_time_ref(Flow)
            τ =  poisson_time(ab(x, θ, c, Flow)..., rand())
            push!(Ξ, (t, x, θ))
        else
            t, x, θ = move_forward(τ, t, x, θ, Flow)
            τref -= τ
            τ = poisson_time(ab(x, θ, c, Flow)..., rand())
            l, lb = λ(∇ϕ, x, θ, Flow), λ_bar(x, θ, c, Flow)
            if rand()*lb < l
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ = reflect!(θ, x, Flow)
                push!(Ξ, (t, x, θ))
            end
        end
    end
    return Ξ, (t, x, θ)
end


include("discretize.jl")
include("localzigzag.jl")
