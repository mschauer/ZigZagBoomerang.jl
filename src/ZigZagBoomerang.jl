module ZigZagBoomerang


# Zig zag and Boomerang reference implementation

export poisson_time, ZigZag, Boomerang, LocalZigZag, pdmp, eventtime, eventposition

eventtime(x) = x[1]
eventposition(x) = x[2]


include("poissontime.jl")

"""
    ContinuousDynamics

Abstract type for the deterministic dynamics of PDMPs
"""
abstract type ContinuousDynamics end
"""
    ZigZag <: ContinuousDynamics

Dynamics preserving the Lebesgue measure (ZigZag sampler)
"""

struct ZigZag <: ContinuousDynamics  end
"""
    Boomerang(μ, λ) <: ContinuousDynamics

Dynamics preserving the N(μ, 1) measure (Boomerang)
with refreshment time `λ`
"""
struct Boomerang{T} <: ContinuousDynamics
    μ::T
    λref::T
end
Boomerang(λ) = Boomerang(0.0, λ)

# ZigZag dynamics (time, space, velocity)
function move_forward(τ, t, x, θ, ::ZigZag)
    τ + t, x + θ*τ , θ
end

# Boomerang dynamics (time, space, velocity)
# dx = -x dt; dv = -v dt
function move_forward(τ, t, x, θ, B::Boomerang)
    x_new = (x - B.μ)*cos(τ) + θ*sin(τ) + B.μ
    θ = -(x - B.μ)*sin(τ) + θ*cos(τ)
    t + τ, x_new, θ
end

pos(x) = max(zero(x), x)


λ(∇ϕ, x, θ, F::ZigZag) = pos(θ*∇ϕ(x))
λ(∇ϕ, x, θ, B::Boomerang) = pos(θ*(∇ϕ(x) - (x - B.μ)))

# affine bounds for Zig-Zag
λ_bar(x, θ, c, ::ZigZag) = pos(c + θ*x)

# constant bound for Boomerang with global bounded |∇ϕ(x)|
# suppose |∇ϕ(x, :Boomerang)| ≤ C. Then λ(x(t),θ(t)) ≤ C*sqrt(x(0)^2 + θ(0)^2)
λ_bar(x, θ, c, B::Boomerang) = sqrt(θ^2 + (x - B.μ)^2)*c #Global bound

# waiting times
ab(x, θ, c, ::ZigZag) = (c + θ*x, one(x))
ab(x, θ, c, B::Boomerang) = (sqrt(θ^2 + (x - B.μ)^2)*c, zero(x))

waiting_time(x, θ, c, Flow::ContinuousDynamics) = poisson_time(ab(x, θ, c, Flow)..., rand())
waiting_time_ref(::ZigZag) = Inf
waiting_time_ref(B::Boomerang) = poisson_time(B.λref, 0.0, rand())


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
    Ξ = [(zero(x), x, θ)]
    τref = waiting_time_ref(Flow)
    τ =  waiting_time(x, θ, c, Flow)
    num = acc = 0
    while t<T
        if τref < τ
            t, x, θ = move_forward(τref, t, x, θ, Flow)
            θ = randn()
            τref = waiting_time_ref(Flow)
            τ =  waiting_time(x, θ, c, Flow)
            push!(Ξ, (t, x, θ))
        else
            t, x, θ = move_forward(τ, t, x, θ, Flow)
            τref -= τ
            τ = waiting_time(x, θ, c, Flow)
            l, lb = λ(∇ϕ, x, θ, Flow), λ_bar(x, θ, c, Flow)
            num += 1
            if rand()*lb < l
                acc += 1
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ = -θ  # In multi dimensions the change of velocity is different:
                        # reflection symmetric on the normal vector of the contour
                push!(Ξ, (t, x, θ))
            end
        end
    end
    return Ξ, acc/num
end


include("discretize.jl")

include("localzigzag.jl")
end
