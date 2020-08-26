# Implementation of d dimesional Boomerang and Bouncy particle samplers (the two
# most known not-factorised PDMC)
using LinearAlgebra

grad_correct!(y, x, F::Union{BouncyParticle, ZigZag}) = y
function grad_correct!(y, x, F::Union{Boomerang, FactBoomerang})
    @. y -= (x - F.μ)
    y
end
λ(∇ϕx, θ, F::Union{BouncyParticle, Boomerang}) = pos(dot(∇ϕx, θ))
sλ(∇ϕx, θ, F::Union{BouncyParticle, Boomerang}) = λ(∇ϕx, θ, F)
sλ̄((a,b), Δt) = pos(a + b*Δt)
# Here use sparsity as the factorised samplers
function ab(x, θ, c, B::BouncyParticle)
    (c + θ'*(B.Γ*x), θ'*(B.Γ*θ))
end

function ab(x, θ, c, B::Boomerang)
    (sqrt(normsq(θ) + normsq((x - B.μ)))*c, 0.0)
end

function event(t, x, θ, Z::Union{BouncyParticle,Boomerang})
    t, copy(x), copy(θ)
end

function pdmp_inner!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, τref, (acc, num),
     Flow::Union{BouncyParticle, Boomerang}, args...; factor=1.5, adapt=false)
    while true
        if τref < t′
            t, x, θ = move_forward!(τref - t, t, x, θ, Flow)
            #θ = randn!(θ)
            θ = refresh!(θ, Flow)
            τref = t + waiting_time_ref(Flow)
            b = ab(x, θ, c, Flow)
            t′ = t + poisson_time(b, rand())
            push!(Ξ, event(t, x, θ, Flow))
            return t, x, θ, (acc, num), c, b, t′, τref
        else
            τ = t′ - t
            t, x, θ = move_forward!(τ, t, x, θ, Flow)
            ∇ϕx = ∇ϕ!(∇ϕx, x, args...)
            ∇ϕx = grad_correct!(∇ϕx, x, Flow)
            l, lb = sλ(∇ϕx, θ, Flow), sλ̄(b, τ)
            num += 1
            if rand()*lb <= l
                acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ = reflect!(∇ϕx, x, θ, Flow)
                push!(Ξ, event(t, x, θ, Flow))
                b = ab(x, θ, c, Flow)
                t′ = t + poisson_time(b, rand())
                return t, x, θ, (acc, num), c, b, t′, τref
            end
            b = ab(x, θ, c, Flow)
            t′ = t + poisson_time(b, rand())
        end
    end
end

"""
    pdmp(∇ϕ!, t0, x0, θ0, T, c, Flow::Union{BouncyParticle, Boomerang}; adapt=false, factor=2.0)

Run a Bouncy particle sampler (`BouncyParticle`) or `Boomerang` sampler from time,
location and velocity `t0, x0, θ0` until time `T`. `∇ϕ!(y, x)` writes the gradient
of the potential (neg. log density) into y.
`c` is a tuning parameter for the upper bound of the Poisson rate.
If `adapt = false`, `c = c*factor` is tried, otherwise an error is thrown.

It returns a `PDMPTrace` (see [`Trace`](@ref)) object `Ξ`, which can be collected
into pairs `t => x` of times and locations and discretized with `discretize`.
Also returns the `num`ber of total and `acc`epted Poisson events and updated bounds
`c` (in case of `adapt==true` the bounds are multiplied by `factor` if they turn
out to be too small.)
"""
function pdmp(∇ϕ!, t0, x0, θ0, T, c, Flow::Union{BouncyParticle, Boomerang}, args...; adapt=false, factor=2.0)
    t, x, θ, ∇ϕx = t0, copy(x0), copy(θ0), copy(θ0)
    Ξ = Trace(t0, x0, θ0, Flow)
    τref = waiting_time_ref(Flow)
    num = acc = 0
    b = ab(x, θ, c, Flow)
    t′ = t + poisson_time(b, rand())
    while t < T
        t, x, θ, (acc, num), c, b, t′, τref = pdmp_inner!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, τref, (acc, num), Flow, args...; factor=factor, adapt=adapt)
    end
    return Ξ, (t, x, θ), (acc, num), c
end
