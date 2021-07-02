using ForwardDiff

#= For test: 
using Distributions
using LinearAlgebra

outer(x) = x*x'
d = 5
μ = randn(d)
Σ = outer(randn(d,d)) + I
f(x) = -logpdf(MvNormal(μ, Σ), x)

x = randn(d)
θ = randn(d)
dt = 0.001
∇f, ∂∇f =  ForwardDiff.gradient(f, x), (ForwardDiff.gradient(f, x + dt*θ) - ForwardDiff.gradient(f, x))/dt
=# 

"""
    potential_gradients(ℓ) -> ((∇ϕ, ∂∇ϕ, t, x, θ) -> (nothing, ∇ϕ, v))

Return function with signature `(∇ϕ, ∂∇ϕ, t, x, θ)` which writes the gradient and derivative 
of the gradient in direction of movement of `-ℓ` into `∇ϕ, ∂∇ϕ` and return `∇ϕ` and `dot(v, ∂∇ϕ)`.

Note: Returns `nothing` as first argument as placeholder for `-ℓ(x)`
"""
function potential_gradients(ℓ)
    function (∇ϕ, ∂∇ϕ, t, x, θ)
        res = ForwardDiff.DiffResults.DiffResult(∇ϕ, ∂∇ϕ)
        ForwardDiff.derivative!(res, t -> -ForwardDiff.gradient(ℓ, x + t*θ), 0.0)
        nothing, ∇ϕ, dot(θ, ∂∇ϕ) # missing ϕ
    end
end

function potential_gradients′(ℓ)
    function (y, ∂∇ϕ, t, x, θ, args...)
        T = ForwardDiff.Tag(ℓ, typeof(x))
        x_ = x + ForwardDiff.Dual{T}(0.0, 1.0)*θ
        y_ = ForwardDiff.gradient(x->-ℓ(x, args...), x_)
        y .= ForwardDiff.value.(y_)
        nothing, y, dot(θ, y_).partials[]
    end
end

 