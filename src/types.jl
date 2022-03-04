abstract type PDMPSampler
end

"""
    ContinuousDynamics

Abstract type for the deterministic dynamics of PDMPs
"""
abstract type ContinuousDynamics end

"""
    struct ZigZag(Γ, μ) <: ContinuousDynamics

Local ZigZag sampler which exploits
any independence structure of the target measure,
in form the argument `Γ`, a sparse precision matrix approximating
target precision. `μ` is the approximate target mean.
"""
struct ZigZag{T,S,S2,R} <: ContinuousDynamics
    Γ::T
    μ::S
    σ::S2
    λref::R
    ρ::R
    ρ̄::R
end
ZigZag(Γ, μ, σ=(Vector(diag(Γ))).^(-0.5); λref=0.0, ρ=0.0) = ZigZag(Γ, μ, σ, λref, ρ, sqrt(1-ρ^2))

"""
    BouncyParticle(λ) <: ContinuousDynamics
Input: argument `Γ`, a sparse precision matrix approximating target precision.
Bouncy particle sampler,  `λ` is the refreshment rate, which has to be
strictly positive.
"""
struct BouncyParticle{T, S, R, V, LT} <: ContinuousDynamics
    Γ::T
    μ::S
    λref::R
    ρ::R
    U::V
    L::LT
end
BouncyParticle(Γ, μ, λ; ρ=0.0) = BouncyParticle(Γ, μ, λ, ρ, nothing, (cholesky(Symmetric(Γ)).L))
# simple constructor for first experiments
BouncyParticle(λ, d) = BouncyParticle(1.0I(d), zeros(d), λ, 0.0, nothing)
"""
    GenBouncyParticle(ρ) <: ContinuousDynamics
Input: argument `ρ`, autoregressive coefficient with 0<ρ<1.
"""
struct GenBouncyParticle{R} <: ContinuousDynamics
    #μ::S
    ρ::R
    #U::V
    #L::LT
end
GenBouncyParticle(; ρ=0.9) = GenBouncyParticle(ρ)
"""
    Boomerang(μ, λ) <: ContinuousDynamics

Dynamics preserving the `N(μ, Σ)` measure (Boomerang)
with refreshment time `λ`
"""
struct Boomerang{U, T, S, LT} <: ContinuousDynamics
    Γ::U
    μ::T
    λref::S
    ρ::S
    L::LT
end
Boomerang(Γ, μ, λ; ρ=0.0) = Boomerang(Γ, μ, λ, ρ, (cholesky(Symmetric(Γ)).L))
"""
    FactBoomerang(Γ, μ, λ) <: ContinuousDynamics

Factorized Boomerang dynamics preserving the `N(μ, inv(Diagonal(Γ)))` measure
with refreshment time `λ`.

Exploits the conditional independence structure of the target measure,
in form the argument `Γ`, a sparse precision matrix approximating
target precision. `μ` is the approximate target mean.
"""
struct FactBoomerang{R, T, S} <: ContinuousDynamics
    Γ::R
    μ::T
    σ::T
    λref::S
    ρ::S
    ρ̄::S
end
FactBoomerang(Γ, μ, λ, σ=(Vector(diag(Γ))).^(-0.5); ρ=0.0) = FactBoomerang(Γ, μ, σ, λ, ρ, sqrt(1-ρ^2))

"""
    ZigZag1d <: ContinuousDynamics

1-d toy ZigZag sampler, dynamics preserving the Lebesgue measure.
"""
struct ZigZag1d <: ContinuousDynamics  end

"""
    Boomerang1d(Σ, μ, λ) <: ContinuousDynamics

1-d toy boomerang samper. Dynamics preserving the `N(μ, Σ)` measure
with refreshment time `λ`.
"""
struct Boomerang1d{S,T} <: ContinuousDynamics
    Σ::S
    μ::T
    λref::T
end
Boomerang1d(λ) = Boomerang1d(1.0, 0.0, λ)
Boomerang1d(μstar, λ) = Boomerang1d(1.0, μstar, λ)


"""
    ExtendedForm()

Indicates as `args[1]` that `∇ϕ` 
depends on the extended arguments

    ∇ϕ(t, x, θ, i, t′, Z, args...)

instead of 

    ∇ϕ(x, i, args...)


Can be used to implement `∇ϕ` depending on random coefficients.
"""
struct ExtendedForm
end

abstract type Bound end
struct LocalBound{T} <: Bound
    c::T
end
struct GlobalBound{T} <: Bound
    c::T
end

Base.:*(C::T, a) where {T <: Bound} = T(C.c*a)