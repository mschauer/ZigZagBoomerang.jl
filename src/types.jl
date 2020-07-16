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
struct ZigZag{T,S,R} <: ContinuousDynamics
    Γ::T
    μ::S
    σ::S
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
struct BouncyParticle{T, S, R} <: ContinuousDynamics
    Γ::T
    μ::S
    λref::R
    ρ::R
end
BouncyParticle(Γ, μ, λ; ρ=0.0) = BouncyParticle(Γ, μ, λ, ρ)

"""
    Boomerang(μ, λ) <: ContinuousDynamics

Dynamics preserving the `N(μ, Σ)` measure (Boomerang)
with refreshment time `λ`
"""
struct Boomerang{U, T, S} <: ContinuousDynamics
    Γ::U
    μ::T
    λref::S
    ρ::S
end
Boomerang(Γ, μ, λ; ρ=0.0) = Boomerang(Γ, μ, λ, ρ)
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
