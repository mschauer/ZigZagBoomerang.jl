"""
    ContinuousDynamics

Abstract type for the deterministic dynamics of PDMPs
"""
abstract type ContinuousDynamics end
"""
    ZigZag1d <: ContinuousDynamics

Dynamics preserving the Lebesgue measure (1 dimensional ZigZag sampler)
"""

struct ZigZag1d <: ContinuousDynamics  end
"""
    Boomerang1d(Σ, μ, λ) <: ContinuousDynamics

Dynamics preserving the N(μ, Σ) measure (Boomerang1d)
with refreshment time `λ`
"""
struct Boomerang1d{S,T} <: ContinuousDynamics
    Σ::S
    μ::T
    λref::T
end
Boomerang1d(λ) = Boomerang1d(1.0, 0.0, λ)


"""
    struct ZigZag(Γ, μ) <: ContinuousDynamics

Type for local implementation of the ZigZag which exploits
any conditional independence structure of the target measure,
in form the argument Γ, a sparse precision matrix approximating
target precision. μ is the approximate target mean.
"""
struct ZigZag{T,S} <: ContinuousDynamics
    Γ::T
    μ::S
end

"""
    Bps{T} <: ContinuousDynamics
λref::T : refreshment rate which has to be strictly positive
Flag for the Bouncy particle sampler
"""
struct Bps{T} <: ContinuousDynamics
    λref::T
end

"""
    Boomerang(μ, λ) <: ContinuousDynamics

Dynamics preserving the N(μ, 1) measure (Boomerang)
with refreshment time `λ`
"""
struct Boomerang{T, S} <: ContinuousDynamics
    μ::T
    λref::S
end

"""
    FactBoomerang(Γ, μ, λ) <: ContinuousDynamics

Factorized Boomerang dynamics preserving the N(μ, inv(Diagonal(Γ))) measure
with refreshment time `λ`.
Exploits the conditional independence structure of the target measure,
in form the argument Γ, a sparse precision matrix approximating
target precision. μ is the approximate target mean.
"""
struct FactBoomerang{R, T, S} <: ContinuousDynamics
    Γ::R
    μ::T
    λref::S
end
FactBoomerang(Γ, λ) = FactBoomerang(Γ, 0.0, λ)
