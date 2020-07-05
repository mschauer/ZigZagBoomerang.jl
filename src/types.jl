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
    Boomerang1d(μ, λ) <: ContinuousDynamics

Dynamics preserving the N(μ, 1) measure (Boomerang1d)
with refreshment time `λ`
"""
struct Boomerang1d{T} <: ContinuousDynamics
    μ::T
    λref::T
end
Boomerang1d(λ) = Boomerang1d(0.0, λ)

"""
    struct ZigZag(Γ, μ) <: ContinuousDynamics

Flag for local implementation of the ZigZag which exploits
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
λref::T : refreshment rate which has to be strivtly positive
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
    FactBoomerang(μ, λ) <: ContinuousDynamics

Dynamics preserving the N(μ, 1) measure (Boomerang)
with refreshment time `λ`
"""
struct FactBoomerang{R, T, S} <: ContinuousDynamics
    Γ::R
    μ::T
    λref::S
end
FactBoomerang(Γ, λ) = FactBoomerang(Γ, 0.0, λ)
