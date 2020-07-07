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
struct Boomerang1d{S,T} <: ContinuousDynamics
    Σ::S
    μ::T
    λref::T
end
Boomerang1d(λ) = Boomerang1d(1.0, 0.0, λ)
