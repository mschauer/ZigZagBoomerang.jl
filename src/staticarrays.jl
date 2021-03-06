using StaticArrays
function ZigZagBoomerang.move_forward!(τ, t, x::SVector, θ, Z::Union{BouncyParticle, ZigZag})
    t += τ
    x += θ .* τ
    t, x, θ
end
ZigZagBoomerang.smove_forward!(τ, t, x::SVector, θ, f, Z::Union{BouncyParticle, ZigZag}) = ZigZagBoomerang.move_forward!(τ, t, x, θ, Z)
function ZigZagBoomerang.reflect!(∇ϕx, x::SVector, θ, ::Boomerang)
    θ -= (2*dot(∇ϕx, θ)/ZigZagBoomerang.normsq(∇ϕx))*∇ϕx
    θ
end
function ZigZagBoomerang.reflect!(∇ϕx, x::SVector, θ, F::BouncyParticle)
    θ -= (2*dot(∇ϕx, θ)/ZigZagBoomerang.normsq(F.L\∇ϕx))*(F.L'\(F.L\∇ϕx))
    θ
end
function ZigZagBoomerang.refresh!(rng, θ::SVector{d}, F) where {d}
    ρ̄ = sqrt(1-F.ρ^2)
    F.ρ*θ + ρ̄*randn(rng, SVector{d})
end

function ZigZagBoomerang.refresh!(rng, θ::SVector{d}, F::BouncyParticle) where {d}
    ρ̄ = sqrt(1-F.ρ^2)
    F.ρ*θ + ρ̄*convert(typeof(θ), F.L'\randn(rng, SVector{d}))
end

