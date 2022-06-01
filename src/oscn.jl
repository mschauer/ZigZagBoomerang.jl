"""
    oscn!(rng, v, ∇ψx, ρ; normalize=false)
    
Orthogonal subspace Crank-Nicolson step with autocorrelation `ρ` for
standard Gaussian or Uniform on the sphere (`normalize = true`).
"""
function oscn!(rng, v, ∇ψx, ρ; normalize=false)
    # Decompose v
    vₚ = (dot(v, ∇ψx)/normsq(∇ψx))*∇ψx
    v⊥ = ρ*(v - vₚ)
    if ρ == 1
        @. v = v - 2vₚ 
    else
        # Sample and project
        z = randn!(rng, similar(v)) * √(1.0f0 - ρ^2)
        z -= (dot(z, ∇ψx)/dot(∇ψx, ∇ψx))*∇ψx
        if normalize
            λ = sqrt(1 - norm(vₚ)^2)/norm(v⊥ + z)
            @. v = -vₚ + λ*(v⊥ + z)
        else
            @. v = -vₚ + v⊥ + z
        end
    end
    v
end