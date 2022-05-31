
# using Pkg
# Pkg.activate(@__DIR__)
# cd(@__DIR__)

#likelihood N(μℓ, Γℓ) where $Γℓ = R'R/σb and μ = Γ^(-1)R'(ones(d).*c)/σb  
# spike and slab = w *slab + (1-w)*spike with 
#slab N(0, I/σa)
using LinearAlgebra
function gibbs_gauss(Γℓ, μℓ, w, iter, x, z, σa, subiter = 10)
    xx = Vector{Float64}[]
    zz = Vector{Float64}[]
    push!(xx, copy(x))
    push!(zz, copy(z))
    for k in 1:iter
        # println("number of active coordinate: $(sum(Z))")
        z =  update_Z!(Γℓ, μℓ, w, z, σa, x)
        # Update β
        x = update_x!(Γℓ, μℓ, w, z, σa, x) 
        if k%subiter == 0
            push!(xx, copy(x))
            push!(zz, copy(z))
        end
    end
    return xx, zz
end


function update_Z!(Γℓ, μℓ, w, Z, σa, x)
    for i in eachindex(Z)
        L = compute_L(Γℓ, μℓ, i, w, Z, σa, x) # L = p1/p0 where p1 = p(Z_i = 1 | Z_{-1}, β), p0 = p(Z_i = 0 | Z_{-1}, β),
        # p  = (1/(L*w/(1-w)) + 1)^(-1)
        p = ((1-w)/(L*w) + 1)^(-1)
        if !(0<=p<=1)
            error("probability p = $p is out of range")
        end
        rand() < p ? Z[i] = 1 : Z[i] = 0  
    end
    Z
end

function compute_L(Γℓ, μℓ, i, w, Z, σa, x) #ok
    zi = Z[i]
    Z[i] = 0
    γ = sum(Z)
    if γ == 0 
        return 0.0
    end
    Γ0 = view(Γℓ, Z, Z)
    Σ0 = inv(Symmetric(Γ0) + I.*σa^2)
    μ0 = Σ0*(Γ0)*view(μℓ, Z)
    x0 = view(x, Z)
    Z[i] = 1
    Γ1 = view(Γℓ, Z, Z)
    Σ1 = inv(Symmetric(Γ1) + I.*σa^2)
    μ1 = Σ1*(Γ1)*view(μℓ, Z)
    x1 = view(x, Z)
    L = (2π)^(-1/2)*exp(0.5*(-logdet(Σ1) + logdet(Σ0) - (x1 - μ1)'*Γ1*(x1 - μ1) + (x0 - μ0)'*Γ0*(x0 - μ0)))
    return L
end

function update_x!(Γℓ, μℓ, w, Z, σa, x) # OK
    γ = sum(Z)
    if γ == 0 
        return x
    end 
    # println(Γ) 
    Γz = Symmetric(Matrix(view(Γ, Z, Z)))
    Σz = Inv(Γz + I.*σ^2)
    Cz = cholesky(Σz)
    μz = Σz*(Γz)*view(μ, Z)
    xz = Cz.L*randn(length(μz)) + μz # check if this is correct
    x[Z] .= xz
    return x
end

try_gauss = false
if try_gauss
    d = 20
    X = randn(50,d)
    # Γ = X'*X
    Γ = Symmetric(Matrix(X'*X + I))
    # Γs = sparse(Γ)   
    μ = ones(d)
    x = randn(d)
    w = 0.4
    N = 100
    Z = ones(Bool, d)
    ββ, ZZ = gibbs_gauss(Γ, μ, w, N, x, Z, 1)
end