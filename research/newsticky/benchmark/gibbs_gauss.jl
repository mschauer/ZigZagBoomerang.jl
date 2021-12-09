
# Y = X \beta + epsilon_i epsilon_i ∼ normal(0, σ2), prior slab = normal(0, c*σ2)
# using Pkg
# Pkg.activate(@__DIR__)
# cd(@__DIR__)
using LinearAlgebra
function gibbs_gauss(Γ, μ, w, iter, x, z, σ0, subiter = 10)
    xx = Vector{Float64}[]
    zz = Vector{Float64}[]
    push!(xx, copy(x))
    push!(zz, copy(z))
    for k in 1:iter
        # println("number of active coordinate: $(sum(Z))")
        z =  update_Z!(Γ, μ, w, z, σ0, x)
        # Update β
        x = update_x!(Γ, μ, w, z, σ0, x) 
        if k%subiter == 0
            push!(xx, copy(x))
            push!(zz, copy(z))
        end
    end
    return xx, zz
end


function update_Z!(Γ, μ, w, Z, σ0, x)
    for i in eachindex(Z)
        L = compute_L(Γ, μ, i, w, Z, σ0, x) # L = p1/p0 where p1 = p(Z_i = 1 | Z_{-1}, β), p0 = p(Z_i = 0 | Z_{-1}, β),
        p  = (1/(L*w/(1-w)) + 1)^(-1)
        if !(0<=p<=1)
            error("probability p = $p is out of range")
        end
        rand() < p ? Z[i] = 1 : Z[i] = 0  
    end
    Z
end

function compute_L(Γ, μ, i, w, Z, σ0, x) #ok
    zi = Z[i]
    Z[i] = 0
    γ = sum(Z)
    if γ == 0 
        error("Don't want to be here")
        return 0.0
    end
    Γ0 = view(Γ, Z, Z)
    Σ0 = inv(Symmetric(Γ0))
    μ0 = Σ0*(Γ0 - I./σ0^2)*view(μ, Z)
    x0 = view(x, Z)
    Z[i] = 1
    Γ1 = view(Γ, Z, Z)
    Σ1 = inv(Symmetric(Γ1))
    μ1 = Σ1*(Γ1 - I./σ0^2)*view(μ, Z)
    x1 = view(x, Z)
    L = (2π)^(-1/2)*exp(0.5*(-logdet(Σ1) + logdet(Σ0) - (x1 - μ1)'*Γ1*(x1 - μ1) + (x0 - μ0)'*Γ0*(x0 - μ0)))
    return L
end

function update_x!(Γ, μ, w, Z, σ0, x) # OK
    γ = sum(Z)
    if γ == 0 
        return x
    end 
    # println(Γ) 
    Γz = Symmetric(Matrix(view(Γ, Z, Z)))
    Cz = cholesky(Γz)
    μz = Cz \ (Γz - I/σ0^2)*view(μ, Z)
    xz = Cz.U \ randn(length(μz)) + μz # check if this is correct
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