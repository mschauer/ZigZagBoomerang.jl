
using LinearAlgebra
using Test
using ZigZagBoomerang
using SparseArrays


#likelihood N(μℓ, Γℓ) where $Γℓ = R'R/σb and μ = Γ^(-1)R'(ones(d).*c)/σb  
# spike and slab = w *slab + (1-w)*spike with 
#slab N(0, I/σa)
using LinearAlgebra
function reversible_jump(Γ, μ, w, iter, x, z, σa, subiter = 10)
    xx = Vector{Float64}[]
    zz = Vector{Float64}[]
    push!(xx, copy(x))
    push!(zz, copy(z))
    for k in 1:iter
        # println("number of active coordinate: $(sum(Z))")
        update_Z!(Γ, μ, w, z, σa, x)
        # Update β
        update_x!(Γ, μ, w, z, σa, x) 
        if k%subiter == 0
            push!(xx, copy(x))
            push!(zz, copy(z))
        end
    end
    return xx, zz
end


function update_Z!(Γ, μ, w, Z, σa, x)
    for i in rand(eachindex(Z))
        L = BF(Γ, μ, i, w, Z, σa, x) # L = p1/p0 where p1 = p(Z_i = 1 | Z_{-1}, β), p0 = p(Z_i = 0 | Z_{-1}, β),
        p = 1/(1 + (1-w)/(L*w)) # w prob of 1, L evidence of 1
        #@show p
        if !(0<=p<=1)
            error("probability p = $p is out of range")
        end
        if rand() < p 
            Z[i] = 1 
        else 
            Z[i] = 0  
        end
    end
    return
end
function cond(Γ::Matrix, Z, μ)
    C0 = cholesky!(Γ[Z,Z])
    μ0 = μ[Z] - C0\(@view(Γ[Z,.~Z])*(-μ[.~Z]))
    C0, μ0
end
function cond(Γ, Z, μ)
    C0 = cholesky!(Matrix(Γ[Z,Z]))
    μZ = μ[Z] 
    μ[Z] .= 0
    μ0 = zeros(sum(Z))
    for (i,z) in enumerate(findall(Z))
        μ0[i] = -ZigZagBoomerang.idot(Γ, z, μ)
    end
    μ[Z] = μZ
    μ0 = μ[Z] - C0\μ0
    C0, μ0
end

"""
Compute Bayes factor
"""
function BF(Γ, μ, i, w, Z, σa, x) #ok
    zi = Z[i]
    Z[i] = 0
    γ = sum(Z)
    if γ == 0 
        return 0.0
    end
    C0, μ0 = cond(Γ, Z, μ)
    
    Z[i] = 1
    C1, μ1 = cond(Γ, Z, μ)
    Z[i] = zi
    return exp((norm(C1.U*μ1)^2 - logdet(C1) + log(2pi))/2 - (norm(C0.U*μ0)^2 - logdet(C0))/2 )
end

function update_x!(Γ, μ, w, Z, σa, x) # OK
    γ = sum(Z)
    if γ == 0 
        return
    end 

    Cz, μz = cond(Γ, Z, μ)
    xz = Cz.U\randn(length(μz)) + μz # check if this is correct
    x[Z] .= xz
    return
end

