
using LinearAlgebra
using Test
using ZigZagBoomerang
using SparseArrays


#likelihood N(μℓ, Γℓ) where $Γℓ = R'R/σb and μ = Γ^(-1)R'(ones(d).*c)/σb  
# spike and slab = w *slab + (1-w)*spike with 
#slab N(0, I/σa)
using LinearAlgebra
function gibbs_gauss(Γ, μ, w, iter, x, z, σa, subiter = 10)
    xx = Vector{Float64}[]
    zz = Vector{Float64}[]
    push!(xx, copy(x))
    push!(zz, copy(z))
    for k in 1:iter
        # println("number of active coordinate: $(sum(Z))")
       # update_Z!(Γ, μ, w, z, σa, x)
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
        L = compute_L(Γ, μ, i, w, Z, σa, x) # L = p1/p0 where p1 = p(Z_i = 1 | Z_{-1}, β), p0 = p(Z_i = 0 | Z_{-1}, β),
        # p  = (1/(L*w/(1-w)) + 1)^(-1)
        p = ((1-w)/(L*w) + 1)^(-1)
        @show p
        if !(0<=p<=1)
            error("probability p = $p is out of range")
        end
        rand() < p ? Z[i] = 1 : Z[i] = 0  
    end
    return
end

function compute_L(Γ, μ, i, w, Z, σa, x) #ok
    zi = Z[i]
    Z[i] = 0
    γ = sum(Z)
    if γ == 0 
        return 0.0
    end
    C0 = cholesky(Γ[Z,Z])
    μ0 = μ[Z] - C0\(Γ[Z,.~Z]*(-μ[.~Z]))
    x0 = view(x, Z)

    
    Z[i] = 1
    C1 = cholesky(Γ[Z,Z])
    μ1 = μ[Z] - C1\(Γ[Z,.~Z]*(-μ[.~Z]))
    x1 = view(x, Z)
    L = (2π)^(-1/2)*exp(0.5*(logdet(C1) - logdet(C0) - norm(C1.U*(x1 - μ1))^2 + norm(C0.U*(x0 - μ0))^2 ))
    return L
end

function update_x!(Γ, μ, w, Z, σa, x) # OK
    γ = sum(Z)
    if γ == 0 
        return
    end 

    Cz = cholesky(Γ[Z,Z])
    μz = μ[Z] - Cz\(Γ[Z,.~Z]*(-μ[.~Z]))

    xz = Cz.U\randn(length(μz)) + μz # check if this is correct
    x[Z] .= xz
    return
end

