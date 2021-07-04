using LinearAlgebra
using Distributions

function gibbs_linear(Y, X, w, iter, β, Z, σ2, c)
    ββ = Vector{Float64}[]
    ZZ = Vector{Float64}[]
    push!(ββ, deepcopy(β))
    push!(ZZ, deepcopy(Z))
    for k in 2:iter
        # Update active coordinates
        Z =  update_Z!(Y, X, w, Z, σ2, c)
        # Update β
        β = update_β!(Y, X, β, Z, σ2, c)
        push!(ββ, deepcopy(β))
        push!(ZZ, deepcopy(Z))
    end
    return ββ, ZZ
end

function update_Z!(Y, X, w, Z, σ2, c)
    for i in eachindex(Z)
        Z[i] = 0
        c0 = compute_terms(Y, X, Z, σ2, c)
        Z[i] = 1
        c1 = compute_terms(Y, X, Z, σ2, c)
        p =  w[i]/(1 + (1-w[i])*c0*sqrt(c)/(w[i]*c1)) # c comes from det(V0| Z_i=1)/det(V0|Z_i=0) = c^γ /c^(γ-1)
        @assert 0<=p<=1
        rand() < p ? Z[i] = 1 : Z[i] = 0  
    end
    Z
end

function compute_terms(Y, X, Z, σ2, c) #ok
    γ = sum(Z)
    if γ == 0 
        # error("should not be here")
        return 0.0
    end
    Xz = view(X, :, Z)
    βhatz = (Xz'*Xz) \ Xz'*Y
    Vz = inv(Symmetric(I(γ)./c + Xz'*Xz))
    Snz = dot(Y - Xz*βhatz, Y - Xz*βhatz) + dot(Xz*βhatz, Xz*βhatz) - (Xz'*Y)'*Vz*(Xz'*Y)  
    return sqrt(det(Vz))*exp(-1/(2*σ2)*Snz)
end

function update_β!(Y, X, β, Z, σ2, c) # OK
    γ = sum(Z)
    if γ == 0 
        return β 
    end 
    Xz = view(X, :, Z)
    Vz = inv(Symmetric(I(γ)./c + Xz'*Xz))
    μ_z = Vz*Xz'*Y
    Σ_z = σ2*Vz
    βz = rand(MvNormal(μ_z, Σ_z))
    β[Z] .= βz
    β
end

# Y = X β + ϵ 
N = 100
p = 50
X = randn(N, p)
β_full = randn(p)
β_sparse = β_full.*abs.(β_full) .> 0.3 # make β spase
σ2 = 1.0
Y = X*β_sparse .+ randn(N)*sqrt(σ2) 
@assert N == length(Y)

w = fill(0.5, p)
Z = w .< 0
c = 1.0
iter = 100
β0 = randn(p)
gibbs_linear(Y, X, w, iter, β0, Z, σ2, c)