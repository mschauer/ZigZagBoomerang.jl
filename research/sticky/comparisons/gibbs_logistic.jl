using LinearAlgebra
using Distributions
include("./poyla.jl")
# pgdist = PolyaGamma(1,3.0)
# rand(pgdist)
function gibbs_logistic(Y, X, w, iter, β, Z, ω, σ2)
    ββ = Vector{Float64}[]
    ZZ = Vector{Float64}[]
    ωω = Vector{Float64}[]
    push!(ββ, deepcopy(β))
    push!(ZZ, deepcopy(Z))
    push!(ωω, deepcopy(ω))
    for k in 2:iter
        # Update active coordinates
        Z =  update_Z!(Y, X, w, Z, ω, σ2)
        # Update β
        β = update_β!(Y, X, β, Z, ω, σ2)
        # Update ω
        ω = update_ω!(Y, X, w, β, Z, ω, σ2)
        push!(ββ, deepcopy(β))
        push!(ZZ, deepcopy(Z))
    end
    return ββ, ZZ
end

function update_ω!(Y, X, w, β, Z, ω, σ2)
    Xz = view(X, :, Z)
    βz = view(β, Z)
    ω .= rand.(PolyaGamma.(1, Xz*βz))
    ω
end

function update_Z!(Y, X, w, Z, ω, σ2)
    for i in eachindex(Z)
        Z[i] = 0
        c0 = compute_terms_logistic(Y, X, Z, ω, σ2)
        Z[i] = 1
        c1 = compute_terms_logistic(Y, X, Z, ω, σ2)
        # println("c0 is $(c0)")
        # println("c1 is $(c1)")
        p = w[i]/(1 + (1-w[i])*sqrt(2π*σ2)*c0/(w[i]*c1)) # c comes from det(V0| Z_i=1)/det(V0|Z_i=0) = c^γ /c^(γ-1)
        # p =  w[i]/(1 + (1-w[i])*c0*sqrt(c)/(w[i]*c1)) # c comes from det(V0| Z_i=1)/det(V0|Z_i=0) = c^γ /c^(γ-1)
        if !(0<=p<=1)
            error("p equal to $p")
        end
        rand() < p ? Z[i] = 1 : Z[i] = 0  
    end
    Z
end

function update_β!(Y, X, β, Z, ω, σ2)
    γ = sum(Z)
    if γ == 0 
        return β 
    end 
    Ω = diagm(ω)
    Xz = view(X, :, Z)
    Vz = inv(Symmetric(I(γ)./σ2 + Xz'*Ω*Xz))
    μz = Vz*Xz'*(Y .- 0.5)
    βz = rand(MvNormal(μz, Vz))
    β[Z] .= βz
    β
end

function compute_terms_logistic(Y, X, Z, ω, σ2)
    γ = sum(Z)
    if γ == 0 
        # error("should not be here")
        return 0.0
    end
    Xz = view(X, :, Z)
    Ω = diagm(ω)
    Vz = inv(Symmetric(I(γ)./σ2 + Xz'*Ω*Xz))
    Snz = exp(0.5*(Y .- 0.5)'*Xz*Vz*Xz'*(Y .- 0.5))
    return sqrt(det(2*π*Vz))*Snz
end

#logistic
sigmoid(x) = exp(x)/(1 + exp(x))

# Y = X β + ϵ 
N = 50
p = 5
X = randn(N, p)
β_full = randn(p)
β_sparse = β_full.*(abs.(β_full) .> 0.3) # make β spase
P = sigmoid.(X*β_sparse)
Y = rand(N) .<= P
@assert N == length(Y)

w = fill(0.5, p)
Z = w .> 0
σ2 = 1.0 # prior on β
iter = 100
β0 = randn(p)
ω = rand.(PolyaGamma.(1, X*β0))
gibbs_logistic(Y, X, w, iter, β0, Z, ω, σ2)

X*β0