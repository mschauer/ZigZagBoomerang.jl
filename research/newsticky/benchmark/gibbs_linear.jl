
# Y = X \beta + epsilon_i epsilon_i ∼ normal(0, σ2), prior slab = normal(0, cσ2)

function gibbs_linear(Y, X, w, iter, β, Z, σ2, c, subiter = 10)
    ββ = Vector{Float64}[]
    ZZ = Vector{Float64}[]
    push!(ββ, copy(β))
    push!(ZZ, copy(Z))
    for k in 1:iter
        # println("number of active coordinate: $(sum(Z))")
        Z =  update_Z!(Y, X, w, Z, σ2, c)
        # Update β
        β = update_β!(Y, X, β, Z, σ2, c)
        if k%subiter == 0
            push!(ββ, copy(β))
            push!(ZZ, copy(Z))
        end
    end
    return ββ, ZZ
end


function update_Z!(Y, X, w, Z, σ2, c)
    for i in eachindex(Z)
        c1 = compute_L(i, Y, X, Z, σ2, c)
        p =  1/(1 + (1-w[i])*sqrt(c)*c1/(w[i]) ) # c comes from det(V0| Z_i=1)/det(V0|Z_i=0) = c^γ /c^(γ-1)
        if !(0<=p<=1)
            error("probability p = $p is out of range")
        end
        rand() < p ? Z[i] = 1 : Z[i] = 0  
    end
    Z
end

function compute_L(i, Y, X, Z, σ2, c) #ok
    zi = Z[i]
    Z[i] = 0
    γ = sum(Z)
    if γ == 0 
        error("Don't want to be here")
        return 0.0
    end
    Xz0 = view(X, :, Z)
    βhatz0 = Xz0 \ Y # why does not have prior of beta
    Vz0 = cholesky(Symmetric(I./c + Xz0'*Xz0)) 
    Snz0 = dot(Y - Xz0*βhatz0, Y - Xz0*βhatz0) + dot(Xz0*βhatz0, Xz0*βhatz0) - (Xz0'*Y)'*(Vz0\(Xz0'*Y))  
    Z[i] = 1
    Xz1 = view(X, :, Z)
    βhatz1 = Xz1 \ Y
    Vz1 = cholesky(Symmetric(I./c + Xz1'*Xz1))
    Snz1 = dot(Y - Xz1*βhatz1, Y - Xz1*βhatz1) + dot(Xz1*βhatz1, Xz1*βhatz1) - (Xz1'*Y)'*(Vz1\(Xz1'*Y))
    Z[i] = zi 
    return exp(0.5logdet(Vz1) - 0.5logdet(Vz0) - 1/(2*σ2)*(Snz0 - Snz1)) #p
end

function update_β!(Y, X, β, Z, σ2, c) # OK
    γ = sum(Z)
    if γ == 0 
        return β 
    end 
    Xz = view(X, :, Z)
    Vz = cholesky(Symmetric(I(γ)./c + Xz'*Xz))
    μ_z = Vz\(Xz'*Y)
    # Σ_z = σ2*Vz
    βz = Vz.U \ randn(length(μ_z)) + μ_z # check if this is correct
    β[Z] .= βz
    β
end

# Y = X β + ϵ 
run_experiment = false
if run_experiment
    N = 200
    p = 50
    X = randn(N, p)
    β_full = randn(p)*3.0
    β_sparse = β_full.*(abs.(β_full) .> 2.0) # make β spase
    sum(β_sparse .== 0.0)/length(β_sparse)
    σ2 = 2.0
    Y = X*β_sparse .+ randn(N)*sqrt(σ2) 
    @assert N == length(Y)

    w = fill(0.5, p)
    Z = w .> 0
    c = 1.0
    iter = 100
    β0 = randn(p)
    ββ, ZZ = gibbs_linear(Y, X, w, iter, β0, Z, σ2, c)

    β_gibs = [mean(getindex.(ββ, i)[end÷2:end]) for i in 1:p]
    error("")
    using GLMakie
    fig = scatter(1:p, β_sparse, color = (:blue, 0.5))
    fig.axis
    hlines!(fig.axis, [2.0, -2.0])
    scatter!(1:p, β_gibs, color = (:red, 0.5))
end


