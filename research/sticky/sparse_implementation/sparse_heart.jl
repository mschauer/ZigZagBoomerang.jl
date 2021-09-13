using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
#using Revise
using ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using FileIO
using Statistics
# using Makie, AbstractPlotting

function gridlaplacian(T, m, n)
    S = sparse(T(0.0)I, n*m, n*m)
    linear = LinearIndices((1:m, 1:n))
    for i in 1:m
        for j in 1:n
            for (i2, j2) in ((i + 1, j), (i, j + 1))
                if i2 <= m && j2 <= n
                    S[linear[i, j], linear[i2, j2]] -= 1.
                    S[linear[i2, j2], linear[i, j]] -= 1.

                    S[linear[i, j], linear[i, j]] += 1.
                    S[linear[i2, j2], linear[i2, j2]] += 1.
                end
            end
        end
    end
    S
end

Random.seed!(1)


# Define precision operator of a Gaussian random field (sparse matrix operating on `vec`s of `n*n` matrices)
#n = 100
n = 100
const σ2 = 0.5
Γ0 = 2gridlaplacian(Float64, n, n)
Γ = 0.1I + Γ0
mat(x) = reshape(x, (n, n)) # vector to matrix
function mat0(y)
    mat(y  .- 0.1)
end
# Γ is very sparse
@show nnz(Γ)/length(Γ) # 0.000496

# Corresponding Gaussian potential
# ϕ(x, Γ, y) = 0.5*x'*Γ*x  + dot(x - y, x - y)/(2*σ2) # not used by the program

# Define ∇ϕ(x, i, Γ) giving the partial derivative of ϕ(x) with respect to x[i]
∇ϕ(x, i, Γ, y) = ZigZagBoomerang.idot(Γ, i, x)  + (x[i]-y[i])/σ2 # more efficient that dot(Γ[:, i], x)


# Random initial values
t0 = 0.0
h(x, y) = x^2+(5y/4-sqrt(abs(x)))^2
heart = [ max.(1 - h(x, y), 0) for x in range(-1.5,1.5, length=n), y   in range(-1.1,1.9, length=n)]
# image(heart)
μ0 = 5.0*vec(heart)
y = μ = μ0 + randn(n*n)
μpost = yhat = (I + Γ)\y
Γpost = (Γ + I)/σ2


x0 = μpost
xs0 = [abs(xi)<=0.5 ?  0.0 : xi for xi in x0]  
xs0 = sparse(xs0)
θf0 = rand([-1.0,1.0], n*n)
θ0 = [iszero(xs0[i]) ? 0.0 : θf0[i] for i in eachindex(θf0)]
# Rejection bounds
c = [norm(Γpost[:, i], 2) for i in 1:n*n]
# Define ZigZag
Z = ZigZag(Γpost, μpost)
# or try the FactBoomerang
#Z = FactBoomerang(Γ, x0*0, 0.1)
κ2 = 0.4
# Run sparse ZigZag for T time units and collect trajectory
T = 100.0
su = false
adapt = false
trace, (t, x, θ), (acc, num), c = @time sspdmp(∇ϕ, t0, xs0, θ0, θf0, T, c, Z, κ2, nothing, Γ, μ;
                                                strong_upperbounds = su ,
                                                adapt = adapt)

@time traj = collect(discretize(trace, 0.2))
error("")
