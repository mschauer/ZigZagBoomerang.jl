using ZigZagBoomerang
using LinearAlgebra
using Random
using SparseArrays
using Distributions
using Statistics
using ZigZagBoomerang: idot
using ReverseDiff
using Test
using FileIO
using GLM

println("Sparse logistic regression")

# Design matrix
Random.seed!(2)
include("sparsedesign.jl")
# create mock design matrix with 2 categorical explanatory variables
# and their interaction effects and 2 continuous explanatory variable
A = sparse_design((20,20), 2, 20)
n, p = size(A)
@show n, p
Γ = A'*A
@show sparsity(A), sparsity(Γ)

const m = 4 # m Bernoulli samples for each set of covariates
println("Av. number of regressors per column ", mean(sum(A .!= 0, dims=1)), ", row ", mean(sum(A .!= 0, dims=2)))

At = SparseMatrixCSC(A')
γ0 = 0.01
# Data from the model
xtrue = 5*randn(p)
sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))
y_ = Matrix(rand(n, m) .< sigmoid.(A*xtrue))
y = sum(y_, dims=2)[:]
ny = m .- y
# prior and potential
# prior precision
ϕ(x, A, y::Vector) = γ0*dot(x,x)/2 - sum(y .* lsigmoid.(A*x) + (m .- y) .* lsigmoid.(-A*x))


# Sparse gradient
# helper functions for sparse gradient
function fdot(A::SparseMatrixCSC, At::SparseMatrixCSC, f, j, x, y)
   rows = rowvals(A)
   vals = nonzeros(A)
   s = zero(eltype(A))
   @inbounds for i in nzrange(A, j)
       s += vals[i]*y[rows[i]]*f(idot(At, rows[i], x))
   end
   s
end

# helper function for sparse gradient estimate through subsampling
function fdotr(A::SparseMatrixCSC, At::SparseMatrixCSC, f, j, x, y, k)
   rows = rowvals(A)
   vals = nonzeros(A)
   s = zero(eltype(A))
   l = length(nzrange(A, j))
   @inbounds for i in rand(nzrange(A, j), k)
       s += l/k*vals[i]*y[rows[i]]*f(idot(At, rows[i], x))
   end
   s
end
sigmoidn(x) = sigmoid(-x)
nsigmoid(x) = -sigmoid(x)

# Gradient (found using http://www.matrixcalculus.org/ S. Laue, M. Mitterreiter, and J. Giesen. A Simple and Efficient Tensor Calculus, AAAI 2020.)
∇ϕ(x, A, At, y) = γ0*x - A'*(y .* sigmoidn.(A*x)) - A'*((m .- y).*nsigmoid.(A*x))

# Element i of the gradient exploiting sparsity
∇ϕ(x, i, A, At, y, ny = m .- y) = γ0*x[i] - fdot(A, At, sigmoidn, i, x, y) - fdot(A, At, nsigmoid, i, x, ny)
# Element i of the gradient exploiting sparsity and random subsampling
∇ϕr(x, i, A, At, y, ny = m .- y, k = 5) = γ0*x[i] - fdotr(A, At, sigmoidn, i, x, y, k) - fdotr(A, At, nsigmoid, i, x, ny, k)

# Tests, to be sure
#@test norm(ReverseDiff.gradient(x->ϕ(x, A, y), xtrue) - ∇ϕ(xtrue, A, At, y)) < 1e-7
#@test norm(ReverseDiff.gradient(x->ϕ(x, A, y), xtrue) - [∇ϕ(xtrue, i, A, At, y) for i in 1:p]) < 1e-7



#res = glm(A, y/m, Binomial(m), LogitLink())
#x0 = coef(res)
#norm(xtrue - x0)

# Some Newton steps towards the mode as starting point
println("Newton steps")
x0 = 0.1rand(p)
@time for i in 1:30
    global x0
    H = hcat((sparse(ReverseDiff.gradient(x -> ∇ϕ(x, i, A, At, y, ny), x0)) for i in 1:p)...)
    x0 = x0 - (Symmetric(0.00I + H))\[∇ϕ(x0, i, A, At, y, ny) for i in 1:p]
end
norm(xtrue - x0)

# Note that the A'A has the same sparsity structure as the Hessian of ϕ
# We can use it to tune the doubly local ZigZig with spdmp sparsifying the
# precision matrix estimate
Γ0 = A'*A
nonzeros(Γ0) .= 1
#@time Γ = sparse(inv(vcov(res)) .* Γ0)
#Γ = sparse(ReverseDiff.hessian(x->ϕ(x, A, y), x0))
Γ = hcat((sparse(ReverseDiff.gradient(x -> ∇ϕ(x, i, A, At, y), x0)) for i in 1:p)...)
μ = copy(x0)

println("Precision Γ is ", 100*nnz(Γ)/length(Γ), "% filled")

# Random initial values
t0 = 0.0
θ0 = rand([-1.0,1.0], p)


# Define ZigZag
c = 5*[norm(Γ[:, i], 2) for i in 1:p] # Rejection bounds
Z = ZigZag(Γ, μ)
T = 2000.0

# Or try Boomerang
if false
    θ0 = [sqrt(Γ[i, i])\randn() for i in 1:p]
    Z = FactBoomerang(Γ, μ, 1/25)
    c = 0.1*[norm(Γ[:, i], 2) for i in 1:p]
    T = 2000.0
end

# Run sparse ZigZag for T time units and collect trajectory
println("Distance starting point")
@show norm(x0 - xtrue)

println("Run spdmp")
traj, u, (acc,num), c = @time spdmp(∇ϕr, t0, x0, θ0, T, c, Z, A, At, y, m .- y, 15; adapt=true)
#traj, u, (acc,num), c = @time spdmp(∇ϕ, t0, x0, θ0, T, c, Z, A, At, y, m .- y; adapt=true)
@show maximum(c ./ ([norm(Γ[:, i], 2) for i in 1:p]))

dt = T/4000
x̂ = mean(x for (t,x) in discretize(traj, dt))
X = Float64[]
ts = Float64[]
for (t,x) in discretize(traj, dt)
    append!(ts, t)
    append!(X, x)
end
X = reshape(X, p, length(X)÷p)

[xtrue x̂ abs.(xtrue - x̂)]

@show norm(xtrue - μ)
@show norm(xtrue - x̂)

println("Plot")
using Makie
using Colors
using GoldenSequences

ps = [4, 22, 50, p-3, p-2, p-1]
p0 = length(ps)
cs = map(x->RGB(x...), (Iterators.take(GoldenSequence(3), p0)))
pis = []
for i in 1:p0
    p_i = lines(ts, X[ps[i], :], color=cs[i])
    lines!(p_i, [0, T], [xtrue[ps[i]], xtrue[ps[i]]], color=cs[i], linewidth=2.0)
    ylabel!(p_i, "var$(ps[i])")
    xlabel!(p_i, "t")
    push!(pis, p_i)
end
p1 = title(hbox(pis...), "Sparse logistic regression n=$n, p=$p")
save(joinpath("figures","logistic$(typeof(Z).name).png"), p1)
p1
