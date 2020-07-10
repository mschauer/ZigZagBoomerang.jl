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

println("Sparse logistic regression")

# Design matrix
Random.seed!(2)
p = 500
const n = 2000
A = I + 0.2sprandn(p, p, 0.01)
println("Av. number of regressors per column: ", mean(sum(A .!= 0, dims=1)))
At = SparseMatrixCSC(A')

# Data from the model
xtrue = randn(p)
sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))
y_ = Matrix(rand(p, n) .< sigmoid.(A*xtrue))
y = sum(y_, dims=2)[:]

# prior and potential
γ0 = 0.01 # prior precision
ϕ(x, A, y::Vector) = γ0*dot(x,x)/2 - sum(y .* lsigmoid.(A*x) + (n .- y) .* lsigmoid.(-A*x))

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
sigmoidn(x) = sigmoid(-x)
nsigmoid(x) = -sigmoid(x)

# Gradient
∇ϕ(x, A, At, y) = γ0*x - A'*(y .* sigmoidn.(A*x)) - A'*((n .- y).*nsigmoid.(A*x))

# Element i of the gradient exploiting sparsity
∇ϕ(x, i, A, At, y, ny = n .- y) = γ0*x[i] - fdot(A, At, sigmoidn, i, x, y) - fdot(A, At, nsigmoid, i, x, ny)

# Tests, to be sure
@test norm(ReverseDiff.gradient(x->ϕ(x, A, y), xtrue) - ∇ϕ(xtrue, A, At, y)) < 1e-7
@test norm(ReverseDiff.gradient(x->ϕ(x, A, y), xtrue) - [∇ϕ(xtrue, i, A, At, y) for i in 1:p]) < 1e-7

# Some Newton steps towards the mode as starting point
x0 = rand(p)
@time for i in 1:4
    global x0
    x0 = x0 - ReverseDiff.hessian(x->ϕ(x, A, y), x0)\ReverseDiff.gradient(x->ϕ(x, A, y), x0)
end
[x0 xtrue]

# Note that the hessian has the same sparsity structure as ∇ϕ
# and we can use it to tune the doubly local ZigZig with spdmp
@time Γ = 0.1I + sparse(ReverseDiff.hessian(x->ϕ(x, A, y), x0))
μ = copy(x0)

println("Precision Γ is ", 100*nnz(Γ)/length(Γ), "% sparse")

# Random initial values
t0 = 0.0
θ0 = rand([-1.0,1.0], p)

# Rejection bounds
c = 2*[norm(Γ[:, i], 2) for i in 1:p]

# Define ZigZag
Z = ZigZag(Γ, μ)

# Run sparse ZigZag for T time units and collect trajectory
T = 200.0
traj, _, (acc,num), c = @time spdmp(∇ϕ, t0, x0, θ0, T, c, Z, A, At, y, n .- y, adapt=true)
@show maximum(c ./ (2*[norm(Γ[:, i], 2) for i in 1:p]))
dt = 0.5
x̂ = mean(x for (t,x) in discretize(traj, dt))
X = Float64[]
for (t,x) in discretize(traj, dt)
    append!(X, x)
end
X = reshape(X, p, length(X)÷p)

[xtrue x̂ abs.(xtrue - x̂)]

@show norm(xtrue - μ)
@show norm(xtrue - x̂)

using Makie
using Colors
using GoldenSequences

p0 = 20
cs = map(x->RGB(x...), (Iterators.take(GoldenSequence(3), p0)))
p1 = scatter(fill(T, p), xtrue, markersize=0.01)
scatter!(p1, fill(T, p0), xtrue[1:p0], markersize=0.2, color=cs)
for i in 1:p0
    lines!(p1, [0, T], [xtrue[i], xtrue[i]], color=cs[i], linestyle=:dot )
    lines!(p1, 0:dt:T, X[i, :], color=cs[i])
end
p1 = title(p1, "Sparse logistic regression p=$p")
save("logistic.png", p1)
p1
