using ZigZagBoomerang
using LinearAlgebra
using Random
using SparseArrays
using Distributions
using Statistics
using ZigZagBoomerang: idot, idot_moving!
using ReverseDiff
using Test
using FileIO
using GLM

println("Sparse logistic regression")

# Design matrix
Random.seed!(2)
sparsity(A, d = 3) = round(nnz(A)/length(A), digits=d)
include("sparsedesign.jl")
# create mock design matrix with 2 categorical explanatory variables
# and their interaction effects and 2 continuous explanatory variable
A = sparse_design((20,20), 2, 4*20)
n, p = size(A)
@show n, p
Γ = A'*A
@show sparsity(A), sparsity(Γ)

const m = 1 # m Bernoulli samples for each set of covariates
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

sigmoidn(x) = sigmoid(-x)
nsigmoid(x) = -sigmoid(x)
# helper function for sparse gradient estimate through subsampling
function fdotr(A::SparseMatrixCSC, At::SparseMatrixCSC, j, x, μ, y, ny, k)
   rows = rowvals(A)
   vals = nonzeros(A)
   s = zero(eltype(A))
   r = nzrange(A, j)
   sampler = Random.SamplerRangeNDL(r)
   l = length(r)
   @inbounds for _ in 1:k
       i = rand(sampler)
       u = idot(At, rows[i], x)
       s += l/k*vals[i]*y[rows[i]]*sigmoidn(u)
       s += l/k*vals[i]*ny[rows[i]]*nsigmoid(u)
       u0 = idot(At, rows[i], μ)
       s -= l/k*vals[i]*y[rows[i]]*sigmoidn(u0)
       s -= l/k*vals[i]*ny[rows[i]]*nsigmoid(u0)   end
   s
end

# the same, but advancing only the coordinates necessary with `idot_moving!`
function fdot_moving(A::SparseMatrixCSC, At::SparseMatrixCSC, j, t, x, θ, t′, F, μ, y, ny, k)
   rows = rowvals(A)
   vals = nonzeros(A)
   s = zero(eltype(A))
   r = nzrange(A, j)
   sampler = Random.SamplerRangeNDL(r)
   l = length(r)
   @inbounds for _ in 1:k
       i = rand(sampler)
       u = idot_moving!(At, rows[i], t, x, θ, t′, F)
       s += l/k*vals[i]*y[rows[i]]*sigmoidn(u)
       s += l/k*vals[i]*ny[rows[i]]*nsigmoid(u)
       u0 = idot(At, rows[i], μ)
       s -= l/k*vals[i]*y[rows[i]]*sigmoidn(u0)
       s -= l/k*vals[i]*ny[rows[i]]*nsigmoid(u0)
   end
   s
end



# Gradient (found using http://www.matrixcalculus.org/ S. Laue, M. Mitterreiter, and J. Giesen. A Simple and Efficient Tensor Calculus, AAAI 2020.)
∇ϕ(x, A, At, y) = γ0*x - A'*(y .* sigmoidn.(A*x)) - A'*((m .- y).*nsigmoid.(A*x))

# Element i of the gradient exploiting sparsity
∇ϕ(x, i, A, At, y, ny = m .- y) = γ0*x[i] - fdot(A, At, sigmoidn, i, x, y) - fdot(A, At, nsigmoid, i, x, ny)
# Element i of the gradient exploiting sparsity and random subsampling
∇ϕr(x, i, A, At, μ, y, ny = m .- y, k = 5) = γ0*x[i] - fdotr(A, At, i, x, μ, y, ny, k)
# The same, but exploiting that only dependencies in subsamples need to be evaluated
∇ϕmoving(t, x, θ, i, t′, F, A, At, μ, y, ny = m .- y, k = 5) = γ0*x[i] - fdot_moving(A, At, i, t, x, θ, t′, F, μ, y, ny, k)
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

Γ = hcat((sparse(ReverseDiff.gradient(x -> ∇ϕ(x, i, A, At, y), x0)) for i in 1:p)...)
μ = copy(x0)

Γdrop = droptol!(copy(Γ), 1e-2)
@assert nnz(diag(Γdrop)) == p
println("Precision Γ is ", 100*nnz(Γ)/length(Γ), "% filled")
println("Dropped precision is ", 100*nnz(Γdrop)/length(Γdrop), "% filled")


# Random initial values
t0 = 0.0


# Define ZigZag
c = 0.01*ones(p) # Rejection bounds
c0 = copy(c)
σ = sqrt.(diag(inv(Matrix(Γ))))
#σ = (Vector(diag(Γ))).^(-0.5) # cheaper

Z = ZigZag(Γ, μ, σ, ρ=0.5, λref=0.00)
Zdiag = ZigZag(sparse(Diagonal(Γ)), μ, σ, ρ=0.5, λref=0.00)
Zdrop = ZigZag(Γdrop, μ, σ, ρ=0.5, λref=0.00)
T = 2000.0

θ0 = rand([-1.0,1.0], p) .* σ


# Run sparse ZigZag for T time units and collect trajectory
println("Distance starting point")
@show norm(x0 - xtrue)

println("Run spdmp")
#traj, u, (acc,num), c = @time spdmp(∇ϕr, t0, x0, θ0, T, c, Z, A, At, μ, y, m .- y, 12; adapt=true, factor=5)
traj, u, (acc,num), c = @time spdmp(∇ϕmoving, t0, x0, θ0, T, c, Zdrop, SelfMoving(), A, At, μ, y, m .- y, 10; adapt=true, factor=5)
#traj, u, (acc,num), c = @time spdmp(∇ϕ, t0, x0, θ0, T, c, Z, A, At, y, m .- y; adapt=true)
@show acc/num
@show extrema(c ./ c0)

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
p1 = title(hbox(pis...), "Sparse design logistic regression n=$(m*n), p=$p", textsize=20)
save(joinpath("figures","logistic$(typeof(Z).name).png"), p1)

p2 = title(vbox(hbox([text("$p", show_axis=false, textsize=10) for p in ps]...),
    [hbox([i==j ? lines(ts, X[i,:]) : lines(X[i, :], X[j, :]) for i in ps]...) for j in ps]...), "$ps")

p1
