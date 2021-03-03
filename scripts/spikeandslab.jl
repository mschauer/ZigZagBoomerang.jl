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

include("exampledesign.jl")
# create mock design matrix with 2 categorical explanatory variables
# and their interaction effects and 2 continuous explanatory variable
A = example_design_matrix(;num_rows = 50_000)

n, p = size(A)
@show n, p


const m = 1 # m Bernoulli samples for each set of covariates
println("Av. number of regressors per column ", mean(sum(A .!= 0, dims=1)), ", row ", mean(sum(A .!= 0, dims=2)))


# Data from the model
xtrue = randn(p) .* (rand(p) .< 0.2)
sigmoid(x) = inv(one(x) + exp(-x))
lsigmoid(x) = -log(one(x) + exp(-x))
y_ = Matrix(rand(n, m) .< sigmoid.(A*xtrue))
y = sum(y_, dims=2)[:]

# gaussian mixture prior and target potential
pri(x, w1 = 0.5, a=(0.1)^2, w2 = 0.5, b=4.0^2) = -log(w1*exp(-x^2/(2*a))/sqrt(2pi*a) + w2*exp(-x^2/(2*b))/sqrt(2pi*b))
∇pri(x, w1 = 0.5, a=(0.1)^2, w2 = 0.5, b=4.0^2) = -(-(w1*x*exp(-x^2/(2*a)))/a^(3/2) - (w2*x*exp(-x^2/(2*b)))/b^(3/2))/(w1*exp(-x^2/(2*a))/sqrt(a) + w2*exp(-x^2/(2*b))/sqrt(b))
ϕ(x, A, y::Vector) = pri.(x) - sum(y .* lsigmoid.(A*x) + (m .- y) .* lsigmoid.(-A*x))


sigmoidn(x) = sigmoid(-x)
nsigmoid(x) = -sigmoid(x)



# Gradient (found using http://www.matrixcalculus.org/ S. Laue, M. Mitterreiter, and J. Giesen. A Simple and Efficient Tensor Calculus, AAAI 2020.)
∇ϕ(x, A, y) = ∇pri.(x) - A'*(y .* sigmoidn.(A*x)) - A'*((m .- y).*nsigmoid.(A*x))
function ∇ϕ!(out, x, A, y)
    out .= ∇pri.(x) - A'*(y .* sigmoidn.(A*x)) - A'*((m .- y).*nsigmoid.(A*x))
    out
end

function ∇ϕ(x, i, A, y) # partial derivative
    s = ∇pri(x[i])
    n, p = size(A)
    for j in 1:n
        z = sum(A[j,r]*x[r] for r in 1:p)
        s += -A[j,i]*(y[j]*sigmoidn(z) + (m - y[j])*nsigmoid(z))
    end
    return s
end

function ∇ϕr(x, i, A, y, control, μ, bias, k) # with subsampling and optional control variate
    s = ∇pri(x[i]) + control*(bias[i] - pri(μ[i]))
    n, p = size(A)
    sampler = Random.SamplerRangeNDL(1:n)
    for _ in 1:k
        j = rand(sampler)
        z = sum(A[j,r]*x[r] for r in 1:p)
        s += -n/k*A[j,i]*(y[j]*sigmoidn(z) + (m - y[j])*nsigmoid(z))
        if control
            z2 = sum(A[j,r]*μ[r] for r in 1:p)
            s -= -n/k*A[j,i]*(y[j]*sigmoidn(z2) + (m - y[j])*nsigmoid(z2))
        end
    end
    return s
end


# Some Newton steps towards the mode as starting point
# The mode can be used as control variate
println("Newton steps")
x0 = zeros(p)
@time for i in 1:8
    println(i)
    global x0
    H = ReverseDiff.jacobian(x -> ∇ϕ(x, A, y), x0)
    x0 = x0 - cholesky(Symmetric(I + H))\∇ϕ(x0, A, y)
end
@show norm(xtrue - x0)


μ = copy(x0)
Γfull = ReverseDiff.jacobian(x -> ∇ϕ(x, A, y), μ)
Γ = sparse(1.0I, p, p)
bias = ∇ϕ(μ, A, y)
@show norm(bias)

xtest = randn(p)
i = 8
@test ∇ϕ(xtest, A, y)[i] ≈ ∇ϕ(xtest, i, A, y)
#@test ∇ϕ(xtest, i, A, y) ≈ mean(∇ϕr(xtest, i, A, y, false, μ, bias, 100) for k in 1:20000)
#@test ∇ϕ(xtest, i, A, y) ≈ mean(∇ϕr(xtest, i, A, y, true, μ, bias, 100) for k in 1:20000)

# Initial values
t0 = 0.0


# Define ZigZag

#σ = sqrt.(diag(inv(Matrix(Γfull))))
#extrema(σ)
σ = 0.1*(Vector(diag(Γ))).^(-0.5) # cheaper

Z = ZigZag(Γ, μ, σ)


θ0 = rand([-1.0,1.0], p) .* σ


# Run sparse ZigZag for T time units and collect trajectory
println("Distance starting point")
@show norm(x0 - xtrue)

println("Run pdmp")
T = 100.0
c = ones(Float64, p) # rejection bounds
c0 = copy(c)
traj, u, (acc,num), c = @time pdmp(∇ϕr, t0, x0, θ0, T, c, Z, A, y, true, μ, bias, 40; adapt=true)
@show acc, num, acc/num
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

ps = [3, 22, 50, p-3, p-2, p-1]
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
p1 = title(hbox(pis...), "Spike and slab Bayesian logistic regression n=$(m*n), p=$p", textsize=20)
save(joinpath("figures","spikeandslab$(typeof(Z).name).png"), p1)

p2 = title(vbox(hbox([text("$p", show_axis=false, textsize=10) for p in ps]...),
    [hbox([i==j ? lines(ts, X[i,:]) : lines(X[i, :], X[j, :]) for i in ps]...) for j in ps]...), "$ps")

r = -3:0.01:3
p3 = lines(r, ∇pri.(r))
lines!(p3, r, exp.(-pri.(r)))

p1
