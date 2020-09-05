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

A = sparse(A)
# estimator of the partial derivative of the energy function
# If you want a Gaussian prior just add x[k]
function ∂ψ_tilde(k, y, x, A)
    j = rand(A[k,:].nzind)
    if A[j,k] == 0
        return 0.0
    elseif y[j] == 0
        return count(!iszero, A[k,:])*(A[j,k]*exp(A[j,:]*x))/(1 + exp(A[j,:]*x))
    else
        return  count(!iszero, A[k,:])*((A[j,k]*exp(A[j,:]*x))/(1 + exp(A[j,:]*x))
                        - y[j]*A[j,k])
    end
end

#full gradient
∇ϕ(x, A, y) = dot.(A, exp.(A*x)./(1 .+ exp.(A*x))) .- dot.(A, y)
# Jacobian of ∇ϕ
function Jacobian!(J, x, A, y)
    for i in eachindex(x)
        for k in 1:i
            J[i,k] = J[k,i] = dot(A[:,i].*A[:,k], exp.(A*x)./((1 .+ exp.(A*x)).^2))
        end
    end
    J
end

# Some Newton steps towards the mode as starting point
# The mode can be used as control variate
# Not working, consider stochastic gradient decent
# println("Newton steps")
# x0 = zeros(p)
# H = zeros(p,p)
# @time for i in 1:1
#     println(i)
#     global x0, H
#     H = Jacobian!(H, x0, A, y)
#     x0 = x0 - cholesky(Symmetric(I + H))\∇ϕ(x0, A, y)
# end
# @show norm(xtrue - x0)
#

μ = copy(x0)
# Γfull = ReverseDiff.jacobian(x -> ∇ϕ(x, A, y), μ)
# Γ = sparse(1.0I, p, p)
bias = ∇ϕ(μ, A, y)
@show norm(bias)

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
