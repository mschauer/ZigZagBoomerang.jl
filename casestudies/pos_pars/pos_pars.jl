using LinearAlgebra
using ZigZagBoomerang
using SparseArrays
# simple illustrative example on how to use the sticky pdmps for sampling
# some parameters which must stay positive 

# x ∼ N(1, I) 1_{(x_i > 0.0)}
∇ϕ(x,i) = x[i] - 1
d = 10
μ = ones(d)
Γ = I(d)
x0 = abs.(randn(d)) # start with x_i > 0
θ0 = rand([-1,1], d)
t0 = 0.0
κ = fill(Inf, d) # never stick at 0
c = fill(eps(), d)
T = 1000.0
modes = [i%2 == 0 ? :restore: :reflect for i in 1:d] # only the parameters with odd index are restricted to be positive
trace0, _, _ = sspdmp2(∇ϕ, t0, x0, θ0, T, c, ZigZag(sparse(1.0I,d,d), ones(d)), κ, modes)

ts0, xs0 = splitpairs(trace0)
using GLMakie
fig = lines(getindex.(xs0, 1), getindex.(xs0, 2))
save("pos_pars.png", fig)

