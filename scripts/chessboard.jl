using LinearAlgebra
using Test
using ZigZagBoomerang
using SparseArrays
d = 10
x = randn(d)
∇x = randn(d)
const σa = 1.0
const σb = 1.0
# ϕ(x) = 0.5sum(-(x[i]/σa)^2  - ((2.0 - x[i] + x[mod1(i-1, length(x))])/σb)^2 for i in 1:length(x))

ϕ(x) = 0.5sum(-(x[i]/σa)^2 for i in 1:length(x)) + 0.5*sum( -((0.0 - x[i] + x[i+1])/σb)^2 for i in 1:length(x)-1)

R = [i == j ? 1 : (j - i == -1 ? -1 : 0) for i in 2:d, j in 1:d]
@test [(0.0 - x[i] + x[i+1]) for i in 1:length(x)-1] ≈ R*x
c1 = [(i == 1 || i == d-1) ? -1 : -1 for i in 1:d-1]
@test R*x .+ ones(d-1)  ≈ [(1.0 - x[i] + x[i+1]) for i in 1:length(x)-1]

@test x'*R'*R*x/σb^2  ≈ sum(((0.0 - x[i] + x[i+1])/σb)^2 for i in 1:length(x)-1)
# this is good
Γ = R'*R/σb^2 + I/σa^2
@test  exp(ϕ(x)) ≈ exp((-0.5*(x)'*Γ*(x)))  

#ok
ci = 2.0
ϕ(x) = 0.5sum(-(x[i]/σa)^2 for i in 1:length(x)) + 0.5*sum( -((ci - x[i] + x[i+1])/σb)^2 for i in 1:length(x)-1)
#only for $\sigmaa = Inf$
μ = -inv(Γ)*R'*(ones(d-1).*ci)
x = rand(d)
c = exp(ϕ(x))/exp(-0.5*(x - μ)'*Γ*(x - μ)) 
x = rand(d)
@test exp(ϕ(x)) ≈ exp(-0.5*(x - μ)'*Γ*(x - μ))*c
"ok"  
sΓ = sparse(Γ)
T = 10000.0
Z = ZigZag(sΓ, μ) 
∇ϕ(x, i, Γ, μ) = dot(Γ[:,i],(x - μ))
# prior w = 0.5
wi = 0.2
ki = 1/(sqrt(2*π*σa^2))/(1/wi - 1)
println("k equal to $(ki)")
x0 = rand(d)
κ = ki*ones(length(x0))
c = fill(0.001, d)
θ0 = rand([-0.1,0.1], d)
t0 = 0.0
println("run once to trigger precompilation")
ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, Γ, μ)

println("sticky Zig-Zag")
# timer inside sspdmp2
x0 = fill(10.0, d)
trace, acc = ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, Γ, μ)
ts, xs = ZigZagBoomerang.sep(collect(trace))
tsh, xsh = ZigZagBoomerang.sep(collect(discretize(trace, 1.0)))
traceh = [xs2b[i][j] for i in 1:length(xs2b), j in 1:d]



using GLMakie
produce_heatmap = true
if produce_heatmap
    fig1 = Figure()
    ax = [Axis(fig1[1, j]) for j in 1:1]
    heatmap!(ax[1], traceh[100:end,:])
end
fig1

