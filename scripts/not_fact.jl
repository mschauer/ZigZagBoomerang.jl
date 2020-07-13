using ZigZagBoomerang
using Random
using LinearAlgebra
using SparseArrays
using ForwardDiff
Random.seed!(0)

n = 10 # 10-dimensional problem
# Create sparse 10-dimensional precision matrix
GS = [1=>2:10, 2=>3:5, 3=>5:10, 4=>[5], 5=>[], 6=>7:8, 7=>[], 8=>[], 9=>[10], 10=>[]] # start from conditional dependence tree
S = Matrix(1.0I, n, n)
for (i, nb) in GS
    if !isempty(nb) && i >= minimum(nb)
        error("not a tree")
    end
    if !isempty(nb)
        for j in nb
            S[i, j] = 0.3randn()
        end
    end
end
Γ = sparse(S * S')


ϕ(x) = 0.5*x'*Γ*x # potential or -log(target density)
∇ϕ!(y, x) = ForwardDiff.gradient!(y, ϕ, x)

λref_bps = 1.0
x0, θ0 = randn(n), randn(n)
c_bps = 0.001
T = 1000.0
B = BouncyParticle(Γ, x0*0, λref_bps)
out1, acc = pdmp(∇ϕ!, 0.0, x0, θ0, T, c_bps, B; adapt=false, factor=2.0)
using Plots
dt = 0.1
xx = ZigZagBoomerang.discretize(out1, B, dt)
p1 = plot(getindex.(xx.x, 1), getindex.(xx.x, 2), linewidth=0.4)


λref_boom = 1.0
c_boom = 3.0
B = Boomerang(Γ, x0*0, λref_boom)
out2, acc = pdmp(∇ϕ!, 0.0, x0, θ0, T, c_boom, B; adapt=false, factor=2.0)
dt = 0.1
xx = ZigZagBoomerang.discretize(out2, B, dt)
p2 = plot(getindex.(xx.x, 1), getindex.(xx.x, 2), linewidth=0.4)
