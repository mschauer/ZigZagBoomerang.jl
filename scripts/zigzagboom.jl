using Revise
using ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using Profile
const ZZB = ZigZagBoomerang
PLOT = true
PROFILE = false
n = 10 # 10-dimensional problem

# Create sparse 10-dimensional precision matrix
GS = [1=>2:10, 2=>3:5, 3=>5:10, 4=>[5], 5=>[], 6=>7:8, 7=>[], 8=>[], 9=>[10], 10=>[]] # start from conditional dependence tree
S = Matrix(1.0I, n, n)
for (i, nb) in GS, j in nb
    S[i, j] = 0.3randn()
end

Γ = sparse(S * S')

ϕ(x) = 0.5*x'*Γ*x # potential or -log(target density)

#using ForwardDiff
#∇ϕ(x) = ForwardDiff.gradient(ϕ, x)

∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # partial derivative of ϕ(x) with respect to x[i]




t0 = 0.0
x0 = rand(n)
θ0 = rand([-1.0,1.0], n)

c = [norm(Γ[:, i], 2) for i in 1:n]

Z = ZigZag(Γ, x0*0)
T = 200.0

if PROFILE
    Profile.init()
    @time trace, (tT, xT, θT), (acc, num) = @profile pdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
    Profile.clear()
end
@time trace, (tT, xT, θT), (acc, num) = @profile pdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)


xs = last.(collect(trace))
@show acc, num, acc/num


if PLOT
    using Makie
    p1 = Makie.lines(first.(xs), getindex.(xs, 2))
    save("figures/localzigzag.png", title(p1, "ZigZag"))
    p2 = Makie.lines(getindex.(xs, 1), getindex.(xs, 2), getindex.(xs, 3))
    save("figures/localzigzag3d.png", title(p2, "ZigZag 3d"))

    xs2 = last.(collect(discretize(trace, 0.1)))
    p3 = Makie.lines(first.(xs), getindex.(xs, 2), linewidth=0.5)
    Makie.scatter!(p3, first.(xs2), getindex.(xs2, 2), markersize=0.05)
    save("figures/localzigzagdis.png", title(p3, "ZigZag discretized"))
end



λref = 1.0
x0 = randn(n)
θ0 = randn(n)
B = FactBoomerang(Γ, x0*0, λref)
@time trace, (tT, xT, θT), (acc, num) = pdmp(∇ϕ, t0, x0, θ0, T, c, B, Γ)

xs = last.(collect(trace))
@show acc, num, acc/num


if PLOT
    using Makie
    p1 = Makie.lines(first.(xs), getindex.(xs, 2))
    save("figures/factboom.png", title(p1, "Factorised Boomerang"))
    p2 = Makie.lines(getindex.(xs, 1), getindex.(xs, 2), getindex.(xs, 3))
    save("figures/factboom3d.png", title(p2, "Factorised Boomerang 3d"))

    xs2 = last.(collect(discretize(trace, 0.1)))
    p3 = Makie.lines(first.(xs), getindex.(xs, 2), linewidth=0.5)
    Makie.scatter!(p3, first.(xs2), getindex.(xs2, 2), markersize=0.05)
    save("figures/factboomdis.png", title(p3, "Factorised Boomerang discretized"))
end
