using Revise
using ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
const ZZB = ZigZagBoomerang
n = 10
GS = [1=>2:10, 2=>3:5, 3=>5:10, 4=>[5], 5=>[], 6=>7:8, 7=>[], 8=>[], 9=>[10], 10=>[]]
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

ϕ(x) = 0.5*x'*Γ*x

#using ForwardDiff
#∇ϕ(x) = ForwardDiff.gradient(ϕ, x)

∇ϕ(x, i) = dot(Γ[:,i], x) # sparse computation




t0 = 0.0
x0 = rand(n)
θ0 = rand([-1,1], n)


c = [norm(Γ[:, i], 2) for i in 1:n]

Z = ZigZag(Γ, x0*0)
T = 200.0

@time trace, (tT, xT, θT), (num, acc) = pdmp(∇ϕ, t0, x0, θ0, T, c, Z)

xs = last.(collect(trace))
@show acc, num, acc/num


B = FactBoomerang(Γ, x0*0, 1.0)




using Makie
p1 = Makie.lines(first.(xs), getindex.(xs, 2))
save("localzigzag.png", title(p1, "ZigZag"))
p2 = Makie.lines(getindex.(xs, 1), getindex.(xs, 2), getindex.(xs, 3))
save("localzigzag3d.png", title(p2, "ZigZag 3d"))

xs2 = last.(collect(discretize(trace, 0.1)))
p3 = Makie.lines(first.(xs), getindex.(xs, 2), linewidth=0.5)
Makie.scatter!(p3, first.(xs2), getindex.(xs2, 2), markersize=0.05)
save("localzigzagdis.png", title(p3, "ZigZag discretized"))
