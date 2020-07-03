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

using ForwardDiff
∇ϕ(x) = ForwardDiff.gradient(ϕ, x)
∇ϕ(x, i) = dot(Γ[:,i], x) # sparse computation




t0 = 0.0
x0 = rand(n)
θ0 = rand([-1,1], n)


c = [norm(Γ[:, i], 2) for i in 1:n]

Z = LocalZigZag(Γ, x0*0)
T = 300.0

@time Ξ, (tT, xT, θT), (num, acc) = pdmp(∇ϕ, t0, x0, θ0, T, c, Z)

t, x, θ = deepcopy((t0, x0, θ0))
xs = [x0]
ts = [t0]
for ξ in Ξ
    global t, x, θ
    local i
    i, ti, xi = ξ
    t, x, θ = ZZB.move_forward!(ti - t, t, x, θ, Z)
    θ[i] = -θ[i]
    @test x[i] ≈ xi
    push!(ts, t)
    push!(xs, copy(x))
end

@show acc, num, acc/num

using Makie
p1 = Makie.lines(first.(xs), getindex.(xs, 2))
save("localzigzag.png", p1)
p2 = Makie.lines(getindex.(xs, 1), getindex.(xs, 2), getindex.(xs, 3))
save("localzigzag3d.png", p2)
