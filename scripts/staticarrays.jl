using Revise
using StaticArrays
using ZigZagBoomerang
using LinearAlgebra
using SparseArrays
using Random

Random.seed!(1)

const d = 2
const 𝕏 = SArray{Tuple{d},Float64,1,d}

n = 5

Γ = sprand(n, n, 0.1).*[0.25*SMatrix{d,d}(randn(4)) for i in 1:n, j in 1:n]
Γ = Γ + Γ' + Diagonal(fill(SMatrix{d,d}(1.0I), n))

∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x)

B
t0 = 0.0
x0 = randn(𝕏, n)
θ0 = [randn(𝕏) for i in 1:n]

μ = 0*x0
c = [50.0 for i in 1:n]
σ = [SMatrix{d,d}(1.0I) for i in 1:n]
Z = ZigZag(Γ, μ, σ; λref=0.05, ρ=0.8)
T = 2000.0

@time trace, (tT, xT, θT), (acc, num) = spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
xs = last.(collect(discretize(trace, 0.01)))

using Makie
using Colors
using GoldenSequences
cs = map(x->RGB(x...), (Iterators.take(GoldenSequence(3), n)))


p1 = scatter(Point2f0.(x0), markersize=0.1, color=cs)
#for x in xs
#    scatter!(p1, Point2f0.(x), markersize=0.1)
#end
for i in 1:n
    scatter!(p1, ((Point2f0∘getindex).(xs, i)), color=cs[i], markersize=0.01)

end
p1
