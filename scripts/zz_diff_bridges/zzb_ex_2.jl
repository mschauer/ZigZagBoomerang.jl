#################################
# Second example of the paper   #
# PDMPs for diffusion bridges   #
#################################
using ZigZagBoomerang, SparseArrays, LinearAlgebra
#using CairoMakie
include("../faberschauder.jl")
const ZZB = ZigZagBoomerang

#Logistic growth model after Lamperti transformation
β = 0.1
r = 0.8
K = 2000.0
b(x) = 0.5*β - r/β + (r/(β*K))*exp(-β*x)
b′(x) =  -(r/K)*exp(-β*x)
b″(x) = (r*β/K)*exp(-β*x)

const L = 6
const T = 500.0
n = (2 << L) + 1
ξ0 = 0randn(n)

u, v = log(50.0)/β, log(1000.0)/β # initial and fianl point
ξ0[1] = u / sqrt(T)
ξ0[end] = v / sqrt(T)
c = ones(n)
c[end] = c[1] = 0.0
θ0 = rand((-1.0, 1.0), n)
θ0[end] = θ0[1] = 0.0 # fix final point
T′ = 20000.0 # final clock of the pdmp
Γ = sparse(1.0I, n, n)
trace, (t, ξ, θ), (acc, num), c = @time spdmp(∇ϕmoving, 0.0, ξ0,
    θ0, T′, c, ZigZag(Γ, ξ0*0), SelfMoving(), L, T, adapt=true);

ts, ξs = splitpairs(discretize(trace, T′/n/50))
S = T*(0:n)/(n+1)

using Makie
using CairoMakie
p1 = lines(S, [exp.(dotψ(ξ, s, L, T)*β) for s in S], linewidth=0.3)
for ξ in ξs[6300:5:end]
    lines!(p1, S, [exp.(dotψ(ξ, s, L, T)*β) for s in S], linewidth=0.3)
end
display(p1)
