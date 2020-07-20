using Makie, ZigZagBoomerang, SparseArrays, LinearAlgebra
# using CairoMakie
include("faberschauder.jl")

# Drift
const α = 1.0
const β = 0.1
b(x) = -β*x + α*sin(2pi*x)
# First derivative
b′(x) = -β + 2*α*2pi*cos(2pi*x)
# Second derivative
b″(x) = -2*α*(2pi)^2*sin(2pi*x)

L = 7
n = (2 << L) + 1
T = 2.0 # length diffusion bridge
ξ0 = 0randn(n)
u, v = 1.5, -0.5  # initial and fianl point
ξ0[1] = u/sqrt(T)
ξ0[end] = v/sqrt(T)
c = ones(n)
c[end] = c[1] = 0.0
θ0 = rand((-1.0, 1.0), n)
θ0[end] = θ0[1] =  0.0 # fix final point
T′ = 2000.0 # final clock of the pdmp

Γ = sparse(1.0I, n, n)
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ!, 0.0, ξ0, θ0, T, 10.0, Boomerang(Γ, ξ0*0, 0.1; ρ=0.9), 1, L, adapt=false);
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ, 0.0, ξ0, rand((-1.0, 1.0), n), T, 40.0*ones(n), ZigZag(Γ, ξ0*0), 5, L, adapt=false);
trace, (t, ξ, θ), (acc, num), c = @time spdmp(∇ϕmoving, 0.0, ξ0,
    θ0, T′, c, ZigZag(Γ, ξ0*0), SelfMoving(), L, T, adapt=true);
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ, 0.0, ξ0, rand((-1.0,1.0), n), T, 100.0*ones(n), FactBoomerang(Γ, ξ0*0, 0.1), 5, L, adapt=false);
ts, ξs = splitpairs(discretize(trace, T′/n))
S = T*(0:n)/(n+1)


#using CairoMakie
p1 = lines(S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
for ξ in ξs[1:5:end]
    lines!(p1, S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
end
display(p1)


p3 = hbox([lines(ts, getindex.(ξs, i)) for i in [1,2,4,8,16,(n+1)÷2]]...)

save("figures/diffbridges.png", p1)
p0 = vbox(p1, p3)
