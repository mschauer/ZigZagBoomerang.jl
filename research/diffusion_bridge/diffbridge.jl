using ZigZagBoomerang, SparseArrays, LinearAlgebra, Revise
include("faberschauder.jl")

# Drift
α = 1.5
β = 0.1
b(x) = -β*x + α*sin(2pi*x)
# First derivative
b′(x) = -β + 2*α*2pi*cos(2pi*x)
# Second derivative
b″(x) = -2*α*(2pi)^2*sin(2pi*x)
# length diffusion bridge
T = 2.0 

L = 7
n = (2 << L) + 1
ξ0 = 0randn(n)
u, v = 1.5, -0.5  # initial and fianl point
ξ0[1] = u/sqrt(T)
ξ0[end] = v/sqrt(T)
c = ones(n)
c[end] = c[1] = 0.0
θ0 = rand((-1.0, 1.0), n)
θ0[end] = θ0[1] =  0.0 # fix final point
T′ = 5000.0 # final clock of the pdmp

Γ = sparse(1.0I, n, n)
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ!, 0.0, ξ0, θ0, T, 10.0, Boomerang(Γ, ξ0*0, 0.1; ρ=0.9), 1, L, adapt=false);
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ, 0.0, ξ0, rand((-1.0, 1.0), n), T, 40.0*ones(n), ZigZag(Γ, ξ0*0), 5, L, adapt=false);
trace, (t, ξ, θ), (acc, num), c = @time spdmp(∇ϕmoving, 0.0, ξ0,
    θ0, T′, c, ZigZag(Γ, ξ0*0), SelfMoving(), L, T, adapt=true);
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ, 0.0, ξ0, rand((-1.0,1.0), n), T, 100.0*ones(n), FactBoomerang(Γ, ξ0*0, 0.1), 5, L, adapt=false);
ts, ξs = splitpairs(discretize(trace, T′/n))
S = T*(0:n)/(n+1)

### change to false is you want to Plot the results
if true
    error("Stop before plotting the results...")
end

using CairoMakie
p1 = lines(S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
for ξ in ξs[1:5:end]
    lines!(S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
end
display(p1)
save("./diffbridges.png", p1)

