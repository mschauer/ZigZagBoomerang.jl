L = 7
n = (2 << L) + 1
T = 2.0 # length diffusion bridge
ξ0 = 0randn(n)
u, v = 2.5, -0.5  # initial and fianl point
ξ0[1] = u/sqrt(T)
ξ0[end] = v/sqrt(T)
c = 2 .^ (lvl.(1:n, L))
#c[end] = 1.0
c[1] = 0.0
θ0 = rand((-1.0, 1.0), n)
#θ0[end] =
θ0[1] =  0.0 # fix final point
T′ = 4000.0 # final clock of the pdmp

Γ = sparse(1.0I, n, n)
trace, (t, ξ, θ), (acc, num), c = @time spdmp(∇ϕmoving, 0.0, ξ0,
    θ0, T′, c, ZigZag(Γ, ξ0*0), SelfMoving(), L, T, adapt=false);
ts, ξs = splitpairs(discretize(trace, T′/n))
S = T*(0:n)/(n+1)



using Makie
p1 = lines(S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
for ξ in ξs[1:5:end]
    lines!(p1, S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
end
display(p1)

p3 = hbox([lines(ts, getindex.(ξs, i)) for i in [2,3,5,9,17,(n+1)÷2 + 1, n]]...)

vbox(p1, p3)

using Statistics
p4 = lines(S, [mean(dotψ(ξ, s, L, T) for ξ in ξs) for s in S])
p4 = lines!(p4, S, @.(u*exp(-β*S)))
p0 = vbox(hbox(p1, p4), p3)
save("figures/diffbridges2.png", p0)
