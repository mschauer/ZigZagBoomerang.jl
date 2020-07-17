using Makie, ZigZagBoomerang, SparseArrays, LinearAlgebra

b(x) = -0.1x + 2sin(2pi*x)
b′(x) = -0.1 + 2*2pi*cos(2pi*x)
b″(x) = -2*(2pi)^2*sin(2pi*x)

Λ(t) = 0.5 - abs((t % 1.0) - 1/2)
Λ(t, l⁻) = Λ(t*(1<<l⁻))/sqrt(1<<l⁻)

function dotψ(ξ, t, L)
    0 <= t < 1.0 || error("out of bounds")
    r = 0.0
    for i in 0:L
        j = floor(Int, t * (1 << (L - i)))*(2 << i) + (1 << i)
        r += ξ[j]*Λ(t, L-i)
    end
    r
end
function lvl(i)
    l = 0
    while (i & 1) == 0
        l += 1
        i = i >> 1
    end
    l
end
# l in 0:L
function ∇ϕ(ξ, i, K, L) # formula (17)
    l = lvl(i)
    k = i ÷ (2 << l)
    δ = 1/(1 << (L-l))
    r = 0.0
    for _ in 1:K
        s = δ*(k + rand())
        x = dotψ(ξ, s, L)
        r += δ*Λ(s, L-l)*(2b(x)*b′(x) + b″(x)) + ξ[i]
    end
    r/K
end
function ∇ϕ!(y, ξ, k, L)
    for i in eachindex(ξ)
        y[i] = ∇ϕ(ξ, i, k, L)
    end
    y
end

L = 7
n = (2 << L) - 1
ξ0 = 0randn(n)
θ0 = randn(n)
T = 2000.0
Γ = sparse(1.0I, n, n)
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ!, 0.0, ξ0, θ0, T, 10.0, Boomerang(Γ, ξ0*0, 0.1; ρ=0.9), 1, L, adapt=false);
trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ, 0.0, ξ0, rand((-1.0, 1.0), n), T, 40.0*ones(n), ZigZag(Γ, ξ0*0), 5, L, adapt=false);
#trace, (t, ξ, θ), (acc, num) = @time pdmp(∇ϕ, 0.0, ξ0, rand((-1.0,1.0), n), T, 100.0*ones(n), FactBoomerang(Γ, ξ0*0, 0.1), 5, L, adapt=false);

ts, ξs = splitpairs(discretize(trace, T/n))

S = (0:n)/(n+1)
p1 = lines(S, [dotψ(ξ, s, L) for s in S], linewidth=0.3)
for ξ in ξs[1:5:end]
    lines!(p1, S, [dotψ(ξ, s, L) for s in S], linewidth=0.3)
end

p2 = surface([dotψ(ξ, s, L) for s in S, ξ in ξs], shading=false, show_axis=false, colormap = :deep)
scale!(p2, 1.0, 1.0, 100.)

p3 = hbox([lines(ts, getindex.(ξs, i)) for i in [1,2,4,8,16,(n+1)÷2]]...)
vbox(p1, p2, p3)
