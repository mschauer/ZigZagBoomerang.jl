using Revise
using Random
using RDatasets
using ForwardDiff
using ForwardDiff: Dual
partiali(f, x, i, args...) = ForwardDiff.partials(f([Dual{}(x[j], 1.0*(i==j)) for j in eachindex(x)], args...))[]

using ZigZagBoomerang
g(y, μ, σ) = exp(-(y-μ)^2/(2σ^2))/sqrt(2*π*σ^2) # Gaussian
lo(x) = exp(x)/(1+exp(x))

galaxies = dataset("mass", "galaxies")
Yobs = galaxies.x1/1000
const n = 82
@assert n == length(Yobs)
Random.seed!(1)

const K = 3
const d = 3K - 1
const σ = 1.0
const R = 10 # subsampling


function probs(x)
    if K == 2
        (lo(x[end]), 1-lo(x[end]))
    elseif K == 3
        a, b = exp(x[end]), exp(x[end-1])
        c = 1.0 + a + b
        (a/c, b/c, 1 - a/c - b/c)
    else
        a = exp.(x[end-K+2:end])
        sa = sum(a)
        [a/(1 + sa); 1 - sa/(1 + sa)]
    end
end
lprior(x) = sum(log(g(x[k], 1.0, 10.0)) for k in 1:d)
f(y, x, ps=probs(x)) = sum(ps[k]*g(y, x[k], σ*x[K+k]) for k in 1:K)
ϕ(x, Y) = -lprior(x) - n/R*sum(log(f(y, x)) for y in rand(Y, R))
∇ϕ(x, i, Y) = partiali(ϕ, x, i, Y)

using LinearAlgebra
using SparseArrays
using Statistics



t0 = 0.0
x0 = collect(1:K)*10.0
x0 = [x0; ones(d-K)]
θ0 = rand([-1.0,1.0], d)

Γ = sparse(I, d, d)

c = [20.0 for i in 1:d]

μ = zeros(d)

Z = LocalZigZag(Γ, μ)
T = 2000.0

@time trace, (tT, xT, θT), (acc, num), c = pdmp(∇ϕ, t0, x0, θ0, T, c, Z, Yobs; adapt=true)
@show acc, acc/num, mean(c)

tr = collect(trace)
ts = first.(tr)
xs = last.(tr)

ms = [getindex.(xs, k) for k in 1:d]
ps_ = [probs(x) for x in xs]
ps = [getindex.(ps_, k) for k in 1:K]

using Makie
using Bridge
col = Bridge.viridis(1:K)
p1 = scatter(T*ones(K), xT[1:K])
for k in 1:K
    band!(p1, ts, ms[k] + 1.96ms[k+K], ms[k] - 1.96ms[k+K], color=(col[k],0.3) )
    scatter!(p1, ts, ms[k], markersize=1.8, color=col[k] )
    lines!(p1, ts, ms[k],  color=col[k] )

end

p2 = lines(ts, ps[1], color=col[1])
for k in 2:K
    lines!(p2, ts, ps[k], color=col[k])
end

m = median.(ms)
r = range(0, 40, length=200)
p3 = lines(r, [f(y, m) for y in r])
linesegments!(p3, [repeat(Yobs, inner=2) repeat([0,0.05], outer=n)])

p = hbox(title(p3, "est. density and obs."), title(p2, "Trace p[k]"), title(p1, "Trace μ[k] ± σ[k]"))
save("galaxy.png", title(p, "Galaxy dataset"))
