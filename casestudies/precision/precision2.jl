using Random
Random.seed!(5)
using Statistics, ZigZagBoomerang, LinearAlgebra, Test, SparseArrays
using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using StructArrays
using StructArrays: components
using LinearAlgebra

function countids(f, s)
    res = Dict{Int, Int}()
    for c in s; 
        i = f(c)
        res[i] = get(res, i, 0) + 1
    end
    return res
end
seed = (UInt(1),UInt(1))

n = 400
d = n*(n+1)÷2
N = 1000

T = 600.0

outer(x) = x*x'
function backform(u, 𝕀)
    L = zeros(𝕀[end][1], 𝕀[end][2])
    for (x, i) in zip(u, 𝕀)
        L[i] = x
    end
    L
end
transform(L, 𝕀) = L[𝕀] 

#const σ2 = 1.0
#const γ0 = 0.1
dia = -0.3ones(n-1)
Γtrue = sparse(SymTridiagonal(1.0ones(n), dia))
Γtrue[1,1] = Γtrue[end,end] = 1/2

Ltrue_ = cholesky(Γtrue).L
Y = Ltrue_\randn(n, N)
YY = Y*Y'
Ltrue = L = sparse(Ltrue_)
# Compute an unbiased estimate of the i'th partial derivative of the negative loglikelihood in a smart way
function ϕ(L, Y)
   # L = reshape(x, d, d)
   sum(2*(diag(L) .- 1.0).^2)/2 - N*sum(log.(diag(L).^2)) + tr(Y'*(L*(L'*Y)))/2 
end
using ForwardDiff
ϕ(Ltrue, Y)
utrue_ = Vector(vec(Ltrue))
@test tr(YY*L*L') ≈ sum(Y[:, i]'*L*L'*Y[:,i] for i in 1:N)

𝕀 = [c for c in CartesianIndices((n,n)) if c[1] >= c[2]]
𝕁 = [[(i,CartesianIndex(c[1], c2[1])) for (i,c2) in enumerate(𝕀) if c2[1] >= c[2] && c[2] == c2[2]] for c in 𝕀]
utrue = Vector(L[𝕀])
function ∇ϕ(u, i, YY, (𝕀, 𝕁), N)
    c = 0.0
    for (j, c2) in 𝕁[i]
        #c += YY[ii[1], c2[1]] * u[j] #L[j,ii[2]]
         c += YY[c2] * u[j] #L[j,ii[2]]

    end
    if 𝕀[i][1] == 𝕀[i][2] 
        c += -N/u[i] #+ 2(u[i]-1)
    end
    c
end 
#∇ϕ(utrue, 2, YY,  (𝕀, 𝕁), N)


function next_rand_reflect(j, i, t′, u, P, YY, (𝕀, 𝕁), N)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    if m[j] == 1 
        return 0, Inf
    end
    if !(𝕀[j][1] == 𝕀[j][2])
        b[j] = ab(G1, j, x, θ, c, F)
    else
        b[j] = ab(G1, j, x, θ, c, F) .+ (N/(x[j]), N*2/(x[j]^2))
    end
    t_old[j] = t′
    0, t[j] + poisson_time(b[j], rand(P.rng))
end
function next_reset(j, _, t′, u, P, YY, (𝕀, 𝕁), N)
    0, !(𝕀[j][1] == 𝕀[j][2]) ? Inf : t′ + 0.5*u.x[j]
end


function freeze!(args...)
    Zig.freeze!(0.0, args...)
end

function next_freezeunfreeze(args...)
    Zig.next_freezeunfreeze(0.0, 0.04, args...)
end 


t0 = 0.0
t = zeros(d)

x0 = utrue #+ 0.01*randn(d) # jiggle the starting point to see convergence

#te = reshape(ForwardDiff.gradient(u -> ϕ(reshape(u, n, n), Y), backform(x0, 𝕀)[:]), n, n)
#@test norm(Vector(te[𝕀]) - [∇ϕ(x0, i, YY, (𝕀, 𝕁), N) for i in 1:d]) < 10d^2*eps()

θ0 = ones(d)

Γ̂ = inv(cov(Y'))

# precision bounds
c = 0.01ones(d)
dt = T/500
Γ̂Z = sparse(1.0I(d))*N
#L̂ = cholesky(sparse(SymTridiagonal(cov(Y')))).L
#μ̂ = transform(Matrix(sparse(L̂)), 𝕀)
L̂ = cholesky(Symmetric(Γ̂)).L
μ̂ = transform(L̂, 𝕀)

F = Z = ZigZag(Γ̂Z, μ̂)

# Graphical structure of posterior
G = [i => first.(j) for (i,j) in enumerate(𝕁)]
# Graphical structure of bounds
G1 = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
# What is needed to update clocks
G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]


b = [ab(G1, i, x0, θ0, c, F) for i in eachindex(θ0)]
  
u0 = StructArray(t=t, x=x0, θ=θ0, θ_old = zeros(d), m=zeros(Int,d), c=c, t_old=copy(t), b=b)
rng = Rng(seed)
t_old = copy(t)
adapt = true
factor = 1.7
P = SPDMP(G, G1, G2, ∇ϕ, F, rng, adapt, factor)
action! = (Zig.reset!, Zig.rand_reflect!, freeze!)
next_action = FunctionWrangler((next_reset, next_rand_reflect, next_freezeunfreeze))
h = Schedule(action!, next_action, u0, T, (P, YY, (𝕀, 𝕁), N))

trc_ = @time simulate(h);
trc = Zig.FactTrace(F, t0, x0, θ0, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_])


#trc, _ = @time ZigZagBoomerang.sspdmp(∇ϕ, t0, x0, θ0, T, c, G, Z, κ, YY, (𝕀, 𝕁), N; structured=true, adapt=true, progress=true)

J = [1,2,5]
J, C = Zig.sep([(i,c) for (i,c) in enumerate(𝕀) if abs(c[1] - c[2]) <= 1])

subtrc = subtrace(trc, J)

ts, xs = ZigZagBoomerang.sep(collect(discretize(subtrc, dt)))
#ts, xs = ZigZagBoomerang.sep(subtrc)

# posterior mean
u = mean(trc)
Lhat = backform(u, 𝕀)
utrue - u

using Makie
ina(i) = "$(𝕀[J[i]][1]),$(𝕀[J[i]][2])"
fig = Figure(resolution=(1800,1000))
ax = fig[1,1:3] = Axis(fig, title="Error Gamma")
ax1 = fig[2,1] = Axis(fig, title="x$(ina(1))")
ax2 = fig[2,2] = Axis(fig, title="x$(ina(2))")
ax3 = fig[2,3] = Axis(fig, title="x$(ina(3))")
linkaxes!(ax1, ax2, ax3)

heatmap!(ax, [Matrix(Γ̂); Matrix(Γtrue); outer(Lhat); Matrix(Γtrue) - outer(Lhat)], colormap=:vik, colorrange=[-1/4,1/4])
lines!(fig[2,1], ts, getindex.(xs, 1))
lines!(fig[2,1], ts, fill(utrue[J[1]], length(ts)), color=:green)
lines!(fig[2,2], ts, getindex.(xs, 2))
lines!(fig[2,2], ts, fill(utrue[J[2]], length(ts)), color=:green)
lines!(fig[2,3], ts, getindex.(xs, 3))
lines!(fig[2,3], ts, fill(utrue[J[3]], length(ts)), color=:green)
display(fig)

using FileIO
save("precision.png", fig)