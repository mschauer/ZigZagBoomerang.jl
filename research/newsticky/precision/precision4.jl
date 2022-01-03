#########################################################
#### `git checkout engine` before running the script ####
#########################################################

using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
using Random
Random.seed!(5)
using Statistics, ZigZagBoomerang, LinearAlgebra, Test, SparseArrays
using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using StructArrays
using StructArrays: components

function countids(f, s)
    res = Dict{Int, Int}()
    for c in s; 
        i = f(c)
        res[i] = get(res, i, 0) + 1
    end
    return res
end
seed = (UInt(1),UInt(1))

n = 200
d = n*(n+1)÷2
N = 1000
γ0 = 0.1
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

dia = -0.3ones(n-1)
Γtrue = sparse(SymTridiagonal(1.0ones(n), dia))
Γtrue[1,1] = Γtrue[end,end] = 1/2
Ltrue_ = cholesky(Γtrue).L
Y = Ltrue_'\ randn(n, N)   # see http://www.statsathome.com/2018/10/19/sampling-from-multivariate-normal-precision-and-covariance-parameterizations

YY = Y*Y'
Ltrue = L = sparse(Ltrue_)
# Compute an unbiased estimate of the i'th partial derivative of the negative loglikelihood in a smart way
# function ϕ(L, Y, γ0)
#    # L = reshape(x, d, d)
#    sum(γ0*(diag(L) .- 1.0).^2)/2 - N*sum(log.(diag(L).^2)) + tr(Y'*(L*(L'*Y)))/2 + γ0*sum(L[:].^2)/2
# end
# ϕ(Ltrue, Y, γ0)


utrue_ = Vector(vec(Ltrue))

@test tr(YY*L*L') ≈ sum(Y[:, i]'*L*L'*Y[:,i] for i in 1:N)


𝕀 = [c for c in CartesianIndices((n,n)) if c[1] >= c[2]] # set of indexes |𝕀| = n * (n+1) / 2
𝕁 = [[(i,CartesianIndex(c[1], c2[1])) for (i,c2) in enumerate(𝕀) if c2[1] >= c[2] && c[2] == c2[2]] for c in 𝕀]
# useful for fast product matrix vs triangular matrix

# test 
utrue = Vector(L[𝕀])
if false
    comp = 0.0
    ii = 2
    for (j, c2) in 𝕁[ii]
        global  utrue, YY, comp
        comp += YY[c2] * utrue[j]
    end
    dot(YY[𝕀[ii][1],:],Matrix(Ltrue)[:,𝕀[ii][2]])
    comp
end


function ∇ϕ(u, i, YY, (𝕀, 𝕁), N)
    c = 0.0
    for (j, c2) in 𝕁[i]
         c += YY[c2] * u[j] #L[j,ii[2]] normal
    end
    if 𝕀[i][1] == 𝕀[i][2] 
        c += - N/u[i]  + γ0*(u[i]-1.0)
    else
        c += γ0*(u[i])
    end
    c
end 
#∇ϕ(utrue, 2, YY,  (𝕀, 𝕁), N)


### draw a poisson time with rate t -> (v*(-2N)/(x + vt))⁺
function nl_poisson_time(i, x, v, N, u)
    if v[i] > 0
        return Inf
    else
        return  x[i]*(u^(-v[i]/(N)) - 1)/v[i]
    end
end

function Zig.ab(i, u, v, YY, 𝕀, 𝕁, N)
    a = 0.0
    b = 0.0
    c = 0.0
    for (j, c2) in 𝕁[i]
         a += YY[c2] * u[j] #L[j,ii[2]]
         b += YY[c2] * v[j]
    end
    if 𝕀[i][1] == 𝕀[i][2] # diagonal
        a += γ0*(u[i] - 1.0)
        b += γ0*v[i]
        c += u[i]
    else
        a += γ0*u[i]
        b += γ0*v[i]
    end
    a *= v[i]
    b *= v[i]
    return a + 0.0001, b + 0.0001, c  #add 0.001 for avoiding numerical problems
end

function next_rand_reflect(j, i, t′, u, P, YY, (𝕀, 𝕁), N)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    if m[j] == 1 
        return 0, Inf
    end
    t_old[j] =  t′
    b[j] = ab(j, x, θ, YY, 𝕀, 𝕁, N) # only a, b
    if (𝕀[j][1] == 𝕀[j][2]) # diagonal elements
        return 0,  t[j] + min(poisson_time(b[j][1:2], rand(P.rng)) , nl_poisson_time(j, x, θ, N, rand(P.rng)))
    else # off diagonal
       
        # println("new time proposed $(ttt) [off-diagonal elelement]")
        return 0,   t[j] + poisson_time(b[j][1:2], rand(P.rng))
    end
end

λ1bar((a,b,c), Δt) = max(0.0, a + b*Δt)
function λ2bar(bb, i, t, told, x, θ, YY, (𝕀, 𝕁), N)
    a, b, c = bb[i] 
    if c == 0.0 
        return 0.0 
    else
        if θ[i] < 0.0
            res = -θ[i]*N/(c + θ[i]*(t[i] - told[i]))
            @assert res > 0.0
            return res
        else 
            return 0.0
        end
    end
end

function Zig.rand_reflect!(i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    smove_forward!(G, i, t, x, θ, m, t′, F)
    ∇ϕi = P.∇ϕ(x, i, args...)
    # l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i] , t[i] - t_old[i])
    l, lb = sλ(∇ϕi, i, x, θ, F), λ1bar(b[i], t[i] - t_old[i]) + λ2bar(b, i, t, t_old, x, θ, args...)
    if rand(P.rng)*lb < l
        if l>= lb + eps() # eps takes care of numerical errors
            !P.adapt && error("Tuning parameter `c` too small. Index $(i), l = $(l), lb = $(lb)")
            adapt!(c, i, P.factor)
        end
        smove_forward!(G2, i, t, x, θ, m, t′, F)
        ZigZagBoomerang.reflect!(i, ∇ϕi, x, θ, F)
        return true, neighbours(G1, i)
    else
        return false, G1[i].first
    end
    
end



function next_reset(j, _, t′, u, P, YY, (𝕀, 𝕁), N)
    0, Inf
end


function freeze!(args...)
    Zig.freeze!(0.0, args...)
end
κ = 0.9
1/(1 + (sqrt(γ0/2pi))/κ)

w = 0.05
κ = (sqrt(γ0/2π))/(1/w - 1)


function next_freezeunfreeze(args...)
    Zig.next_freezeunfreeze(0.0, 0.002, args...)
end 


t0 = 0.0
t = zeros(d)
x0 = utrue  + randn(d) # jiggle the starting point to see convergence
L0 = backform(x0, 𝕀)
[L0[i,i] = abs(L0[i,i]) for i in 1:n]
x0 = Vector(vec(L0[𝕀]))
θ0 = ones(d)

c = 0.01ones(d)
dt = T/500
I0 = spzeros(d,d) #+ I(d)
μ0 = zeros(d)
F = Z = ZigZag(I0, μ0)

# Graphical structure of posterior
G = G1 = [i => first.(j) for (i,j) in enumerate(𝕁)]
G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]

b = [ab(i, x0, θ0, YY, 𝕀, 𝕁, N) for i in eachindex(θ0)]  
u0 = StructArray(t=t, x=x0, θ=θ0, θ_old = zeros(d), m=zeros(Int,d), c=c, t_old=copy(t), b=b)
rng = Rng(seed)
t_old = copy(t)
adapt = false
factor = 0.0
P = SPDMP(G, G1, G2, ∇ϕ, F, rng, adapt, factor)
action! = (Zig.reset!, Zig.rand_reflect!, freeze!)
next_action = FunctionWrangler((next_reset, next_rand_reflect, next_freezeunfreeze))
h = Schedule(action!, next_action, u0, T, (P, YY, (𝕀, 𝕁), N))
trc_ = @time simulate(h);
trc = Zig.FactTrace(F, t0, x0, θ0, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_])
error("")

#trc, _ = @time ZigZagBoomerang.sspdmp(∇ϕ, t0, x0, θ0, T, c, G, Z, κ, YY, (𝕀, 𝕁), N; structured=true, adapt=true, progress=true)
J, C = Zig.sep([(i,c) for (i,c) in enumerate(𝕀) if abs(c[1] - c[2]) <= 1])

subtrc = subtrace(trc, J)
dt = T/500
ts, xs = ZigZagBoomerang.sep(collect(discretize(subtrc, dt)))
#ts, xs = ZigZagBoomerang.sep(subtrc)

# posterior mean
u = mean(trc)
Lhat = backform(u, 𝕀)
utrue - u
using GLMakie
ina(i) = "$(𝕀[J[i]][1]),$(𝕀[J[i]][2])"
# fig = Figure(resolution=(900,500))
# ax = fig[1,1:3] = Axis(fig, title="Error Gamma")
# ax1 = fig[2,1] = Axis(fig, title="x$(ina(1))")
# ax2 = fig[2,2] = Axis(fig, title="x$(ina(2))")
# ax3 = fig[2,3] = Axis(fig, title="x$(ina(37))")
# linkaxes!(ax1, ax2, ax3)
# heatmap!(ax, [Matrix(Γtrue); outer(Lhat); Matrix(Ltrue); Lhat], colormap=:vik, colorrange=[-1/2,1/2])
# lines!(fig[2,1], ts, getindex.(xs, 1))
# lines!(fig[2,1], ts, fill(utrue[J[1]], length(ts)), color=:green)
# lines!(fig[2,2], ts, getindex.(xs, 2))
# lines!(fig[2,2], ts, fill(utrue[J[2]], length(ts)), color=:green)
# lines!(fig[2,3], ts, getindex.(xs, 37))
# lines!(fig[2,3], ts, fill(utrue[J[37]], length(ts)), color=:green)
# display(fig)
fig = Figure(resolution=(900,500))
ax01 = fig[1,1] = Axis(fig)
ax02 = fig[1,2] = Axis(fig)
ax1 = fig[2,1] = Axis(fig, title="x$(ina(1))")
ax2 = fig[2,2] = Axis(fig, title="x$(ina(2))")
# ax3 = fig[2,3] = Axis(fig, title="x$(ina(37))")
linkaxes!(ax1, ax2)
heatmap!(ax01, Matrix(Γtrue), colormap=:vik, colorrange=[-1/2,1/2])
heatmap!(ax02, outer(Lhat), colormap=:vik, colorrange=[-1/2,1/2])
lines!(fig[2,1], ts, getindex.(xs, 1))
lines!(fig[2,1], ts, fill(utrue[J[1]], length(ts)), color=:green)
lines!(fig[2,2], ts, getindex.(xs, 2))
lines!(fig[2,2], ts, fill(utrue[J[2]], length(ts)), color=:green)
# lines!(fig[2,3], ts, getindex.(xs, 37))
# lines!(fig[2,3], ts, fill(utrue[J[37]], length(ts)), color=:green)
display(fig)


using FileIO
save("precision2.png", fig)


