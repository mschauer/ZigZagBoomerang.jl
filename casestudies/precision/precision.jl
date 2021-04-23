using Random
Random.seed!(5)
using Statistics, ZigZagBoomerang, LinearAlgebra, Test, SparseArrays
n = 20
d = n*(n+1)÷2
N = 200
K = 20
T = 300.0

outer(x) = x*x'
function backform(u, 𝕀)
    L = zeros(𝕀[end][1], 𝕀[end][2])
    for (x, i) in zip(u, 𝕀)
        L[i] = x
    end
    L
end
transform(L, 𝕀) = L[𝕀] 

const σ2 = 1.0
const γ0 = 0.1
dia = -0.4ones(n-1)
Γtrue = sparse(SymTridiagonal(1.0ones(n), dia))

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
        c += -2.0*N/u[i] + 2(u[i]-1)
    end
    c
end 
∇ϕ(utrue, 2, YY,  (𝕀, 𝕁), N)


t0 = 0.0

x0 = utrue + 0.01*randn(d) # jiggle the starting point to see convergence

#te = reshape(ForwardDiff.gradient(u -> ϕ(reshape(u, n, n), Y), backform(x0, 𝕀)[:]), n, n)
#@test norm(Vector(te[𝕀]) - [∇ϕ(x0, i, YY, (𝕀, 𝕁), N) for i in 1:d]) < 10d^2*eps()

θ0 = rand([-1.0, 1.0], d)

# Graphical structure of posterior
G = [i => first.(j) for (i,j) in enumerate(𝕁)]

# precision bounds
c = 0.5ones(d)
dt = T/100
Γ̂ = sparse(1.0I(d))
L̂ = cholesky(sparse(SymTridiagonal(cov(Y')))).L
μ̂ = transform(Matrix(sparse(L̂)), 𝕀)
Z = ZigZag(Γ̂, μ̂)
κ = 1.0ones(d)


trc, _ = @time ZigZagBoomerang.sspdmp(∇ϕ, t0, x0, θ0, T, c, G, Z, κ, YY, (𝕀, 𝕁), N; structured=true, adapt=true, progress=true)

J = [1,2,5]
subtrc = subtrace(trc, J)

#ts, xs = ZigZagBoomerang.sep(collect(discretize(subtrc, dt)))
ts, xs = ZigZagBoomerang.sep(subtrc)

# posterior mean
u = mean(trc)
Lhat = backform(u, 𝕀)
utrue - u


fig = Figure()
ax = fig[1,1:3] = Axis(fig, title="L")
fig[2,1] = Axis(fig, title="x$(𝕁[J[1]])")
fig[2,2] = Axis(fig, title="x$(𝕁[J[2]])")
fig[2,3] = Axis(fig, title="x$(𝕁[J[3]])")
jj = 50
heatmap!(ax, [Matrix(Γtrue) outer(backform(u, 𝕀))])
lines!(fig[2,1], ts, getindex.(xs, 1))
lines!(fig[2,1], ts, fill(utrue[J[1]], length(ts)), color=:green)
lines!(fig[2,2], ts, getindex.(xs, 2))
lines!(fig[2,2], ts, fill(utrue[J[2]], length(ts)), color=:green)
lines!(fig[2,3], ts, getindex.(xs, 3))
lines!(fig[2,3], ts, fill(utrue[J[2]], length(ts)), color=:green)
display(fig)

using FileIO
save("precision.png", fig)