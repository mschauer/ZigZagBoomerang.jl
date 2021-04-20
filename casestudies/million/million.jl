using Random
Random.seed!(4)
using Statistics, ZigZagBoomerang, LinearAlgebra, Test, SparseArrays


n = 65536
K = 1000
T = 50.0
const D = 2
const σ2 = 1.0
const γ0 = 0.1
d = D + n

β = randn(D)
X = [randn(n) for i in 1:D]
t = range(0.0, 1.0, length=n)

x = sin.(2pi*t)
y = sum(X .* β) + x + sqrt(σ2)*randn(n)

dia = -0.5ones(n-1)
Γ = 10*sparse(SymTridiagonal(1.0ones(n), dia))^2
Γ2 = [γ0*I(2)/n zeros(2,n); zeros(2,n)' Γ]
dropzeros!(Γ2)
# (GX'*GX)\GX'*y
GX = [X[1] X[2] ones(n)]
βpost = (GX\y)[1:D]
μpost = (I + Γ2)\[βpost; y - sum(X .* βpost)]
Γpost = (Γ2 + I)/σ2


function ∇ϕ(u, i_, Γ, X, y, K = 100)
    β = u[1:D]
    x = @view u[D+1:end]
    if i_ > D
        i = i_ - D
        a = ZigZagBoomerang.idot(Γ, i, x) + (x[i] - y[i])/σ2
        for di in 1:D
            a += β[di]*X[di][i]/σ2  
        end  
        return a
    else #x'*C*x +  (a+B*x)'*A*(B*x+a) on http://www.matrixcalculus.org
        i = i_
        a = γ0*β[i]
        s = n/K/σ2
        for k in 1:K
            j = rand(1:n)
            a += s*X[i][j]*(x[j] + X[1][j]*β[1] + X[2][j]*β[2] - y[j])
        end
        return a  
    end
end 

t0 = 0.0
u = μpost
x0 = [β; x] + 0.1randn(d)
θ0 = rand([-1.0, 1.0], d)
θ0[1] = θ0[2] = 0.05
G2 = [i+D => D .+ rowvals(Γ)[nzrange(Γ, i)] for i in eachindex(y)]
G1 = [1 => [1:d;], 2 => [1:d;]]
G = [G1; G2]
c = 1*[norm(Γ[:, i], 2) for i in 1:n]
c = [θ0[1]; θ0[2]; c]

dt = 1.0

Z = ZigZag(Γpost, μpost)
tr2, _ = @time ZigZagBoomerang.spdmp(∇ϕ, t0, x0, θ0, T, c, G, Z, Γ, X, y, K; structured=true, adapt=true, progress=true)
str = subtrace(tr2, [1,2, n÷2])

ts, xs = ZigZagBoomerang.sep(collect(discretize(str, dt)))

u = mean(tr2)
β̂ = u[1:D]
x̂ = u[D+1:end]

β - β̂
x - x̂

fig = Figure()
ax = fig[1,1:3] = Axis(fig)
fig[2,1] = Axis(fig)
fig[2,2] = Axis(fig)
fig[2,3] = Axis(fig)
scatter!(ax, t, y, markersize=0.1)
lines!(ax, t, x̂, color=:blue)
lines!(ax, t, x)
lines!(fig[2,1], ts, getindex.(xs, 1))
lines!(fig[2,2], ts, getindex.(xs, 2))
lines!(fig[2,3], ts, getindex.(xs, 3))
display(fig)


