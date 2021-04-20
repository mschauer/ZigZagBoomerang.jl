using Random
Random.seed!(4)
using Statistics, ZigZagBoomerang, LinearAlgebra, Test, SparseArrays

n = 65536
n = 500
T = 500.0
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
Γ = 100*sparse(SymTridiagonal(1.0ones(n), dia))^2
Γ2 = [γ0*I(2) zeros(2,n); zeros(2,n)' Γ]
dropzeros!(Γ2)

function ∇ϕ(u, i_, Γ, X, y)
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
        a = γ0*β[i] + dot(X[i], x + X[1]*β[1] + X[2]*β[2] - y)/σ2
        return a  
    end
end 

t0 = 0.0
u = [β; x]
x0 = 0.1randn(d)
θ0 = rand([-1.0, 1.0], d)
θ0[1] = θ0[2] = 0.01
G2 = [i+D => D .+ rowvals(Γ)[nzrange(Γ, i)] for i in eachindex(y)]
G1 = [1 => [1:d;], 2 => [1:d;]]
G = [G1; G2]
c = 1*[norm(Γ[:, i], 2) for i in 1:n]
c = [10.0; 10.0; c]

dt = 1.0

Z = ZigZag(Γ2, u)
tr2, _ = @time ZigZagBoomerang.spdmp(∇ϕ, t0, x0, θ0, T, c, G, Z, Γ, X, y; structured=true, adapt=true, progress=true)
ts, xs = ZigZagBoomerang.sep(collect(discretize(tr2, dt)))

u = mean(tr2)
β̂ = u[1:D]
x̂ = u[D+1:end]

β - β̂
x - x̂

fig = scatter(t, y, markersize=0.1)
lines!(t, x̂, color=:blue)
lines!(t, x)
display(fig)