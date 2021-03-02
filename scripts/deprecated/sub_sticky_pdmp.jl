using LinearAlgebra
κ = 1.0
n = 100
p = 10
x0 = randn(p)
#linear regression
A = randn(n,p)
σ0 = 0.01
y0 = A*x0 .+ randn(n)*σ0


θ0 = rand((-1.0,1.0), n)
T = 100.0
c = 10ones(n)
function ∇ϕ!(x, j, A, y0, σ0, n)
    i = rand(1:n)
    A[i,j]*(dot(A[i,:],x) - y[j])/σ0^2
end


function ∇ϕ!(x, args...)
    i = rand(1:length(x))
    ∇ϕ!(x, i, args...)
return E_tilde, i


#@time trace0, _ = ZigZagBoomerang.spdmp(ϕ, 0.0, x0, θ0, T, c, ZigZag(sparse(1.0I,n,n), zeros(n)), μ)

@time trace0, _ = ZigZagBoomerang.sspdmp(ϕ, 0.0, x0, θ0, T, c, κ, ZigZag(sparse(1.0I,n,n), zeros(n)), μ)
ts0, xs0 = splitpairs(trace0)
