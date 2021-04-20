using Random
Random.seed!(5)
using Statistics, ZigZagBoomerang, LinearAlgebra, Test, SparseArrays
parallel = Threads.nthreads() > 1
const D = 2
#n = 1_000_000
n = (2<<19) - D
d = D + n
if parallel
    κ = Threads.nthreads() # number of threads
    Δ = 0.01 # how much samples per epoch
    d2 = d÷κ
    partition = ZigZagBoomerang.Partition(κ, d)
end

K = 20
T = 10.0

const σ2 = 1.0
const γ0 = 0.1


β = randn(D)
X = [randn(n) for i in 1:D]
t = range(0.0, 1.0, length=n)

x = sin.(2pi*t)
y = sum(X .* β) + x + sqrt(σ2)*randn(n)

dia = -0.5ones(n-1)
Γ = 100*sparse(SymTridiagonal(1.0ones(n), dia))^2
GX0 = [X[1] X[2] ones(n)] 
GX = [X[1] X[2]] 

# rough point estimate, trend as constant or piecewise constant
xpost = 0*x
βpost = (GX0\(y))[1:D]    
for i in 1:3 # 3 em steps
    global βpost, xpost
    xpost = (I'*I + σ2*Γ)\(I'*y - sum(X .* βpost))
    βpost = (GX\(y - xpost))
end

# rough estimate of posterior precision matrix using order of standard errors
Γpost = [I(2)*(γ0 + n) zeros(2,n); zeros(2,n)' (Γ + I/σ2)]

if parallel # if we want to do it parallel, we have to break some dependencies in our posterior proxy
    for k1 in 0:κ-1
        for k2 in 0:κ-1
            k2 == k1 && continue
            abs(k2-k1) > 1 && continue
            Γpost[(k1*d2 + 1):(k1+1)*d2,(k2*d2 + 1):(k2+1)*d2] *= 0
        end
    end
end
dropzeros!(Γpost)
μpost = [βpost; xpost]

bias = [0.0, 0.0]
for i in 1:D
    for j = 1:n # use an unbiased estimate of the log-likelihood
        global bias
        bias[i] += X[i][j]*(xpost[j] + X[1][j]*βpost[1] + X[2][j]*βpost[2] - y[j])/σ2
    end
end
bias

# Compute an unbiased estimate of the i'th partial derivative of the negative loglikelihood in a smart way
function ∇ϕ(u, i_, Γ, X, y, (xhat, βhat, bias), K = 20)
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
        for k in 1:K # use an unbiased estimate of the log-likelihood with control variate
            j = rand(1:n)
            a += s*X[i][j]*(x[j] + X[1][j]*β[1] + X[2][j]*β[2] - y[j])
            a -= s*X[i][j]*(xhat[j] + X[1][j]*βhat[1] + X[2][j]*βhat[2] - y[j])
        end
        return a + bias[i]
    end
end 

t0 = 0.0
u = μpost 
x0 = copy(u) + 0.01*randn(d) # jiggle the starting point to see convergence
θ0 = rand([-1.0, 1.0], d)
θ0[1] = θ0[2] = 5*sqrt(1/n)

# Graphical structure of posterior
G2 = [i+D => D .+ rowvals(Γ)[nzrange(Γ, i)] for i in eachindex(y)]
G1 = [1 => [1:d;], 2 => [1:d;]]
G = [G1; G2]

# precision bounds
c = 1e-9*ones(d)
dt = T/100

Z = ZigZag(Γpost, μpost)
if parallel
    trc, _ = @time ZigZagBoomerang.parallel_spdmp(partition, ∇ϕ, t0, x0, θ0, T, c, G, Z, Γ, X, y, (xpost, βpost, bias), K;  adapt=true, progress=true, Δ=Δ)
else
    trc, _ = @time ZigZagBoomerang.spdmp(∇ϕ, t0, x0, θ0, T, c, G, Z, Γ, X, y, (xpost, βpost, bias), K; structured=true, adapt=true, progress=true)
end
J = [1,2, n÷2]
subtrc = subtrace(trc, J)

#ts, xs = ZigZagBoomerang.sep(collect(discretize(subtrc, dt)))
ts, xs = ZigZagBoomerang.sep(subtrc)

# posterior mean
u = mean(trc)
β̂ = u[1:D]
x̂ = u[D+1:end]

β - β̂
x - x̂

fig = Figure()
ax = fig[1,1:3] = Axis(fig, title="x")
fig[2,1] = Axis(fig, title="β1")
fig[2,2] = Axis(fig, title="β2")
fig[2,3] = Axis(fig, title="x[$(J[end]-2)]")
jj = 50
scatter!(ax, t[1:jj:end], y[1:jj:end], markersize=0.1)
lines!(ax, t[1:jj:end], x̂[1:jj:end], color=:lightblue)
lines!(ax, t[1:jj:end], x[1:jj:end], color=:green, linewidth=3.0)
lines!(fig[2,1], ts, getindex.(xs, 1))
lines!(fig[2,1], ts, fill(β[1], length(ts)), color=:green)
lines!(fig[2,2], ts, getindex.(xs, 2))
lines!(fig[2,2], ts, fill(β[2], length(ts)), color=:green)
lines!(fig[2,3], ts, getindex.(xs, 3))
lines!(fig[2,3], ts, fill(x[J[end]-2], length(ts)), color=:green)
display(fig)

using FileIO
save("million.png", fig)