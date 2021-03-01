
# linear stochastic differential equation
using LinearAlgebra
using ZigZagBoomerang
const ZZB = ZigZagBoomerang

n = 10
X  = -1.0*I(n) + zeros(n,n)
for i in 1:n
    j = rand(collect(1:n)[1:end .!= i]) # pick a random off diagonal element
    X[i,j] = 1.0
end
X = 2.0*X



# Simulate data y
function simulate_data(X)
    n = size(X)[1]
    σ = 0.1
    b(y) = X*y
    sigma(x) = fill(σ, n)
    δt = 0.01
    T = 1.0
    tt = range(0.0, step = δt, stop = T)
    Y = zeros(n, length(tt))
    y0 = randn(n)
    Y[:, 1] = y0
# euler forward simulation
    for i in eachindex(tt)[2:end]
        dt = tt[i] - tt[i-1]
        y = Y[:, i-1]
        Y[:, i] = y + dt*b(y) + sigma(y).*randn(n)
    end
    Y, tt
end
Y, tt = simulate_data(X)
using Plots
plot(tt, Y[1, :])


## helper for ∫y^(i)_t dy^(j)_t for each i,j
# with ito approximation: see https://en.wikipedia.org/wiki/It%C3%B4_calculus
function tilde_Psi(XX, i, j, tt)
    res = 0.0
    for k in eachindex(tt)[2:end]
        yi = XX[i, k-1]
        dyj = (XX[j, k] - XX[j, k-1])
        res += yi*dyj
    end
    res
end

## helper for ∫y^(i)_t y^(j)_t dt for each i,j
function Psi(XX, i, j, tt)
    res = 0.0
    for k in eachindex(tt)[2:end]
        yi = (XX[i, k-1] + XX[i, k])/2
        yj = (XX[j, k-1] + XX[j, k])/2
        dt = tt[k] - tt[k-1]
        res += yi*yj*dt
    end
    res
end

#pre-computing matrices
tilde_Ψ = [tilde_Psi(Y, i[1], i[2], tt) for i in CartesianIndices(X)]
tilde_Ψt = tilde_Ψ'

Ψ = [Psi(Y, i[1], i[2], tt) for i in CartesianIndices(X)]
Ψ - Ψ' ≈ zeros(n,n) ? println("symmetric matrix") : error("not symmetric matrix")

# x sparse, y whatever
function sdot(x, y)
    r = 0.0
    @inbounds for i in 1:length(x)
        if x[i] == 0
            continue
        else
            r += x[i]*y[i]
        end
    end
    r
end

# matrix to vector

# vector to matrix

# stupid: getting column of matrix from vector indexation of matrix
# col(k,n) = div(k-1, n) + 1
# row(k,n) = rem(k-1, n)+1

firstelement(k, n) = div(k-1,n)*n + 1
lastelement(k, n) = (div(k-1,n)+1)*(n)
# partial ϕi (Gaussian)
function ∇ϕ(x, k, tilde_Ψ, tilde_Ψt, Ψ, γ0, n)
    k1, k2 = firstelement(k, n), lastelement(k, n)
    if !(k1 <= k <= k2)
        error("something is wrong")
    end
    return -tilde_Ψt[k] + sdot(x[k1:k2], Ψ[k1:k2]) + γ0*x[k]
end


## Override ab
function ZZB.ab(G, k, x, θ, c, F::ZigZag,  tilde_Ψ, tilde_Ψt, Ψ, γ0, n)
    k1, k2 = firstelement(k, n), lastelement(k, n)
    a = c + (θ[k]*(-tilde_Ψt[k] + sdot(x[k1:k2], Ψ[k1:k2]) + γ0*x[k]))
    b = θ[k]*(sdot(θ[k1:k2], Ψ[k1:k2]) + γ0*θ[k])
    return a, b
end

# Prior σ0^(-1)
γ0 = 0.2
#kappa given the prior (here Gaussian) and w
nz = sum(X[:] .!= 0) #non zero elements
w = nz/(n*n) # proportion of non zero elemenets
κ = (γ0/sqrt(2π))/(1/w -1) 
p = n*n
μ = zeros(p)
# TODO
# Γ = ones(p,p)
Γ = ones(p,p)


Z = ZigZag(Γ, μ)
c = 0.001
t0, x0, T = 0.0, randn(p), 1000.0
θ0 = rand([-1.0,1.0], p)
su = false #strong upperbounds
adapt = false
traj, (t, x, θ), (acc, num), c = @time ZZB.sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ,
                                                 tilde_Ψ, tilde_Ψt, Ψ, n;
                                                strong_upperbounds = su ,
                                                adapt = adapt)
