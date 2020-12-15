##### Packaged loglikelihood
using LinearAlgebra
using Turing
using AdvancedHMC
include("faberschauder.jl")
l = lvl(3, 0)
# π(x) = exp(-ϕ)*C
# ∇ϕ : x → [∇ϕ]_i (exluding initial and final point which are indexed by 1 and (2 << L) +1)
function ∇ϕ(ξ, i, L, T; p = T/(1<<L)) # formula (17)
    if i == (2 << L) + 1    # final point
        error("fixed final point")
    elseif i == 1   # initial point
        error("fixed initial point")
    else
        l = lvl(i, L)
        k = (i - 1) ÷ (2 << l)
        # println("index ($l, $k) out of $L level")
        δ = T/(1 << (L-l)) # T/(2^(L-l))
        r = 0.0
        s = δ*k + p*0.5
        # println("T = $T")
        for i in 1:l+1
            # println("integrating at the point $s")
            x = dotψ(ξ, s, L,  T)
            r += 0.5*p*Λ(s, L-l, T)*(2b(x)*b′(x) + b″(x))
            s += p
            @assert δ*k < s < δ*(k + 1) + p
        end
        return r + ξ[i]
    end
end

# up to constant of proportionality
# π(x) = exp(-ϕ)*C
function ϕ(ξ, L, T; p = T/(1<<L)) # formula (17)
        r = 0.0
        s = p*0.5
        for i in 1:L+1
            x = dotψ(ξ, s, L,  T)
            r += 0.5*p*(b(x)^2* + b′(x))
            s += p
            @assert 0.0 < s < T+p
        end
    return r + 0.5*norm(ξ)^2
end

ϕ(randn(n), L, T)
# Drift
const α = 1.0
b(x) = α * sin(x)
# First derivative
b′(x) = α * cos(x)
# Second derivative
b″(x) = -α * sin(x)

# Drift
const α = 1.5
const L = 7
const T = 50.0
n = (2 << L) + 1
u, v = -π, 3π  # initial and fianl point
#ξ0 = 0randn(n)
#ξ[1] = u / sqrt(T)
#ξ[end] = v / sqrt(T)
T′ = 30000.0 # final clock of the pdmp
D = n - 2

# gradient of loglikelihood
function grad_f(q)
    ξ = [u / sqrt(T); q; v / sqrt(T)]
    return [-∇ϕ(ξ, i, L, T) for i in 2:(2 << L)]
end


# loglikelihood
function logdensity_f(q)
    ξ = [u/sqrt(T); q;  v/sqrt(T)]
    return -ϕ(ξ, K, L, T)
end

#advanced HMC
using AdvancedHMC
n_samples, n_adapts, target = 10_000, 2_000, 0.8
q0 = randn(D)
metric = DiagEuclideanMetric(D)
h = Hamiltonian(metric, logdensity_f, grad_f)
eps_init = find_good_stepsize(h, q0)

int = Leapfrog(eps_init)
traj = NUTS{Multinomial,GeneralisedNoUTurn}(int)
adaptor = StanHMCAdaptor(
    n_adapts, Preconditioner(metric), NesterovDualAveraging(target, eps_init)
    )
samples, stats = sample(h, traj, q0, n_samples, adaptor, n_adapts)
