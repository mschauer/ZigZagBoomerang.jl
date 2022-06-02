# Using the ZigZagBoomerang with Turing with the BouncyParticle sampler
# (The approach taken here is retrieving the likelihood function from Turing and sampling
# directly with ZigZagBoomerang and not using Turings `AbstractMCMC` )

using Revise
using Soss
using ZigZagBoomerang 
using StatsFuns
const ZZB = ZigZagBoomerang 
using LinearAlgebra
const ∅ = nothing
using DelimitedFiles
using Random
Random.seed!(1)

include("../turing/plot_chain.jl") # simple visualization of chains with GLMakie
# read data
function readlrdata()
    fname = joinpath(dirname(@__FILE__), "lr.data")
    z = readdlm(fname)
    x = z[:,1:end-1]
    x = [ones(size(x,1)) x]
    y = z[:,end] .- 1
    return x, y
end
x, y = readlrdata()


# following https://github.com/TuringLang/Turing.jl/blob/master/benchmarks/nuts/lr.jl
m = @model (x, σ) begin
    θ ~ Normal(σ=σ)^size(x,2)
    y ~ For(eachcol(x)) do xj
        Bernoulli(logitp = dot(xj, θ))
    end
end
const σ = 100.0
x0 = randn(size(x,2))
logdensityof(m(x, σ) | (;y), (;θ=x0))

#rand(m(x,σ))
# define problem




# sample with ZigZagBoomerang

using ForwardDiff
using ForwardDiff: Dual, value
"""
    make_gradient_and_dhessian_logp(turingmodel) -> ∇nlogp!

Gradient of negative log-likelihood and second derivative in direction of movement 

Following https://github.com/TuringLang/Turing.jl/blob/master/src/core/ad.jl
"""
function make_derivatives_neglogp(m, x, y, σ)
    obj(θ) = -logdensityof(m(x, σ) | (;y), (;θ))

    f1 = function (t, x, v, args...) # two directional derivatives
        u = ForwardDiff.derivative(t -> obj(x + t*v), Dual{:hSrkahPmmC}(0.0, 1.0))
        u.value, u.partials[]
    end
    f2 = function (y, t, x, args...)
        ForwardDiff.gradient!(y, obj, x)
        y
    end
    return f1, f2, init, trans
end



dneglogp, ∇neglogp!, init, trans = make_derivatives_neglogp(model);

d = 1 + 24 # number of parameters 
t0 = 0.0
x0 = zeros(d) # starting point sampler
T1 = 100.0 # adapt mass matrix after 100
T = 500. # end time (similar to number of samples in MCMC)
c = 5.0 # initial guess for the bound
M = I
#M = Diagonal(1 ./ [1.7, 0.08, 0.01, 0.09, 0.01, 0.06, 0.08, 0.12, 0.09, 0.11, 0.01, 0.11, 0.18, 0.29, 0.21, 0.88, 0.21, 0.39, 0.44, 0.65, 0.4, 0.35, 0.6, 0.31, 0.3])
θ0 = M\randn(d) # starting direction sampler

# define BouncyParticle sampler (has two relevant parameters) 
Z = BouncyParticle(∅, ∅, # ignored
    2.0, # momentum refreshment rate 
    0.95, # momentum correlation / only gradually change momentum in refreshment/momentum update
    0.0, # ignored
    M # cholesky of momentum precision
) 

trace, final, (acc, num), cs = @time pdmp(
        dneglogp, # return first two directional derivatives of negative target log-likelihood in direction v
        ∇neglogp!, # return gradient of negative target log-likelihood
        t0, x0, θ0, T1, # initial state and duration
        ZZB.LocalBound(c), # use Hessian information 
        Z; # sampler
        adapt=true, # adapt bound c
        progress=true, # show progress bar
        subsample=true # keep only samples at refreshment times
)


t, x = ZigZagBoomerang.sep(trace)

# plot bouncy particle sampler
fig3 = plot_chain(t, x, false, color=:green)
save("lrbouncy.png", fig3)


