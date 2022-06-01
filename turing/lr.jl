# Using the ZigZagBoomerang with Turing with the BouncyParticle sampler
# (The approach taken here is retrieving the likelihood function from Turing and sampling
# directly with ZigZagBoomerang and not using Turings `AbstractMCMC` )

using Revise
using Turing
using ZigZagBoomerang 
const ZZB = ZigZagBoomerang 
using LinearAlgebra
const ∅ = nothing
using DelimitedFiles

include("plot_chain.jl") # simple visualization of chains with GLMakie

# define Turing Logit regression model 
# following https://github.com/TuringLang/Turing.jl/blob/master/benchmarks/nuts/lr.jl
@model lr_nuts(x, y, σ) = begin

    N,D = size(x)

    α ~ Normal(0, σ)
    β ~ MvNormal(zeros(D), ones(D)*σ)

    for n = 1:N
        y[n] ~ BinomialLogit(1, dot(x[n,:], β) + α)
    end
end

# read data
function readlrdata()
    fname = joinpath(dirname(@__FILE__), "lr_nuts.data")
    z = readdlm(fname)
    x = z[:,1:end-1]
    y = z[:,end] .- 1
    return x, y
end
x, y = readlrdata()

# define problem
model = lr_nuts(x, y, 100.0)


# sample First with Turing and Nuts


n_samples = 1_000 # Sampling parameter settings
nuts_chain = @time sample(model, NUTS(0.62), n_samples) # (a bit frickle, sometimes adapts wrong, ϵ = 0.1 seems good)
# sampling took 383 s (ϵ = 0.1) or 768 s (ϵ = 0.05)

# plot NUTS
fig2 = plot_chain(1:n_samples, collect(eachrow(dropdims(nuts_chain[nuts_chain.name_map.parameters].value.data, dims=3)) ))
save("lrnuts.png", fig2)


# sample with ZigZagBoomerang

using ForwardDiff
using ForwardDiff: Dual, value
"""
    make_derivatives_neglogp(turingmodel) -> dneglogp, ∇neglogp!

Make two functions `dneglogp`, `∇neglogp!` which compute

1.) `dneglogp(_, x, v)` computes for `f(t) = -log(p(x + t*v))` the first directional derivative `f'(0)` of negative target log-likelihood `p` at `x` 
in direction `v` and the corresponding second derivative `f''(0)`.

2.) `∇neglogp!(y, _, x)` computes the gradient of negative target log-likelihood `log(p(x))` at `x` inplace.
            
Following https://github.com/TuringLang/Turing.jl/blob/master/src/core/ad.jl
"""
function make_derivatives_neglogp(
    model::Turing.Model,
    sampler=Turing.SampleFromPrior(),
    ctx::Turing.DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    vi = Turing.VarInfo(model)

    # define function to compute log joint.
    function ℓ(θ)
        new_vi = Turing.VarInfo(vi, sampler, θ)
        model(new_vi, sampler, ctx)
        logp = Turing.getlogp(new_vi)
        return logp
    end
    f1 = function (t, x, v, args...) # two directional derivatives
        u = ForwardDiff.derivative(t -> -ℓ(x + t*v), Dual{:hSrkahPmmC}(0.0, 1.0))
        u.value, u.partials[]
    end
    f2 = function (y, t, x, args...)
        ForwardDiff.gradient!(y, ℓ, x)
        y .= -y
        y
    end
    return f1, f2
end


dneglogp, ∇neglogp! = make_derivatives_neglogp(model)

d = 1 + 24 # number of parameters 
t0 = 0.0
x0 = zeros(d) # starting point sampler
θ0 = randn(d) # starting direction sampler
T = 200. # end time (similar to number of samples in MCMC)
c = 50.0 # initial guess for the bound

# define BouncyParticle sampler (has two relevant parameters) 
Z = BouncyParticle(∅, ∅, # ignored
    10.0, # momentum refreshment rate 
    0.95, # momentum correlation / only gradually change momentum in refreshment/momentum update
    0.0, # ignored
    I # cholesky of momentum precision
) 

trace, final, (acc, num), cs = @time pdmp(
        dneglogp, # return first two directional derivatives of negative target log-likelihood in direction v
        ∇neglogp!, # return gradient of negative target log-likelihood
        t0, x0, θ0, T, # initial state and duration
        ZZB.LocalBound(c), # use Hessian information 
        Z; # sampler
        oscn=false, # no orthogonal subspace pCR
        adapt=true, # adapt bound c
        progress=true, # show progress bar
        subsample=true # keep only samples at refreshment times
)
# took 101 s

# obtain direction change times and points of piecewise linear trace
t, x = ZigZagBoomerang.sep(trace)


# plot bouncy particle sampler
fig3 = plot_chain(t, x, false)
save("lrbouncy.png", fig3)

# check visually 
# lines(mean(trace))
# lines!(mean(nuts_chain).nt[:mean])

# show both in one plot
fig4 = plot_chain(1:n_samples, collect(eachrow(dropdims(nuts_chain[nuts_chain.name_map.parameters].value.data, dims=3)) ), 
    color=:red, title="Green: Bouncy Particle. Red: NUTS.")
fig4 = plot_chain!(fig4, t*n_samples/T, x, false, color=:green)
save("lrboth.png", fig4) 
