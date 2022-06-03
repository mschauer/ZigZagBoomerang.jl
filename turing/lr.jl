# Using the ZigZagBoomerang with Turing with the BouncyParticle sampler
# (The approach taken here is retrieving the likelihood function from Turing and sampling
# directly with ZigZagBoomerang and not using Turings `AbstractMCMC` )

using Revise
using Turing
using ZigZagBoomerang 
using Pathfinder
const ZZB = ZigZagBoomerang 
using LinearAlgebra
const ∅ = nothing
using DelimitedFiles
using Random
using MCMCChains
Random.seed!(1)

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
if false
n_samples = 1_000 # Sampling parameter settings
nuts_chain = @time sample(model, NUTS(0.65), n_samples) # (a bit frickle, sometimes adapts wrong, ϵ = 0.1 seems good)
# sampling took 383 s (ϵ = 0.1) or 768 s (ϵ = 0.05)

# plot NUTS
fig2 = plot_chain(1:n_samples, collect(eachrow(dropdims(nuts_chain[nuts_chain.name_map.parameters].value.data, dims=3)) ))
save("lrnuts.png", fig2)
end

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
    obj_, init, trans = optim_objective(model, MAP(); constrained=false)
    obj = obj_ # ∘trans
#    @show obj(x0) ℓ(x0)

    f1 = function (t, x, v, args...) # two directional derivatives
        u = ForwardDiff.derivative(t -> obj(x + t*v), Dual{:hSrkahPmmC}(0.0, 1.0))
        u.value, u.partials[]
    end
    f2 = function (y, t, x, args...)
        ForwardDiff.gradient!(y, obj, x)
        y
    end
    return obj, f1, f2, init, trans
end



neglogp, dneglogp, ∇neglogp!, init, trans = make_derivatives_neglogp(model);

d = 1 + 24 # number of parameters 
init_scale = 4.0
@time result = pathfinder(x->-neglogp(x); dim=d, init_scale)

t0 = 0.0
T = 600. # end time (similar to number of samples in MCMC)
c = 5.0 # initial guess for the bound
M = Diagonal(1 ./ sqrt.(diag(result.fit_distribution.Σ)))
#M = Diagonal(1 ./ [1.7, 0.08, 0.01, 0.09, 0.01, 0.06, 0.08, 0.12, 0.09, 0.11, 0.01, 0.11, 0.18, 0.29, 0.21, 0.88, 0.21, 0.39, 0.44, 0.65, 0.4, 0.35, 0.6, 0.31, 0.3])
#x0 = zeros(d) # starting point sampler
x0 = result.fit_distribution.μ
θ0 = M\randn(d) # starting direction sampler

# define BouncyParticle sampler (has two relevant parameters) 
Z = BouncyParticle(∅, ∅, # ignored
    2.0, # momentum refreshment rate 
    0.97, # momentum correlation / only gradually change momentum in refreshment/momentum update
    0.0, # ignored
    M # cholesky of momentum precision
) 

el = @elapsed begin

trace, final, (acc, num), cs = @time pdmp(
        dneglogp, # return first two directional derivatives of negative target log-likelihood in direction v
        ∇neglogp!, # return gradient of negative target log-likelihood
        t0, x0, θ0, T, # initial state and duration
        ZZB.LocalBound(c), # use Hessian information 
        Z; # sampler
        adapt=true, # adapt bound c
        progress=true, # show progress bar
        subsample=true # keep only samples at refreshment times
)
end


# obtain direction change times and points of piecewise linear trace
t, x = ZigZagBoomerang.sep(trace)

#x = trans.(x)
bps_chain = MCMCChains.Chains([xj[i] for xj in x[end÷4:end], i in 1:d])
bps_chain = setinfo(bps_chain,  (;start_time=0.0, stop_time = el))
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
