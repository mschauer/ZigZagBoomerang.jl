using Turing
using ZigZagBoomerang 
const ZZB = ZigZagBoomerang 
using LinearAlgebra
const ∅ = nothing
using DelimitedFiles

using GLMakie
function plot_chain(t, x, FULL=false; skip=0, color=:black)
    r = eachindex(t)[1:1+skip:end]
    fig3 = Figure(resolution=(2000,2000))
    e = 6
    for i in 1:e^2
        u = CartesianIndices((e,e))[i]
        if u[1] == u[2]
            FULL && GLMakie.lines(fig3[u[1],u[2]], t, getindex.(x, u[1]),  color=color)
            !FULL && GLMakie.scatter(fig3[u[1],u[2]], t[r], getindex.(x, u[1])[r], markersize=0.5,  color=color)
        elseif u[1] < u[2]
            FULL && GLMakie.lines(fig3[u[1],u[2]], getindex.(x, u[1]),  getindex.(x, u[2]), linewidth=0.5, color=(color,1.0))
            !FULL && GLMakie.scatter(fig3[u[1],u[2]], getindex.(x, u[1])[r],  getindex.(x, u[2])[r], linewidth=0.5, color=(color,1.0), markersize= 0.5, strokewidth=0)
        end
    end
    fig3
end
function plot_chain!(fig3, t, x, FULL=false; color=:black, skip=0)
    r = eachindex(t)[1:1+skip:end]
    e = 6
    for i in 1:e^2
        u = CartesianIndices((e,e))[i]
        if u[1] == u[2]
            FULL && GLMakie.lines!(fig3[u[1],u[2]], t, getindex.(x, u[1]), color=color)
            !FULL && GLMakie.scatter!(fig3[u[1],u[2]], t[r], getindex.(x, u[1])[r], markersize=0.5, color=color)
        elseif u[1] < u[2]
            FULL && GLMakie.lines!(fig3[u[1],u[2]], getindex.(x, u[1]),  getindex.(x, u[2]), linewidth=0.5, color=(color,1.0))
            !FULL && GLMakie.scatter!(fig3[u[1],u[2]], getindex.(x, u[1])[r],  getindex.(x, u[2])[r], linewidth=0.5, color=(color,1.0), markersize= 0.5, strokewidth=0)
        end
    end
    fig3
end

function readlrdata()

    fname = joinpath(dirname(@__FILE__), "lr_nuts.data")
    z = readdlm(fname)
    x = z[:,1:end-1]
    y = z[:,end] .- 1
    return x, y
end

x, y = readlrdata()
# following https://github.com/TuringLang/Turing.jl/blob/master/benchmarks/nuts/lr.jl
@model lr_nuts(x, y, σ) = begin

    N,D = size(x)

    α ~ Normal(0, σ)
    β ~ MvNormal(zeros(D), ones(D)*σ)

    for n = 1:N
        y[n] ~ BinomialLogit(1, dot(x[n,:], β) + α)
    end
end
model = lr_nuts(x, y, 100.0)
if true

    # Sampling parameter settings
    n_samples = 1_000

    # Sampling # 440 s
    nuts_chain = @time sample(model, NUTS(0.65), n_samples)

    using GLMakie
    fig2 = plot_chain(1:n_samples, collect(eachrow(dropdims(nuts_chain[nuts_chain.name_map.parameters].value.data, dims=3)) ))
    save("lrnuts.png", fig2)
end

using ForwardDiff
using ForwardDiff: Dual, value
"""
Gradient of negative log-likelihood and second derivative in direction of movement 

Following https://github.com/TuringLang/Turing.jl/blob/master/src/core/ad.jl
"""
function make_gradient_and_dhessian_logp(
    model::Turing.Model,
    sampler=Turing.SampleFromPrior(),
    ctx::Turing.DynamicPPL.AbstractContext = DynamicPPL.DefaultContext()
)
    vi = Turing.VarInfo(model)

    # Define function to compute log joint.
    function ℓ(θ)
        new_vi = Turing.VarInfo(vi, sampler, θ)
        model(new_vi, sampler, ctx)
        logp = Turing.getlogp(new_vi)
        return logp
    end

    return function (y, t, x, θ, args...)
        x_ = x + Dual{:hSrkahPmmC}(0.0, 1.0)*θ
        y_ = ForwardDiff.gradient(x->-ℓ(x), x_)
        y .= value.(y_)
        y, dot(θ, y_).partials[]
    end
end


∇ϕ2! = make_gradient_and_dhessian_logp(model)

d = 1 + 24
t0 = 0.0
x0 = zeros(d) 
θ0 = randn(d)
T = 200.
c = 50.0

sampler = "bouncy" # 299 s
Z = BouncyParticle(∅, ∅, 10.0, 0.95, 0.0) 
trace, final, (acc, num), cs = @time pdmp(∇ϕ2!, t0, x0, θ0, T, ZZB.LocalBound(c), 
        Z ; adapt=true, progress=true, subsample=true)
t, x = ZigZagBoomerang.sep(trace)

if true
    fig3 = plot_chain(t, x, false)
    save("lrbouncy.png", fig3)
end

# check 
# lines(mean(trace))
# lines!(mean(nuts_chain).nt[:mean])
fig4 = plot_chain(1:n_samples, collect(eachrow(dropdims(nuts_chain[nuts_chain.name_map.parameters].value.data, dims=3)) ), color=:red)
fig4 = plot_chain!(fig4, t*n_samples/T, x, false, color=:green)
save("lrboth.png", fig4)