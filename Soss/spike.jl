using Soss
using Soss: logdensity, xform, ConditionalModel
using ZigZagBoomerang
#using Makie
### Define the target distribution and its gradient
using ForwardDiff
using ForwardDiff: gradient!
using LinearAlgebra
using SparseArrays
using StructArrays
using TransformVariables

using MeasureTheory

kappa(m::SpikeMixture{WeightedMeasure{Float64,Lebesgue{ℝ}},Float64}) = m.w̄/exp(u.m.logweight)
kappa(m::WeightedMeasure{Float64,Lebesgue{ℝ}}) = Inf
Soss.xform(m::SpikeMixture) = Soss.xform(m.m)

function sparse_zigzag(m::ConditionalModel, T = 1000.0; c=10.0, adapt=false)



    ℓ(pars) = logdensity(m, pars)

    t = xform(m)

    function f(x)
        (θ, logjac) = TransformVariables.transform_and_logjac(t, x)
        -ℓ(θ) - logjac
    end

    d = t.dimension

    function partiali()
        ith = zeros(d)
        function (x,i)
            ith[i] = 1
            sa = StructArray{ForwardDiff.Dual{}}((x, ith))
            δ = f(sa).partials[]
            ith[i] = 0
            return δ
        end
    end

    ∇ϕi = partiali()

    # Draw a random starting points and velocity
    tkeys = keys(t(zeros(d)))
    vars = Soss.select(rand(m), tkeys)

    bm_ = Soss.basemeasure(m, vars)
    bm = getindex.(Ref(bm_.data), (keys(vars)))
    κ = kappa.(bm)
    
    t0 = 0.0
    x0 = inverse(t, vars)
    θ0 = randn(d)
    
    sspdmp(∇ϕi, t0, x0, θ0, T, c*ones(d), ZigZag(sparse(I(d)), 0*x0), κ; adapt=adapt)
end



m = @model x begin
    α ~ Normal()
    β ~ Normal()
    yhat = α .+ β .* x
    y ~ For(eachindex(x)) do j
        Normal(yhat[j], 2.0)
    end
end


m = @model x begin
    α ~ SpikeMixture(Normal(), 0.2) # 0.2*Normal() + 0.8*Dirac(0)
    β ~ SpikeMixture(Normal(), 0.2)
    yhat = α .+ β .* x
    y ~ For(eachindex(x)) do j
        Normal(yhat[j], 2.0)
    end
end


x = randn(20);
obs = -0.1 .+ 2x + 1randn(20); 
T = 100.0
posterior = m(x=x) | (y=obs,)

trace, final, (num, acc) = @time sparse_zigzag(posterior, T, c=50)

ts, xs = ZigZagBoomerang.sep(discretize(trace, 0.1))

xs = xform(posterior).(xs)

using Plots
p = plot(ts, getindex.(xs, 1))
plot!(ts, getindex.(xs, 2), color=:red)
p2 = plot(first.(xs), last.(xs))
p
