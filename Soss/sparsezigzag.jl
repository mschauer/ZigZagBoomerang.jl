
using Soss: logdensity, xform, ConditionalModel
using ZigZagBoomerang
using ForwardDiff
using ForwardDiff: gradient!
using LinearAlgebra
using SparseArrays
using StructArrays
using TransformVariables


kappa(m::SpikeMixture{WeightedMeasure{Float64,Lebesgue{ℝ}},Float64}) = exp(m.m.logweight)/(1/m.w-1)
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

