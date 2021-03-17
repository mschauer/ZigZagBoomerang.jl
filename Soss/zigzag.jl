
using Soss: logdensity, xform, ConditionalModel
using ZigZagBoomerang
using ForwardDiff
using ForwardDiff: gradient!
using LinearAlgebra
using SparseArrays
using StructArrays
using TransformVariables

Soss.xform(m::SpikeMixture) = Soss.xform(m.m)
function zigzag(m::ConditionalModel, T = 1000.0; c=10.0, adapt=false)

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

    t0 = 0.0
    x0 = inverse(t, vars)
    θ0 = randn(d)
    
    pdmp(∇ϕi, t0, x0, θ0, T, c*ones(d), ZigZag(sparse(I(d)), 0*x0); adapt=adapt)
end

