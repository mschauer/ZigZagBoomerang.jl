using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
#using Revise
using ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using FileIO
using Statistics
using Makie, AbstractPlotting
using ForwardDiff
using StructArrays
const ρ0 = 0.0
Σ = [1 ρ; ρ 1]
const d = 2
const dist = 1.5

# patches

const RefZigZag = ZigZag(sparse(I(d)), 0*x0)
ZigZagBoomerang.freezing_time(a, b) = ZigZagBoomerang.freezing_time(a, b, RefZigZag) 

phi(x, y, rho) =  1/(2*pi*sqrt(1-rho^2))*exp(-0.5*(x^2 + y^2 - 2x*y*rho)/(1-rho^2))
logdensity(x, y) = log(phi(x - dist, y - dist, 0.0) + phi(x + dist, y + dist, 0.0))
r = -4.5:0.01:4.5

surf = surface(r, r, [10*exp(logdensity(x1, x2)) for x1 in r, x2 in r])
#=
function partiali()
    ith = zeros(d)
    function (x,i)
        ith[i] = 1
        sa = StructArray{ForwardDiff.Dual{}}((x, ith))
        δ = density(sa).partials[]
        ith[i] = 0
        return δ
    end
end
=#

function gradϕ(x,i) 
    -ForwardDiff.partials(logdensity(ForwardDiff.Dual{}(x[1], 1.0*(i==1)), ForwardDiff.Dual{}(x[2], 1.0*(i==2))))[]
end

function gradϕ!(y, x) 
    y[1] = gradϕ(x,1)
    y[2] = gradϕ(x,2)
    y
end

κ = 0.1*[1.0, 1.0]
t0 = 0.0
x0 = randn(d)
θ0 = [1.0, 1.0]
c = 20.0
T = 1000.0

#sspdmp(∇ϕi, t0, x0, θ0, T, c*ones(d), ZigZag(sparse(I(d)), 0*x0), κ; adapt=false)
trace, (tT, xT, θT), (acc, num) = sspdmp(gradϕ, t0, x0, θ0, T, c*ones(d), ZigZag(sparse(I(d)), 0*x0), κ; adapt=false)
#ts, xs = ZigZagBoomerang.sep(collect(discretize(trace, 0.05)))
ts, xs = ZigZagBoomerang.sep(collect(trace))

p1 = lines(first.(xs), last.(xs), color=ts)

scene, layout = layoutscene(resolution = (1200, 900))
layout[1, 1] = ax1 = Axis(scene)
layout[2, 1] = ax2 = Axis(scene)

linkyaxes!(ax1, ax2)
linkxaxes!(ax1, ax2)

lines!(ax1, ts, last.(xs))
lines!(ax2, ts, last.(xs))
p1a = scene

trace, (tT, xT, θT), (acc, num) = sticky_pdmp(gradϕ!, t0, x0, θ0, T, c, BouncyParticle(sparse(I(d)), 0*x0, 0.1), κ[1]; adapt=false)
#ts, xs = ZigZagBoomerang.sep(collect(discretize(trace, 0.05)))
ts, xs = ZigZagBoomerang.sep(collect(trace))

p2 = lines(first.(xs), last.(xs), color=ts)


scene, layout = layoutscene(resolution = (1200, 900))
layout[1, 1] = ax1 = Axis(scene)
layout[2, 1] = ax2 = Axis(scene)

linkyaxes!(ax1, ax2)
linkxaxes!(ax1, ax2)

lines!(ax1, ts, last.(xs))
lines!(ax2, ts, last.(xs))
p2a = scene

save("zigzagphase.png", p1)
save("zigzagtrace.png", p1a)

save("bouncyphase.png", p2)
save("bouncytrace.png", p2a)
