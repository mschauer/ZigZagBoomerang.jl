using Base: sign_mask
using ZigZagBoomerang, Distributions, ForwardDiff, LinearAlgebra, SparseArrays, StructArrays
using StatsBase, JLD2
const ZZB = ZigZagBoomerang
using Test, Random
using BenchmarkTools
using ForwardDiff: Dual, value

StatsBase.cov2cor(C) = cov2cor(C, sqrt.(diag(C)))
cd(@__DIR__)
outer(x) = x*x'
d = 5
Random.seed!(3)
μ1 = 0.5randn(d)
μ2 = 0.5randn(d)
Σ1 = 0.1*outer(randn(d,d))
Σ2 = 0.2*outer(randn(d,d))

ℓ(x, P)  = log(0.5pdf(MvNormal(P.μ1, P.Σ1), x)+0.5pdf(MvNormal(P.μ2, P.Σ2), x))# + randn()
#ℓ(x, P)  = logpdf(MvNormal(P.μ1, P.Σ1), x)

function negpartiali(f, d)
    id = collect(I(d))
    ith = [id[:,i] for i in 1:d]
    function (x, i, args...)
        sa = StructArray{Dual{}}((x, ith[i]))
        δ = -f(sa, args...).partials[]
        return δ
    end
end

function gradientsi(f, d)
    id = collect(1.0*I(d))
    ith = [id[:,i] for i in 1:d]
    z = zeros(d)
    function (t, x, θ, i, t′, F, args...)
       # sa =  (reinterpret(Dual{:Sion, Dual{:Confu, Float64, 1}, 1}, StructVector{Tuple{Float64, Float64, Float64, Float64}}((x, θ, ith[i], z))))
        sa = reinterpret(Dual{:Sion, Dual{:Confu, Float64, 1}, 1}, collect(zip(x, θ, ith[i], z)))
        δ = -f(sa, args...)
        if isnan(δ.partials[].value)
            @show t, x, θ, i, t′
            error()
        end
        return δ.partials[].value, (δ.partials[].partials[])*θ[i]
    end
end
∇ϕi = negpartiali(ℓ, d)
∇ϕi2 = gradientsi(ℓ, d)
function gradients(ℓ, d)
    function (y, t, x, θ, args...)
        x_ = x + Dual{:Bz0iEasFfw}(0.0, 1.0)*θ
        y_ = ForwardDiff.gradient(x->-ℓ(x, args...), x_)
        y .= value.(y_)
        y, dot(θ, y_).partials[]
    end
end

∇ϕ2! = gradients(ℓ, d)
μ = μ1 #+ μ2

t0 = 0.0
x0 = 1.01μ
θ0 = 1.0*ones(d)
c = .1
 
T = 2000.0
#Γ = sparse(1.0I(d))
Γ = sparse(inv(Σ1))

P = (;μ1=μ1, μ2=μ2, Σ1=Σ1 , Σ2=Σ2)
#P = (;μ1=μ1, μ2=μ1, Σ1=Σ1 , Σ2=Σ1)

#=
t = fill(t0, d)
x = x0
θ = θ0
i = 3
t′ = t0

∇ϕi_, vi = ∇ϕi2(t, x, θ, i, t′, nothing, P)
∇ϕi(x, i, P), θ[i]'*ZZB.idot(sparse(inv(Σ1)), i, θ)
∅ = nothing
a = ZZB.ab(∅, i, x, θ, ZZB.LocalBound(0*ones(d)), ∇ϕi_, vi, ZigZag(Γ, μ))
a = ZZB.ab(∅, i, x, θ, ZZB.LocalBound(c*ones(d)), ∇ϕi_, vi, ZigZag(Γ, μ))
ZZB.next_time(t[i], a, rand())

#∇ϕi2([9.058005597805279, 9.058005597805279, 9.058005597805279, 9.058005597805279, 9.058005597805279], [-6.787367020095527, -9.699425733413173, -6.783498890630347, -9.056495876472626, -7.964177763780483], [-1.0, -1.0, -1.0, -1.0, -1.0], 1, 9.058005597805279, 0, P)
=#

if false
    Z = ZigZag(Γ, μ)
    #Z = ZigZag(∅, ∅, ∅, 0.0, 0.0, 1.0)
    trace, final, (acc, num), cs = pdmp(∇ϕi2, t0, x0, θ0, T, ZZB.LocalBound(c*ones(d)), 
        Z, P; adapt=true, progress=true)
else
    Z = BouncyParticle(Γ, μ, 1.0, 0.8, cholesky(Symmetric(Γ0)).U )
    trace, final, (acc, num), cs = pdmp(∇ϕ2!, t0, x0, θ0, T, ZZB.LocalBound(c), 
        Z, P ; adapt=true, progress=true)
end
cs isa Vector ? @show(summarystats(cs)) : @show(cs)
t, x = ZigZagBoomerang.sep(trace)
r = eachindex(t)[end-100:end-1]
r = eachindex(t)[2:end-1]
cx = cummean(trace)

using GLMakie
if false
fig2 = Figure(resolution=(2000,2000))
e = 4
for i in 1:min(d, e^2)
    u = CartesianIndices((e,e))[i]
    lines(fig2[u[1],u[2]], t[r], getindex.(x, i)[r])
    if cs isa Vector
        lines!(fig2[u[1],u[2]], (cx[i].t), (cx[i].y), linewidth=3.0, color=:orange) 
    end
end
display(fig2)
end
if true
    fig3 = Figure(resolution=(2000,2000))
    e = d
    for i in 1:min(d^2, e^2)
        u = CartesianIndices((e,e))[i]
        if u[1] == u[2]
            lines(fig3[u[1],u[2]], t[r], getindex.(x, u[1])[r])
            lines!(fig3[u[1],u[2]], t[r], fill(μ1[u[1]], length(r)), color=:green)
            lines!(fig3[u[1],u[2]], t[r], fill(μ2[u[1]], length(r)), color=:green)
            
        else
            lines(fig3[u[1],u[2]], getindex.(x, u[1])[r],  getindex.(x, u[2])[r])
            scatter!(fig3[u[1],u[2]], [
                    μ1[u[1]]
                    μ1[u[1]] 
                    μ2[u[1]] 
                    μ2[u[1]]
                    ],[
                    μ1[u[2]] 
                    μ2[u[2]]
                    μ1[u[2]]
                    μ2[u[2]]
                    ], markersize=5.0, color=:green)    
        end
    end
    display(fig3)
end