using Revise
import ZigZagBoomerang: PDMPTrace, discretize, poisson_time # poisson_time(a, b, u), poisson_time((a, b, c), u)
import ZigZagBoomerang
using Random
const ZZB = ZigZagBoomerang

struct Linear
end

normsq(q) = dot(q, q)

"""
    oscn!(rng, v, ∇ψx, ρ; normalize=false)

Orthogonal subspace Crank-Nicolson step with autocorrelation `ρ` for
standard Gaussian or Uniform on the sphere (`normalize = true`).
"""
function oscn!(rng, v, ∇ψx, ρ; normalize=false)
    # Decompose v
    vₚ = (dot(v, ∇ψx)/normsq(∇ψx))*∇ψx
    v⊥ = ρ*(v - vₚ)
    if ρ == 1
        @. v = v - 2vₚ 
    else
        # Sample and project
        z = randn!(rng, similar(v)) * √(1.0f0 - ρ^2)
        z -= (dot(z, ∇ψx)/dot(∇ψx, ∇ψx))*∇ψx
        if normalize
            λ = sqrt(1 - norm(vₚ)^2)/norm(v⊥ + z)
            @. v = -vₚ + λ*(v⊥ + z)
        else
            @. v = -vₚ + v⊥ + z
        end
    end
    v
end
function Qrefresh!(rng, u, ρ=0.95)
    t, x, v = u
    v .= ρ*v + sqrt(1-ρ^2)*randn(rng, length(v))/sqrt(F(normsq(x - μₓ)/2))
    v
end


function Qbounce!(rng, u, ∇ψx)
    t, x, v = u
    #oscn!(rng, v, ∇ψx, 0.99)
    oscn!(rng, v, ∇ψx, 1.0)
end
slope(x) = x[1][]
grad(x) = x[2][]
bound(x) = x[3][]


function subpdmp!(rng, Ξ, Ψ!, u, J, Δ, flow, Qbounce!, sgb)
    while true
        τ, action = next_event(rng, u, bound(sgb)) # split up for factorized?
        τ > Δ && return false 
        move_forward!(flow, J, τ, u)
        if action == :expire
            Ψ!(rng, u, flow, sgb)
            continue
        elseif action == :bounce
            Ψ!(rng, u, flow, sgb)
            a = max(0, slope(sgb))/λ̄(τ, bound(sgb))
            a > 1.0 && @warn("bound")
            if rand(rng) <= a
                Qbounce!(rng, u, grad(sgb))
                Ψ!(rng, u, flow, sgb)
            else
                continue
            end
        elseif action == :velobounce
            error("todo")
        elseif action == :refresh
            Qrefresh!(rng, u)
            Ψ!(rng, u, flow, sgb)    
        end
        push!(Ξ, (copy(u[1]), copy(u[2]), copy(u[3])))
        return true
    end
end

function bouncy(rng, Ψ!, u0, J, T, flow, Qbounce!)
    Ξ = [deepcopy(u0)]
    u = deepcopy(u0)
    sgb = Ψ!(rng, u, flow)    
    changed = true
    while changed
         changed = subpdmp!(rng, Ξ, Ψ!, u, J, T, flow, Qbounce!, sgb)
         #@show u
    end
    return Ξ, u
end
function next_event(rng, u, ab)
    told, a, b, c, Δ = ab
    t, x, v = u
    @assert reduce(==, t)
    @assert told == t[1]

    # next event time
    τ = t[1] + poisson_time((a, b, c), rand(rng))
#    τrefresh = t[1] + Inf
    τrefresh = t[1] + 0.1*randexp(rng)

    # next event
    when, what = findmin((τ, Δ, τrefresh))
    return when, (:bounce, :expire, :refresh)[what]
end

function simplebound(rng, u, a)
    t, x, v = u
    @assert reduce(==, t)
    c = 0.1
    b = 0.0
    Δ = t[1] + 1.0
    (t[1], a, b, c, Δ)
end


λ̄(t, (told, a, b, c, Δ)) = max(a + b*(t - told), 0) + c
function move_forward!(flow::Linear, ::typeof(:), τ, u)
   t, x, v = u
   @assert all(τ .>= t) 
   @. x = x + v*(τ - t)
   @. t = τ
   return u
end
using LinearAlgebra
using ForwardDiff
const μₓ = [1/3, 2/3]
const ς = 2.0
ψ(x) = normsq(x - μₓ)/2
#F(x) = 1.0
#F(x) = 1 + sqrt(x)
#F′(x) = 0.5*x^(-0.5)

#  +normsq(v)/2*F(normsq(x - μₓ)/2) - log(F(x))
#F(x) = 1/(2.0 + 10*x^(ς))
#F′(x) = 10*ς*x^(ς - 1)
choice = 2
#F(x) = exp(-1.2x)
if choice == 1
    F(x) = exp(-0.3x)
elseif choice == 2
    F(x) = exp(-0.5x)
elseif choice == 3 
    F(x) = 1.0
end
#F′(x) = ς*x*exp(ς*x)

#F(x) = 1.0
#F′(x) = 0.0
F2(x) = -0.5*2*log(F(x))
F′(x) = ForwardDiff.derivative(F, x)
F2′(x) = ForwardDiff.derivative(F2, x)


function Ψ!(rng, u, flow, (slope, grad, bound)=([0.0], [zero(u[3])], [simplebound(rng, u, 500.0)]))  
    t, x, v = u
    # grad[] .= x .- μₓ #ForwardDiff.gradient(ψ, x)
    c = normsq(v)/2*F′(normsq(x - μₓ)/2)*(x - μₓ) + F2′(normsq(x - μₓ)/2)*(x - μₓ) 
    @. grad[] = x - μₓ + c
    slope[] = dot(v, grad[])
    bound[] = simplebound(rng, u, 2000.0)
    slope, grad, bound
end
sep(Ξ) = map(x->x[1][1], Ξ), map(x->x[2], Ξ)


if choice == 1
    T = 600.0
    x0 = [3.0f0, 0.6f0]
elseif choice in (2, 3)
    T = 2000.0
    x0 .= 1μₓ + 0.02randn(2)
end
v0 = -0.5normalize(ForwardDiff.gradient(ψ, x0))/sqrt(F(normsq(x0-μₓ)/2))
u0 = ([0.0, 0.0], x0, v0)
rng =  Random.default_rng()
Qrefresh!(rng, u0, 0.0)

bouncy(rng, Ψ!, u0, :, T, Linear(), Qbounce!)
Ξ, u = @time bouncy(rng, Ψ!, u0, :, T, Linear(), Qbounce!)
#BP = ZZB.BouncyParticle(Matrix(1.0*I(2)), μₓ, 0.0)
BP = Linear()

function ZZB.smove_forward!(dt, t, x, v, _, ::Linear)
    @. x = x + v*dt 
    return t + dt, x, v
end

Ξ2 = deepcopy(Ξ)
trc = ZigZagBoomerang.PDMPTrace(BP, u0[1][1], copy(u0[2]), copy(u0[3]), ones(Bool, length(μₓ)), map(ξ->(ξ[1][1], copy(ξ[2]), copy(ξ[3]), nothing), Ξ[2:end]))
ts2, xs2 = sep(Ξ2)
ts, xs = ZZB.sep(ZZB.discretize(trc, 0.05))

using GLMakie
fig = fig1 = Figure(fontsize = 30,  resolution = (1200, 900))
ax = Axis(fig[1,1],  xlabel = "t", ylabel = "x₁")
lines!(ax, ts, getindex.(xs, 1))
ax = Axis(fig[2,1],  xlabel = "t", ylabel = "x₂")
lines!(ax, ts, getindex.(xs, 2))
ax = Axis(fig[1:2, 2],  xlabel = "x₁", ylabel = "x₂")
scatter!(ax, getindex.(xs2, 1),  getindex.(xs2, 2), markersize=2)
#r1 = (1:100) .+ 8500
r1 = (1:500) .+ 5500

scatter!(ax, getindex.(xs2[r1], 1),  getindex.(xs2[r1], 2), color=:darkblue, markersize=5)
lines!(ax, getindex.(xs2[r1], 1),  getindex.(xs2[r1], 2), linewidth=0.5, color=:darkblue)


fig
save("vr/fig$(choice).png", fig)


using GLMakie
fig = fig4 = Figure(fontsize = 30,  resolution = (1200, 900))
ax = Axis(fig[1, 1],  xlabel = "x₁", ylabel = "x₂")
scatter!(ax, getindex.(xs2, 1),  getindex.(xs2, 2), markersize=2)
fig
save("vr/comp$(choice).png", fig)



fig2 = Figure()

ax = Axis(fig2[1,1])

scatter!(ax, getindex.(xs, 1), getindex.(xs, 2), markersize=1.)
if false
    lines!(getindex.(xs2, 1), getindex.(xs2, 2), color=norm.(getindex.(Ξ, 3)), linewidth=0.4)
end
scatter!(ax, [μₓ[1]], [μₓ[2]])
scatter!(ax, [x0[1]], [x0[2]])


#using ProfileView
#@code_warntype bouncy(rng, Ψ!, u0, :, 1000. , Linear(), Qbounce!)

# Target

ψ(x) = normsq(x - μₓ)/2
#ρ(v, x) = normsq(v-f(x))/2
ρ(v, x) = normsq(v)/2*F(normsq(x - μₓ)/2)
#   ∇ₓρ = normsq(v)/2*F′(normsq(x - μₓ)/2))*(x - μₓ)
# ψ(x) == const are circles

#=
function Qvreflect!(rng, u, _)
    t, x, v = u
    ∇ₓρ = normsq(v)/2*(x - μₓ) 
    v .= v - 2dot(v, ∇ₓρ)/normsq(∇ₓρ)*∇ₓρ
    v
end
=#
u0
function nlz!(x)
    l, u = extrema(x)
    @. x = (x - l)/(u-l)
    x
end

using Colors
using Statistics

@show mean(xs)  μₓ
@show cov(xs)


alphac(x) = RGBA(0, 0.1, 1.0, x)   

#ax = Axis(fig[1,2])
fig3 = Figure()
ax = Axis(fig3[1,1])
lines!(ax, getindex.(xs2, 1), getindex.(xs2, 2), linewidth=1, color=norm.(getindex.(Ξ[1:end], 3)))
#scatter!(getindex.(xs, 1), getindex.(xs, 2), markersize=1.0)


p = sortperm(norm.((getindex.(Ξ, 2) .- Ref(μₓ))))

scatter(norm.(getindex.(Ξ, 3))[p])

pΔ = 500
dist = [mean(norm.((getindex.(Ξ, 2) .- Ref(μₓ)))[p[i:i+pΔ-1]]) for i in 1:pΔ:length(p)-pΔ]
sig = [mean(std((getindex.(Ξ, 3))[p[i:i+pΔ-1]])) for  i in 1:pΔ:length(p)-pΔ]
sig2 = @. 1/sqrt(F(dist^2/2))

fig = fig2 = Figure(fontsize = 30,  resolution = (1200, 900))
ax = Axis(fig2[1,1], xlabel = "d", ylabel = "σ")
lines!(ax, dist, sig)
lines!(ax, dist, sig2)
save("vr/control$(choice).png", fig)
display(fig2)

fig = fig3 = Figure(fontsize = 30,  resolution = (1200, 900))
ax = Axis(fig[1,1])
qqnorm!(ax, getindex.(xs2, 1), qqline = :fitrobust)
save("vr/control2$(choice).png", fig)
display(fig)

