using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using Random
using StructArrays
using StructArrays: components
using LinearAlgebra
using ZigZagBoomerang: SPriorityQueue, enqueue!
function lastiterate(itr) 
    ϕ  = iterate(itr)
    if ϕ === nothing
        error("empty")
    end
    x, state = ϕ
    while true
        ϕ = iterate(itr, state)
        if ϕ === nothing 
            return x
        end
        x, state = ϕ
    end
end

T = 500.0
d = 80
seed = (UInt(1),UInt(1))


𝕁(j) = mod(j,2) == 0
function next_reset(j, _, t′, u, P, args...)
    (!𝕁(j)) ? Inf : t′ + 0.5*u.x[j]
end



function reflect0!(i, t′, u, P::SPDMP, args...)
    Zig.reflect!(0.0, 1, i, t′, u, P::SPDMP, args...)
end

function reflect1!(i, t′, u, P::SPDMP, args...)
    Zig.reflect!(1.0, -1, i, t′, u, P::SPDMP, args...)
end

function next_rand_reflect(j, i, t′, u, P, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    if m[j] == 1 
        return Inf
    end
    if !𝕁(j)
        b[j] = ab(G1, j, x, θ, c, F)
    else
        b[j] = ab(G1, j, x, θ, c, F) .+ (1/(x[j]), 2/(x[j]^2))
    end
    t_old[j] = t′
    t[j] + poisson_time(b[j], rand(P.rng))
end

function next_reflect0(j, i, t′, u, args...) 
    Zig.next_reflect(0.0, 1, j, i, t′, u, args...) 
#=
    t, x, θ, θ_old, m = components(u)

    if x[j] < 0
        return t[j]
    end
    θ[j]*x[j] >= 0 ? Inf : t[j] - x[j]/θ[j]=#
end

function next_reflect1(j, i, t′, u, args...) 
    Zig.next_reflect(1.0, -1, j, i, t′, u, args...) 
    #=
    t, x, θ, θ_old, m = components(u)
    if x[j] > 1
        return t[j]
    end
    θ[j]*(x[j]-1) >= 0 ? Inf : t[j] - (x[j]-1)/θ[j]
    =#
end

function freeze!(args...)
    Zig.freeze!(0.25, args...)
end
#if !@isdefined(discontinuity!)
function discontinuity!(args...) 
    Zig.discontinuity_at!(0.5, 0.0, 1, args...)
end
function next_discontinuity(args...)
    Zig.next_hit(0.5, args...)
end 

function next_freezeunfreeze(args...)
    Zig.next_freezeunfreeze(0.25, 1.0, args...)
end 

Random.seed!(1)

using SparseArrays

S = 1.3I + 0.5sprandn(d, d, 0.1)
Γ = S*S'

∇ϕ(x, i, Γ) = -(𝕁(i))/x[i] + Zig.idot(Γ, i, x) # sparse computation

# t, x, θ, θ_old, m, c, t_old, b 
t0 = 0.0
t = fill(t0, d)
x = rand(d) 
θ = θ0 = rand([-1.0, 1.0], d)
F = ZigZag(0.9Γ, x*0)



c = .6*[norm(Γ[:, i], 2) for i in 1:d]
G = G1 = [[i] => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]

b = [ab(G1, i, x, θ, c, F) for i in eachindex(θ)]
  
u0 = StructArray(t=t, x=x, θ=θ, θ_old = zeros(d), m=zeros(Int,d), c=c, t_old=copy(t), b=b)


rng = Rng(seed)
t_old = copy(t)
adapt = false
factor = 1.7
P = SPDMP(G, G1, G2, ∇ϕ, F, rng, adapt, factor)

#action! = FunctionWrangler((reset!, rand_reflect!, reflect0!, reflect1!))
#next_action = FunctionWrangler((next_reset, next_rand_reflect,  next_reflect0, next_reflect1))

action! = (Zig.reset!, Zig.rand_reflect!, discontinuity!, freeze!, reflect0!, reflect1!)
next_action = FunctionWrangler((next_reset, next_rand_reflect, next_discontinuity, next_freezeunfreeze, next_reflect0, next_reflect1))

#action! = (reset!, rand_reflect!, discontinuity!, reflect0!, reflect1!)
#next_action = FunctionWrangler((next_reset, next_rand_reflect, next_discontinuity,  next_reflect0, next_reflect1))

#action! = ((reset!, rand_reflect!))
#next_action = FunctionWrangler((next_reset, next_rand_reflect))


#h = Schedule(FunctionWrangler(action!), next_action, u0, T, (P, Γ))

h = Schedule(action!, next_action, u0, T, (P, Γ))

l_ = lastiterate(h) 

using ProfileView, Profile
Profile.init(10000000, 0.00001)
ProfileView.@profview lastiterate(h)
l_ = @time lastiterate(h) 

total, l = simulate(h)
_, l = @time simulate(h)
trc_ = @time collect(h);
trc = Zig.FactTrace(F, t0, x, θ, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_])


trace, _, acc = @time spdmp(∇ϕ, t0, x, θ, T, c, G, F, Γ);
#@code_warntype handler(zeros(d), T, (f1!, f2!));

#using ProfileView
error()
#ProfileView.@profview handler(zeros(d), 10T);
using GLMakie
#subtrace1 = [t for t in trc_ if t[2] == 1]
#lines(getindex.(subtrace1, 1), getfield.(getindex.(subtrace1, 3), :x))

ts, xs = Zig.sep(Zig.subtrace(trc, [1,4]))

lines(ts, getindex.(xs, 2))
fig = lines(getindex.(xs, 1), getindex.(xs, 2))