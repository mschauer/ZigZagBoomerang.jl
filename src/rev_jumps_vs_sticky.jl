using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using Random
using StructArrays
using SparseArrays
using StructArrays: components
using LinearAlgebra
using ZigZagBoomerang: SPriorityQueue, enqueue!, lastiterate

function next_rand_reflect(j, i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    if m[j] == 1 
        return 0, Inf
    end
    b[j] = ab(G1, j, x, θ, c, F)
    t_old[j] = t′
    return 0, t[j] + poisson_time(b[j], rand(P.rng))
end

function multiple_freeze!(ξξ, rev, i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    # find integer
    fake = (t′ - t[i]) == 0.0
    ev, jj = findmin([norm((x[i] + θ[i]*(t′ - t[i]) - ξ)) for ξ in ξξ])
    @assert ev < 1e-7
    smove_forward!(G, i, t, x, θ, m, t′, F)
    smove_forward!(G2, i, t, x, θ, m, t′, F)
    
    if m[i] == 0 # to freeze
        x[i] = ξξ[jj]
        t[i] = t′
        m[i] = 1
        θ_old[i], θ[i] = θ[i], 0.0
    else # to unfreeze
        m[i] = 0
        t[i] = t′
        if rev == true && rand() < 0.5 && fake == false #reversibility
            θ[i] = -θ_old[i]
        else
            θ[i] = θ_old[i]
        end
    end
    return true, neighbours(G1, i)
end

function next_freezeunfreeze(ξξ, rev, κ, j, i, t′, u, P::SPDMP, args...) 
    t, x, θ, θ_old, m = components(u)
    if m[j] == 0
        ev, kk = findmin([θ[j]*(x[j] - ξ) >= 0 ? Inf : t[j] - (x[j] - ξ)/θ[j] for ξ in ξξ]) 
        return 0, ev
    else
        if rev == true # tuning parameter of Chevallier
            if rand() > 0.6
                return 0, t[j] #don't stick
            else
                return 0, t[j] + poisson_time(κ*0.6, rand(P.rng))
            end
        else    # rev = false
            return 0, t[j] + poisson_time(κ, rand(P.rng))
        end
    end
end

sticky_floors = collect(-5.0:0.5:5.0)

function multiple_rev_freeze!(args...)
    multiple_freeze!(sticky_floors, true, args...)
end

function multiple_nonrev_freeze!(args...)
    multiple_freeze!(sticky_floors, false, args...)
end

function next_freezeunfreeze_rev(args...)
    next_freezeunfreeze(sticky_floors, true, 1.0, args...)
end 

function next_freezeunfreeze_nonrev(args...)
    next_freezeunfreeze(sticky_floors, false, 1.0, args...)
end 


T = 100.0
d = 50
seed = (UInt(1),UInt(1))
Random.seed!(1)



S = I(d)*0.1 .+ 0.9 
Γ = inv(S)

∇ϕ(x, i, Γ) = Zig.idot(Γ, i, x) # sparse computation

# t, x, θ, θ_old, m, c, t_old, b 
t0 = 0.0
t = fill(t0, d)
x = rand(d) 
θ = θ0 = rand([-1.0, 1.0], d)
F = ZigZag(Γ, x*0)



c = .01*[norm(Γ[:, i], 2) for i in 1:d]
G = G1 = [[i] => collect(1:d) for i in eachindex(θ0)]
G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]

b = [ab(G1, i, x, θ, c, F) for i in eachindex(θ)]
  
u0 = StructArray(t=t, x=x, θ=θ, θ_old = zeros(d), m=zeros(Int,d), c=c, t_old=copy(t), b=b)


rng = Rng(seed)
t_old = copy(t)
adapt = false
factor = 1.7
P = SPDMP(G, G1, G2, ∇ϕ, F, rng, adapt, factor)

action1! = (Zig.rand_reflect!, multiple_rev_freeze!,)
next_action1 = FunctionWrangler((next_rand_reflect, next_freezeunfreeze_rev,))


# h = Schedule(action!, next_action, u0, T, (P, Γ))

# l_ = lastiterate(h) 

# using ProfileView, Profile
# Profile.init(10000000, 0.00001)
# ProfileView.@profview lastiterate(h)
# l_ = @time lastiterate(h) 

h1 = Schedule(action1!, next_action1, u0, T, (P, Γ))
trc_1 = @time collect(h1);
trc1 = Zig.FactTrace(F, t0, x, θ, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_1])


action2! = (Zig.rand_reflect!, multiple_nonrev_freeze!,)
next_action2 = FunctionWrangler((next_rand_reflect, next_freezeunfreeze_nonrev,))

h2 = Schedule(action2!, next_action2, u0, T, (P, Γ))
trc_2 = @time collect(h2);
trc2 = Zig.FactTrace(F, t0, x, θ, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_2])

# h = Schedule(action!, next_action, u0, T, (P, Γ))

# l_ = lastiterate(h) 

# using ProfileView, Profile
# Profile.init(10000000, 0.00001)
# ProfileView.@profview lastiterate(h)
# l_ = @time lastiterate(h) 



#trace, _, acc = @time spdmp(∇ϕ, t0, x, θ, T, c, G, F, Γ);
#@code_warntype handler(zeros(d), T, (f1!, f2!));

#using ProfileView
error()
#ProfileView.@profview handler(zeros(d), 10T);
using GLMakie
#subtrace1 = [t for t in trc_ if t[2] == 1]
#lines(getindex.(subtrace1, 1), getfield.(getindex.(subtrace1, 3), :x))

ts1, xs1 = Zig.sep(Zig.subtrace(trc1, [1,4]))
lines(ts1, getindex.(xs1, 2))
fig = lines(getindex.(xs1, 1), getindex.(xs1, 2))

ts2, xs2 = Zig.sep(Zig.subtrace(trc2, [1,4]))
lines(ts2, getindex.(xs2, 2))
fig = lines(getindex.(xs2, 1), getindex.(xs2, 2))