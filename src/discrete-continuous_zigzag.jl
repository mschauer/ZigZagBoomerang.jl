using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using ZigZagBoomerang: next_rand_reflect, reflect!
using Random
using StructArrays
using StructArrays: components
using LinearAlgebra
using ZigZagBoomerang: SPriorityQueue, enqueue!, lastiterate

import ZigZagBoomerang: next_rand_reflect, rand_reflect!, reflect!, reset!


function rand_reflect!(i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    smove_forward!(G, i, t, x, θ, m, t′, F) 
    ∇ϕi = P.∇ϕ[1](x, i, args...) #change just here from the original function
    l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i], t[i] - t_old[i])
    if rand(P.rng)*lb < l
        if l >= lb
            !P.adapt && error("Tuning parameter `c` too small.")
            adapt!(c, i, P.factor)
        end
        smove_forward!(G2, i, t, x, θ, m, t′, F)
        ZigZagBoomerang.reflect!(i, ∇ϕi, x, θ, F)
        return true, neighbours(G1, i)
    else
        return false, G1[i].first
    end
    
end

# random reflection of the Zig-Zag
function next_jump_move(j, i, t′, u, P::SPDMP, args...)
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    0, t[j] + randexp() # new exponential time
end


function jump_move!(i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    if  d[i] == 0 # make sure that jumps are allowed 
        error("Discrete moves not available")
    end
    t, x, θ = smove_forward!(G, i, t, x, θ, m, t′, F) 
    x′ = x + θ # propose a jump
    ϕx′ = P.∇ϕ[2](x′)
    ϕx = P.∇ϕ[2](x)
    if  ϕx′ < ϕx || randexp() >   ϕx′ - ϕx 
        x .= x′ # jump
    else
        θ[i] *= -1
    end
    return true, neighbours(G1, i)    
end

# Discrete Gaussian random variable
# t, x, θ, c, t_old, b, d
d = 1
t0 = 0.0
t = fill(t0, d)
x = [1.0] 
θ = θ0 = rand([-1.0, 1.0], d)
Γ = I(1)
F = ZigZag(Γ, x*0)
c = (zero(x) .+ 0.1)

G = G1 = [[1] =>[1]] 
G2 = [[1] => []]
b = [Zig.ab(G1, i, x, θ, c, F) for i in eachindex(θ)] # does not really matter
  
u0 = StructArray(t=t, x=x, θ=θ, θ_old=zeros(d), m=zeros(Int,d), c=c, t_old=copy(t), b=b, θ2 = [1])

Random.seed!(1)
seed = (UInt(1),UInt(1))
rng = Rng(seed)
T = 50.0
t_old = copy(t)
adapt = false
factor = 1.7
σ2 = 5.0 
∇ϕ(x,i) = x[i]/σ2
ϕ(x) = sum(x.^2)/σ2 
P = SPDMP(G, G1, G2, (∇ϕ, ϕ), F, rng, adapt, factor) 


action! = (rand_reflect!, jump_move!)
next_action = FunctionWrangler((next_rand_reflect, next_jump_move))

# action! = (reset!, rand_reflect!, circle_hit1!)
# next_action = FunctionWrangler((Zig.never_reset, next_rand_reflect, n
h = Schedule(action!, next_action, u0, T, (P, ))
trc_ = Zig.simulate(h, progress=true)
trc = Zig.FactTrace(F, t0, x, θ, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_])






ts, xs = getindex.(trc.events,1), getindex.(trc.events,3)
# ts, xs = Zig.sep(Zig.discretize(trc, 0.001))
using Makie
fig = scatter(ts, xs)