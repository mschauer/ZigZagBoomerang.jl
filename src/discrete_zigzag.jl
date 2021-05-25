using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using Random
using StructArrays
using SparseArrays
using StructArrays: components
using LinearAlgebra
using ZigZagBoomerang: SPriorityQueue, enqueue!, lastiterate

# import standard reflection of zigzag

# t, x, θ, θ_old, m, c, t_old, b, θ2 = components(u) 
#introducing θ2 which is used for giving the direction when jumping  

# random reflection of the Zig-Zag
function next_discrete_move(j, i, t′, u, P::SPDMP, args...)
    t, x, θ, θ_old, m, c, t_old, b, θ2 = components(u)
    @assert m[j] == 1
    d[j] == 1 || return false, Inf # if it s not a discrete random variable, then return Inf
    # G, G1, G2 = P.G, P.G1, P.G2
    # F = P.F
    # t_old[j] = t′
    0, t[j] + randexp() # new exponential time
end


function discrete_move!(i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b, θ2 = components(u)
    if  d[i] == 0 #make sure it s a discrete random variable
        error("Discrete moves not available for continuous components")
    end
    @assert m[i] == 1
    t, x, θ = smove_forward!(G, i, t, x, θ, m, t′, F) 
    t[i] = t′
    # ∇ϕi = P.∇ϕ(x, i, args...)
    # l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i], t[i] - t_old[i])
    x′ = deepcopy(x) 
    x′ = x + θ2 #proposed a jump
    ϕx′ = P.∇ϕ[2](x′)
    ϕx = P.∇ϕ[2](x)
    println("")
    println("current position $(x[1]),$(θ2[1]), with  $(exp(-ϕx))")
    println("new proposed position $(x′[1]),$(θ2[1]) with probability $(exp(-ϕx′))")
    if  ϕx′ < ϕx || randexp() >   ϕx′ - ϕx 
        println("accepted")
        x .= x′
    else
        println("rejected")
        θ2[i] *= -1
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
  
u0 = StructArray(t=t, x=x, θ=θ, θ_old=zeros(d), m=ones(Int,d), c=c, t_old=copy(t), b=b, θd = [1])

Random.seed!(1)
seed = (UInt(1),UInt(1))
rng = Rng(seed)
T = 10.0
t_old = copy(t)
adapt = false
factor = 1.7
σ2 = 5.0 
∇ϕ(x,i) = x[i]/σ2
ϕ(x) = sum(x.^2)/σ2 
P = SPDMP(G, G1, G2, (∇ϕ, ϕ), F, rng, adapt, factor) 


action! = (discrete_move!,)
next_action = FunctionWrangler((next_discrete_move,))

# action! = (reset!, rand_reflect!, circle_hit1!)
# next_action = FunctionWrangler((Zig.never_reset, next_rand_reflect, n
h = Schedule(action!, next_action, u0, T, (P, ))
trc_ = Zig.simulate(h, progress=true)
trc = Zig.FactTrace(F, t0, x, θ, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_])






ts, xs = getindex.(trc.events,1), getindex.(trc.events,3)
# ts, xs = Zig.sep(Zig.discretize(trc, 0.001))
using Makie
fig = lines(ts, xs)