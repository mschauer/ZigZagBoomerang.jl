using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using ZigZagBoomerang: next_rand_reflect, reflect!
using Random
using StructArrays
using SparseArrays
using StructArrays: components
using LinearAlgebra
using ZigZagBoomerang: SPriorityQueue, enqueue!, lastiterate
# import standard reflection of zigzag
import ZigZagBoomerang: next_rand_reflect, rand_reflect!, reflect!, reset!


# overload simulate and handle! to allow having 2 more clocks than coordinates
# from `d` to `d+2` 
function Zig.simulate(handler; progress=true, progress_stops = 20)
    T = handler.T
    if progress
        prg = Zig.Progress(progress_stops, 1)
    else
        prg = missing
    end
    stops = ismissing(prg) ? 0 : max(prg.n - 1, 0) # allow one stop for cleanup
    tstop = T/stops

    u = deepcopy(handler.state)
    action! = handler.action!
    next_action = handler.next_action
    d = length(handler.state)
    action = ones(Int, d+2) # resets only 
    
    Q = Zig.SPriorityQueue(zeros(d+2))
    
    evs = Zig.handle!(u, action!, next_action, action, Q, handler.args...)
 #   evs = [ev]
    while true
        ev = Zig.handle!(u, action!, next_action, action, Q, handler.args...)
        t′ = ev[end][1]
        t′ > T && break
        append!(evs, ev)
        if t′ > tstop
            tstop += T/stops
            Zig.next!(prg) 
        end  
    end
    ismissing(prg) || Zig.ProgressMeter.finish!(prg)
    return evs
end

function Zig.handle!(u, action!, next_action, action, Q, args::Vararg{Any, N}) where {N}
    # Who is (i) next_action, when (t′) and what (j) happens?
    done = false
    local e, t′, i
    num = 0
    while !done
        num += 1
        i, t′ = Zig.peek(Q)
        e = action[i]
        #done = action_nextaction(action!, next_action, Q, action, e, i, t′, u, args...)
        done = Zig.switch(e, action!, next_action, (Q, action), i, t′, u, args...)
    end
    traceevent(t′, i, u, action, num)
end

function traceevent(t′, i, u, action, num)
    if 1 <= i <= length(u)
        return  [(t′, i, u[i], action[i], num)]
    else
        return [(t′, 1, u[1], action[1], num), (t′, 2, u[2], action[2], num)] 
    end
end

# random reflection of the Zig-Zag
function next_rand_reflect(j, i, t′, u, P::SPDMP, args...)
    1 <= j <= length(u) || return 0, Inf
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    if m[j] == 1 
        return 0, Inf
    end
    b[j] = ab(G1, j, x, θ, c, F)
    t_old[j] = t′
    0, t[j] + poisson_time(b[j], rand(P.rng))
end

function rand_reflect!(i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    @assert 1 <= i <= length(u)
    smove_forward!(G, i, t, x, θ, m, t′, F) 
    ∇ϕi = P.∇ϕ(x, i, args...)
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

# reset
function reset!(i, t′, u, P::SPDMP, args...) # should move the coordinates if called later in the game
    false, P.G1[i].first
end



include("./zigzag_circ.jl")
# stay outside ball centered at μ1 with radius rsq1, if hit then teleport or reflect
# α(μ, rsq, x, v) = (-x + 2*μ, v) # teleportation
α = nothing # no teleportation
next_circle_hit1(j, i, t′, u, P::SPDMP, args...) = next_circle_hit([-0.7, -0.7], 0.5, 1, j, i, t′, u, P::SPDMP, args...) 
circle_hit1!(i, t′, u, P::SPDMP, args...) = circle_hit!(α, [-0.7, -0.7], 0.5, 1, i, t′, u, P::SPDMP, args...)

# stay outside ball centered at μ1 with radius rsq1, if hit then teleport or reflect 
next_circle_hit2(j, i, t′, u, P::SPDMP, args...) = next_circle_hit([0.0, 0.0], 4.0, -1, j, i, t′, u, P::SPDMP, args...) 
circle_hit2!(i, t′, u, P::SPDMP, args...) = circle_hit!(α, [0.0, 0.0], 4.0, -1, i, t′, u, P::SPDMP, args...)

Sigma = Matrix([1.0 0.0; 
                    0.0 1.0])

Γ = sparse(Sigma^(-1))

ϕ(x) =  -0.5*x'*Γ*x  # negated log-density
∇ϕ(x, i) = Zig.idot(Γ, i, x) # sparse computation

# t, x, θ, c, t_old, b 
d = 2
t0 = 0.0
t = fill(t0, d)
x = [1.0, 1.0] + rand(d) 
θ = θ0 = rand([-1.0, 1.0], d)
F = ZigZag(Γ, x*0)



# G, G2: Clocks to coordinates
# G1: Clocks to clocks

c = (zero(x) .+ 0.1)
G = [[i] => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
push!(G, [3] => [1,2]) # move all coordinates
push!(G, [4] => [1,2]) # move all coordinates
G1 = [[i] => [rowvals(F.Γ)[nzrange(F.Γ, i)]; [3, 4]] for i in eachindex(θ0)]
push!(G1, [3] => [1, 2, 3, 4]) # invalidate ALL the clocks yeah
push!(G1, [4] => [1, 2, 3, 4]) # invalidate ALL the clocks yeah
# push!(G1, [3] => [1, 2, 3]) # invalidate ALL the clocks yeah

# G2 = [1 => [2], 2 => [1], 3=>Int[]] # 
G2 = [1 => [2], 2 => [1], 3=>Int[], 4=>Int[]] # 
#G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G)]

b = [ab(G1, i, x, θ, c, F) for i in eachindex(θ)] 
  
u0 = StructArray(t=t, x=x, θ=θ, θ_old=zeros(d), m=zeros(Int,d), c=c, t_old=copy(t), b=b)

Random.seed!(1)
seed = (UInt(1),UInt(1))
rng = Rng(seed)
T = 500.0
t_old = copy(t)
adapt = false
factor = 1.7
P = SPDMP(G, G1, G2, ∇ϕ, F, rng, adapt, factor) 

action! = (reset!, rand_reflect!, circle_hit1!, circle_hit2!)
next_action = FunctionWrangler((Zig.never_reset, next_rand_reflect, next_circle_hit1, next_circle_hit2))

# action! = (reset!, rand_reflect!, circle_hit1!)
# next_action = FunctionWrangler((Zig.never_reset, next_rand_reflect, next_circle_hit1))

h = Schedule(action!, next_action, u0, T, (P, ))
trc_ = Zig.simulate(h, progress=true)
trc = Zig.FactTrace(F, t0, x, θ, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_])
error("")

#subtrace1 = [t for t in trc_ if t[2] == 1]
#lines(getindex.(subtrace1, 1), getfield.(getindex.(subtrace1, 3), :x))
ts, xs = Zig.sep(Zig.discretize(trc, 0.001))

using Makie
# scatter(ts, getindex.(xs, 2))
fig = lines(getindex.(xs, 1), getindex.(xs, 2), )
lines!(draw_circ([0.0, 0.0], 4.0), color = :red)
lines!(draw_circ( [-0.7, -0.7], 0.5), color = :red)
save("two_balls_one_inside_the_other.png", fig)