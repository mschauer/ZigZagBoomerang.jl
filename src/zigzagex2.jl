#####################################################################
# 2d Zig-Zag ouside ball: |x| > c                                   #
# with glued boundaries for x in Γ (exit-non-entrance):  x -> -x    #
# fake 3rd coordinate for reflections and hitting times             #
#####################################################################

using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using ZigZagBoomerang: next_rand_reflect, reflect!
using Random
using StructArrays
using StructArrays: components
using LinearAlgebra
using ZigZagBoomerang: SPriorityQueue, enqueue!, lastiterate
# import standard reflection of zigzag
import ZigZagBoomerang: next_rand_reflect, rand_reflect!, reflect!, reset!


# overload simulate and handle! to allow having more clocks than coordinates
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
    action = ones(Int, d+1) # resets
    
    Q = Zig.SPriorityQueue(zeros(d+1))
    
    evs = Zig.handle!(u, action!, next_action, action, Q, handler.args...)
 #   evs = [ev]
    while true
        ev = Zig.handle!(u, action!, next_action, action, Q, handler.args...)
        if length(ev) == 1
            println("reflection")
        else
            println("hitting the circle")
        end
        t′ = ev[end][1]
        println("at time current time: $(t′)")
        println("")
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

function rand_reflect!(i, t′, u, P::SPDMP, nt, args...)
    μ, rsq = nt.μ, nt.rsq
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)
    @assert 1 <= i <= length(u)
    smove_forward!(G, i, t, x, θ, m, t′, F) 
    ∇ϕi = P.∇ϕ(x, i, nt, args...)
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



Random.seed!(1)

using SparseArrays
T = 500.0
d = 2
seed = (UInt(1),UInt(1))


function reset!(i, t′, u, P::SPDMP, args...) # should move the coordinates if called later in the game
    false, P.G1[i].first
end

# coefficients of the quadratic equation coming for the condition \|x + θ*t - μ\|^2 > rsq
function abc_eq2d(x, θ, μ, rsq)
    a = θ[1]^2 + θ[2]^2
    b = 2*((x[1] -μ[2]) *θ[1] + (x[2] - μ[2])*θ[2]) 
    c = (x[1]-μ[1])^2 + (x[2]-μ[2])^2 - rsq 
    a, b, c
end

# joint reflection at the boundary
function circle_boundary_reflection!(x, θ, μ)
    for i in eachindex(x)
        if θ[i]*(x[i]-μ[i]) < 0.0 
             θ[i]*=-1
        end 
    end
    θ
end

function next_circle_hit(j, i, t′, u, P::SPDMP, nt, args...) 
    μ, rsq = nt.μ, nt.rsq
    if j <= length(u) # not applicable
        return 0, Inf
    else #hitting time to the ball with radius `radius`
        t, x, θ, θ_old, m, c, t_old, b = components(u)
        a1, a2, a3 = abc_eq2d(x, θ, μ, rsq) #solving quadradic equation
        dis = a2^2 - 4a1*a3 #discriminant
         # no solutions or inside
        if dis <=  1e-7 || (x[1] - μ[1])^2 + (x[2] - μ[2])^2  - rsq < 1e-7 
            return 0, Inf 
        else #pick the first positive event time 
            hitting_time = min((-a2 - sqrt(dis))/(2*a1),(-a2 + sqrt(dis))/(2*a1))
            if hitting_time <= 0.0
                return 0, Inf
            end

            return 0, t′ + hitting_time #hitting time
        end 
    end
end

# either standard reflection, or bounce at the boundary or traversing the boundary
function circle_hit!(i, t′, u, P::SPDMP, nt, args...)
    μ, rsq = nt.μ, nt.rsq
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)

    if i == 3 #hitting boundary
        smove_forward!(G, i, t, x, θ, m, t′, F)
 
        if abs((x[1] - μ[1])^2 + (x[2] - μ[2])^2  - rsq) > 1e-7 # make sure to hit be on the circle
            dump(u)
            error("not on the circle")
        end
        disc =  ϕ(x) - ϕ(-x + 2*μ) # magnitude of the discontinuity
        if  disc < 0.0 || rand() > 1 - exp(-disc) # teleport
        # if false # never teleport
            # jump on the other side drawing a line passing through the center of the ball
            x .= -x + 2 .*μ # improve by looking at G[3]
        else    # bounce off 
            θ .= circle_boundary_reflection!(x, θ, μ)    
        end
        return true, neighbours(G1, i)
    else 
        error("action not available for clock $i")
    end
end



Sigma = Matrix([1.0 0.9; 
                    0.9 1.0])

Γ = sparse(Sigma^(-1))

ϕ(x) =  -0.5*x'*Γ*x  # negated log-density
∇ϕ(x, i, nt) = Zig.idot(nt.Γ, i, x) # sparse computation

# t, x, θ, c, t_old, b 
d = 2
t0 = 0.0
t = fill(t0, d)
x = [2.0, 2.0] + rand(d) 
θ = θ0 = rand([-1.0, 1.0], d)
F = ZigZag(Γ, x*0)

# G, G2: Clocks to coordinates
# G1: Clocks to clocks

c = (zero(x) .+ 0.1)
G = [[i] => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
push!(G, [3] => [1,2]) # move all coordinates
G1 = [[i] => [rowvals(F.Γ)[nzrange(F.Γ, i)];3] for i in eachindex(θ0)]
push!(G1, [3] => [1, 2, 3]) # invalidate ALL the clocks yeah
G2 = [1 => [2], 2 => [1], 3=>Int[]] # 
#G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G)]

b = [ab(G1, i, x, θ, c, F) for i in eachindex(θ)] # specific for subsampling
  
u0 = StructArray(t=t, x=x, θ=θ, θ_old=zeros(d), m=zeros(Int,d), c=c, t_old=copy(t), b=b)


rng = Rng(seed)
t_old = copy(t)
adapt = false
factor = 1.7
P = SPDMP(G, G1, G2, ∇ϕ, F, rng, adapt, factor) # ?

action! = (reset!, rand_reflect!, circle_hit!)
next_action = FunctionWrangler((Zig.never_reset, next_rand_reflect, next_circle_hit))
rsq = 3.0
μ = [-1.0, -1.0]
h = Schedule(action!, next_action, u0, T, (P, (Γ=Γ, μ=μ, rsq=rsq)))
trc_ = Zig.simulate(h, progress=true)
trc = Zig.FactTrace(F, t0, x, θ, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_])

ts, xs = Zig.sep(Zig.discretize(trc, 0.001))
error("")

using Makie
# scatter(ts, getindex.(xs, 2))
fig = lines(getindex.(xs, 1), getindex.(xs, 2), )
axes = Axis(fig)
#draw the circle with radius r, centered in μ
r = sqrt(rsq)
x1 = Float64.(-r:0.0001:r)
x2 = zero(x1)
for i in eachindex(x1)
    x2[i] = sqrt(r^2 - x1[i]^2)
end
lines!(x1 .+ μ[1], x2 .+ μ[2], color = "red")
lines!(x1 .+ μ[1], -x2 .+ μ[2], color = "red")
save("bounce_off_the_ball_with_teleportation.png", fig)