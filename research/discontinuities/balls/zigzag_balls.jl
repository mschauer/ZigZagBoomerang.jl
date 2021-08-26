#  flexible functions defining boundaries of the form of a $d$-dimensional ball 
using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using ZigZagBoomerang: next_rand_reflect, reflect!, adapt!, loosen, idot, Trace
using ZigZagBoomerang: move_forward!, λ, pos
using Random
using StructArrays
using StructArrays: components
using LinearAlgebra
using ZigZagBoomerang: SPriorityQueue, enqueue!, lastiterate


# implementation of factorised samplers
using DataStructures
using Statistics
using SparseArrays
using LinearAlgebra
norm2(x) = dot(x, x)
# coefficients of the quadratic equation coming for the condition \|x_i(t) - x_1(t)|^2 = rsq 
function abc_eq2d(i, x, v, ϵ, d)
    ii = d*i+1:d*(i+1)
    a = norm2(v[ii] - v[1:d])
    b = 2*sum((x[ii] - x[1:d]).*(v[ii] - v[1:d]))
    c = norm2(x[ii] - x[1:d]) - ϵ^2
    a, b, c
end


# joint reflection at the boundary for the Zig-Zag sampler
function circle_boundary_reflection!(i, x, v, ϵ, d)
    ii = d*i+1:d*(i+1)
    # v[1:d] .*= -1
    v[ii] .*= -1
    v
end


function next_circle_hit(i, x, v, ϵ)
    d = 2 
    ii = d*i+1:d*(i+1) 
    a1, a2, a3 = abc_eq2d(i, x, v, ϵ, d)
    dis = a2^2 - 4a1*a3 #discriminant
    # no solutions or TABU region  
    if dis <=  1e-7 || (norm2(x[ii] - x[1:d])  - ϵ^2 -  1e-7) < 0.0 
        return Inf 
    else #pick the first positive event time
        hitting_time = min((-a2 - sqrt(dis))/(2*a1),(-a2 + sqrt(dis))/(2*a1))
        if hitting_time <= 0.0
            return Inf
        end
        return hitting_time #hitting time
    end 
end




# either standard reflection, or bounce at the boundary or traverse the boundary
function circle_hit!(i, x, v, ϵ; α =nothing)
    bounce = true
    d = 2
    ii = 2*i+1:2*(i+1)  
    if α == nothing
        v .= circle_boundary_reflection!(i, x, v, ϵ, d)
    else
        xnew = -x[ii] + 2*x[1:d]
        disc =  ϕ(x) - ϕ(xnew) # magnitude of the discontinuity
        if  disc < 0.0 || rand() > 1 - exp(-disc) # teleport
            # jump on the other side drawing a line passing through the center of the ball
            # println("...teleporting...")
            x[ii] .= xnew 
            v[ii] .= vnew 
            bounce = false
        else    # bounce off 
            # println("...bouncing...")
            v .= circle_boundary_reflection!(i, x, v, ϵ, d)
        end
    end
    return x, v, bounce
end


function event(i, t, x, θ, Z::Union{ZigZag,FactBoomerang})
    t, i, deepcopy(x), θ
end

function event_NaN(i, t, x, θ, Z::Union{ZigZag,FactBoomerang})
    y = deepcopy(x)
    y[2*i+1:2*(i +1)] .= NaN
    t, i, y, θ
end

# function Zig.ab(G, i, x, θ, c, Z::ZigZag, args...)
#     a = loosen(c[i], (idot(Z.Γ, i, x)  - idot(Z.Γ, i, Z.μ))'*θ[i])
#     b = loosen(c[i]/100, θ[i]'*idot(Z.Γ, i, θ))
#     a, b
# end


"""
    pdmp_inner!(Ξ, G, ∇ϕ, t, x, θ, Q, c, a, b, t_old, (acc, num),
        F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)
        = t, x, θ, (acc, num), c, a, b, t_old

Inner loop of the factorised samplers: the factorised Boomerang algorithm and
the Zig-Zag sampler. Given a dependency graph `G`, gradient `∇ϕ`,
current position `x`, velocity `θ`, Queue of events `Q`, time `t`, tuning parameter `c`,
terms of the affine bounds `a`,`b` and time when the upper bounds were computed `t_old`

The sampler 1) extracts from the queue the first event time. 2) moves deterministically
according to its dynamics until event time. 3) Evaluates whether the event
time is a accepted reflection or refreshment time or shadow time. 4) If it is a
reflection time, the velocity reflects according its reflection rule, if it is a
refreshment time, the sampler updates the velocity from its prior distribution (Gaussian).
In both cases, updates `Q` according to the dependency graph `G`. The sampler proceeds
until the next accepted reflection time or refreshment time. `(num, acc)`
incrementally counts how many event times occour and how many of those are real reflection times.
"""
function pdmp_inner!(rng, Ξ, G, ∇ϕ, t, x, θ, Q, c, a, b, t_old, (acc, num), N,
    F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)

    while true
        (hit, i), t′ = peek(Q)
        if t′ - t < 0
            error("negative time")
        end
        t, x, θ = move_forward!(t′ - t, t, x, θ, F)
        if hit # is a hitting time
            if abs(norm(x[1:2] - x[2*i+1:2*i+2])  - ϵ) > 1e-3 # make sure to hit be on the circle
                error("not at the boundary. distance equal to $(abs(norm(x[1:2] - x[ii])  - ϵ))")
            end
            # check if at the boundary, teleport or bounce off
            push!(Ξ, event(i, t, x, θ, F)) # save
            x, θ, bounce = circle_hit!(i, x, θ, ϵ; α)
            if bounce
                # update reflections 
                for j in eachindex(x)
                    a[j], b[j] = ab(G, j, x, θ, c, F)
                    t_old[j] = t
                    Q[(false, j)] = t + poisson_time(a[j], b[j], rand(rng))
                end
                # update hitting times
                for j in 1:N
                    if j != i
                        Q[(true, j)] = t + next_circle_hit(j, x, θ, ϵ)
                    end
                end
            else  
                push!(Ξ, event_NaN(i, t, x, θ, F)) # break lines when plotting
                push!(Ξ, event(i, t, x, θ, F)) 
                # update relfections  
                for j in eachindex(x)
                    a[j], b[j] = ab(G, j, x, θ, c, F)
                    t_old[j] = t
                    Q[(false, j)] = t + poisson_time(a[j], b[j], rand(rng))
                end
            end
            # enqueue new reflection and (true Inf) hitting time
            Q[(true, i)] = Inf
        else 
            ∇ϕi = ∇ϕ(x, i, args...)
            l, lb = λ(∇ϕi, i, x, θ, F), pos(a[i] + b[i]*(t - t_old[i]))
            num += 1
            if rand(rng)*lb < l
                acc += 1
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c[i] *= factor
                end
                θ = reflect!(i, ∇ϕi, x, θ, F)
                for j in neighbours(G, i)
                    a[j], b[j] = ab(G, j, x, θ, c, F)
                    t_old[j] = t
                    Q[(false, j)] = t + poisson_time(a[j], b[j], rand(rng))
                end
                if i <= 2 # the big ball reflected
                    for j in 1:N #update all the hitting times
                        Q[(true, j)] = t + next_circle_hit(j, x, θ, ϵ)
                    end
                else
                    ii = floor(Int, (i-1)/2)
                    Q[(true, ii)] = t + next_circle_hit(ii, x, θ, ϵ)
                end
            else
                # Move a, b, t_old inside the queue as auxiliary variables
                a[i], b[i] = ab(G, i, x, θ, c, F)
                t_old[i] = t
                Q[(false, i)] = t + poisson_time(a[i], b[i], rand(rng))
                continue
            end
        end
        push!(Ξ, event(i, t, x, θ, F))
        return t, x, θ, (acc, num), c, a, b, t_old
    end
end



"""
    pdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag, FactBoomerang}, args...; factor=1.5, adapt=false) = Ξ, (t, x, θ), (acc, num), c

Outer loop of the factorised samplers, the Factorised Boomerang algorithm
and the Zig-Zag sampler. Inputs are a function `∇ϕ` giving `i`th element of gradient
of negative log target density `∇ϕ(x, i, args...)`, starting time and position `t0, x0`,
velocities `θ0`, and tuning vector `c` for rejection bounds and final clock `T`.

The process moves to time `T` with invariant mesure μ(dx) ∝ exp(-ϕ(x))dx and outputs
a collection of reflection points which, together with the initial triple `t`, `x`
`θ` are sufficient for reconstructuing continuously the continuous path.
It returns a `FactTrace` (see [`Trace`](@ref)) object `Ξ`, which can be collected
into pairs `t => x` of times and locations and discretized with `discretize`.
Also returns the `num`ber of total and `acc`epted Poisson events and updated bounds
`c` (in case of `adapt==true` the bounds are multiplied by `factor` if they turn
out to be too small.)

This version does not assume that `∇ϕ` has sparse conditional dependencies,
see [`spdmp`](@ref).
"""

# true for hitting times, false for random reflections 
function pdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, args...;
        factor=1.5, adapt=false)
    n = length(x0)
    N = Int((length(x0)-2)/2)
    println("Number of particles: $(N)")
    a = zeros(n)
    b = zeros(n)
    t_old = zeros(n)
    #sparsity graph
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    t, x, θ = t0, copy(x0), copy(θ0)
    num = acc = 0
    rng = Rng()
    Q = PriorityQueue{Tuple{Bool, Int64},Float64}()
    for i in eachindex(θ)
        a[i], b[i] = ab(G, i, x, θ, c, F)
        t_old[i] = t
        enqueue!(Q, (false, i) => t + poisson_time(a[i], b[i], rand(rng)))
    end
    for i in 1:N
        enqueue!(Q, (true, i) => t + next_circle_hit(i, x, θ, ϵ))
    end
    Ξ = [(t0, 0, x0, θ0)]
    
    while t < T
        t, x, θ, (acc, num), c, a, b, t_old = pdmp_inner!(rng, Ξ, G, ∇ϕ, t, x, θ, Q, c, a, b, t_old, (acc, num), N, F, args...; factor=factor, adapt=adapt)
    end
    Ξ, (t, x, θ), (acc, num), c
end




N = 20
dim = 2 # particles in a plane
x = randn((N+1)*dim)
N
# legal region 
function legal(i, x, epsilon)
    y = x[dim*i+1:dim*(i+1)] 
    norm(y - x[1:dim]) > epsilon
end


ϵ = 2.0
# initialize particles in a legal region
for i in 1:N
    while true 
        if legal(i, x, ϵ)
            break
        end
        x[i*dim + 1: dim*(i+1)] =  randn(dim)
    end
end

# standard gaussian log-likelihood (for now no interactions) 
ϕ(x) = sum(x.^2)/2
∇ϕi(x, i, α) = x[i]  
adapt = false
x0 = deepcopy(x) # initial position
dd = length(x0)
t0 = 0.0
θ0 = rand([1.0,-1.0], dd)
θ0[1:2] = rand([-0.5,0.5], 2)
T = 500.0
c = zero(x0) .+ 0.1
Γ = sparse(I(dd))
F = ZigZag(Γ, zero(x0))
α = nothing
Ξ1, (t, x, θ), (acc, num), c = pdmp(∇ϕi, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, α; adapt=false)


tt1, xx1 = getindex.(Ξ1,1),  getindex.(Ξ1,3)
fig = Figure()
ax1 = Axis(fig[1,1])
limits!(ax1, -5, 5, -5, 5)
lines!(ax1, getindex.(xx1, 1),getindex.(xx1, 2), label = "volume", color = (:red, 0.1) )
lines!(ax1, getindex.(xx1, 3),getindex.(xx1, 4),  color = (:blue, 0.5))

adapt = false
x0 = deepcopy(x) # initial position
dd = length(x0)
t0 = 0.0
θ0 = rand([1.0,-1.0], dd)
θ0[1:2] = rand([-0.5,0.5], 2)
c = zero(x0) .+ 0.1
Γ = sparse(I(dd))
F = ZigZag(Γ, zero(x0))
α = 1
Ξ2, (t, x, θ), (acc, num), c = pdmp(∇ϕi, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, α; adapt=false)

tt2, xx2 = getindex.(Ξ2,1),  getindex.(Ξ2,3)
ax2 = Axis(fig[1,2])
limits!(ax2, -5, 5, -5, 5)
lines!(ax2, getindex.(xx2, 1),getindex.(xx2, 2), label = "volume", color = (:red, 0.1) )
lines!(ax2, getindex.(xx2, 3),getindex.(xx2, 4),  color = (:blue, 0.5))


current_figure()