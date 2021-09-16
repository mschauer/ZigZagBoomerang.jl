#  flexible functions defining boundaries of the form of a $d$-dimensional ball 
using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using ZigZagBoomerang: next_rand_reflect, reflect!, adapt!, loosen, idot, Trace
using ZigZagBoomerang: move_forward!, λ, pos, PDMPTrace
using Random
using StructArrays
using StructArrays: components
using LinearAlgebra
using ZigZagBoomerang: SPriorityQueue, enqueue!, lastiterate

Random.seed!(1)
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
    v[1:d] .*= -1
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
        disc =  ϕ(x[ii]) - ϕ(xnew) # magnitude of the discontinuity
        if  disc < 0.0 || rand() > 1 - exp(-disc) # teleport
            # jump on the other side drawing a line passing through the center of the ball
            println("...teleporting...")
            x[ii] .= xnew 
            bounce = false
        else    # bounce off 
            println("...bouncing...")
            v = circle_boundary_reflection!(i, x, v, ϵ, d)
        end
    end
    return x, v, bounce
end


function event(i, t, x, θ, Z::Union{ZigZag,FactBoomerang})
    t, i, deepcopy(x), deepcopy(θ)
end

function event_NaN(i, t, x, θ, Z::Union{ZigZag,FactBoomerang})
    y = deepcopy(x)
    y[2*i+1:2*(i +1)] .= NaN
    t, i, y, deepcopy(θ)
end

# function Zig.ab(G, i, x, θ, c, Z::ZigZag, args...)
#     a = loosen(c[i], (idot(Z.Γ, i, x)  - idot(Z.Γ, i, Z.μ))'*θ[i])
#     b = loosen(c[i]/100, θ[i]'*idot(Z.Γ, i, θ))
#     a, b
# end

function pdmp_inner!(rng, Ξ, G, ∇ϕ, t, x, θ, Q, c, a, b, t_old, (acc, num), N,
    F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)
    while true
        (hit, i), t′ = peek(Q)
        if t′ - t < 0
            error("negative time")
        end
        t, x, θ = move_forward!(t′ - t, t, x, θ, F)
        if hit # is a hitting time
            if abs(norm(x[1:2] - x[2*i+1:2*i+2]) - ϵ ) > 1e-7 # make sure to hit be on the circle
                error("not at the boundary. distance equal to $(abs(norm(x[1:2] - x[2*i+1:2*i+2])  - ϵ))")
            end
            push!(Ξ, event(i, t, x, θ, F)) # save
            # teleport or bounce off
            x, θ, bounce = circle_hit!(i, x, θ, ϵ; α)
            Q[(true, i)] = Inf
            # update reflections 
            for j in eachindex(x)
                a[j], b[j] = ab(G, j, x, θ, c, F)
                t_old[j] = t
                Q[(false, j)] = t + poisson_time(a[j], b[j], rand(rng))
            end
            if bounce
                # update hitting times
                for j in 1:N
                    if j != i
                        Q[(true, j)] = t + next_circle_hit(j, x, θ, ϵ)
                    end
                end
            else  
                push!(Ξ, event_NaN(i, t, x, θ, F)) # break lines when plotting
                # push!(Ξ, event(i, t, x, θ, F)) 
            end
            # enqueue new reflection and (true Inf) hitting time
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
                θ[i] *= -1 
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
                    i0 = floor(Int, (i-1)/2)
                    Q[(true, i0)] = t + next_circle_hit(i0, x, θ, ϵ)
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




N = 50
dim = 2 # particles in a plane
x = randn((N+1)*dim)
N
# legal region 
function legal(i, x, epsilon)
    y = x[dim*i+1:dim*(i+1)] 
    norm(y - x[1:dim]) > epsilon
end


ϵ = 0.5
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
# θ0[1:2] = rand([-0.5,0.5], 2)
T = 10.0
c = zero(x0) .+ 0.1
Γ = sparse(I(dd))
F = ZigZag(Γ, zero(x0))
α = nothing
Ξ1, (t, x, θ), (acc, num), c = pdmp(∇ϕi, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, α; adapt=false)


# with teleportation
adapt = false
α = 1
Ξ2, (t, x, θ), (acc, num), c = pdmp(∇ϕi, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, α; adapt=false)



function check(Ξ2, N)
    k = 0
    for event in Ξ2
        k += 1
        t, i, x, θ = event
            for i in 1:N
                ii = 2*i+1:2*(i+1) 
                if (ϵ-norm(x[ii] - x[1:2])) > 1e-7
                    println("at iteration $k, ball $i is inside the ball")
                    println("with distnace from the boundary equal to $(ϵ - norm(x[ii] - x[1:2]))") 
                    error("something is wrong")
                end
            end
    end
    true
end
println("check standard pdmp succesful: $(check(Ξ1, N))")
println("check pdmp with teleportation succesful: $(check(Ξ2, N))")
# error("stop without plotting")
##### Plotting 
using GLMakie
GLMakie.activate!
odd = 3:2:2*N+1
even = 4:2:2*N+2

FB = BouncyParticle(Γ, zero(x0), 0.0)
trace = PDMPTrace(FB, t0, x0, θ0, ones(Bool, length(x0)), [(t, x, θ, nothing) for (t, i, x, θ) in Ξ1])
trc1 = collect(discretize(trace, 0.01))

xnode = Node(copy(x0))
xodd = lift(xnode) do x; (x[odd]) end
xeven = lift(xnode) do x; (x[even]) end
circ = lift(xnode) do x; Circle(Point2f0(x[1:2]...), ϵ - 0.01) end

fig = Figure(backgroundcolor = RGBf0(0.98, 0.98, 0.98),
resolution = (700, 700),)
ax1 = Axis(fig[1,1])
limits!(ax1, -5, 5, -5, 5)
scatter!(ax1, xodd, xeven,  marker = 'o', markersize = 20, color = :black)
poly!(circ, color = (:blue,0.6))
Label(fig[1, 1, Top()], "Configuration hard walls", valign = :bottom,
    padding = (0, 0, 2, 0))
nnn = length(trc1)
record(fig, "bigball_hard_wall.mp4", 1:nnn; framerate = 120) do i
        xnode[] = trc1[i][2]
end


FB = BouncyParticle(Γ, zero(x0), 0.0)
trace2 = PDMPTrace(FB, t0, x0, θ0, ones(Bool, length(x0)), [(t, x, θ, nothing) for (t, i, x, θ) in Ξ2])
trc2 = collect(discretize(trace2, 0.01))


xnode = Node(copy(x0))
xodd = lift(xnode) do x; (x[odd]) end
xeven = lift(xnode) do x; (x[even]) end
circ = lift(xnode) do x; Circle(Point2f0(x[1:2]...), ϵ-0.01) end


fig = Figure(backgroundcolor = RGBf0(0.98, 0.98, 0.98),
resolution = (700, 700),)
ax1 = Axis(fig[1,1])
limits!(ax1, -5, 5, -5, 5)
scatter!(ax1, xodd, xeven,  marker = 'o', markersize = 20, color = :black)
poly!(circ, color = (:blue,0.6))
Label(fig[1, 1, Top()], "Configuration teleportation", valign = :bottom,
    padding = (0, 0, 2, 0))
nnn = length(trc2)
record(fig, "bigball_teleportation.mp4", 1:nnn; framerate = 120) do i
        xnode[] = trc2[i][2]
end