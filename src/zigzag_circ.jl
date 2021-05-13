#  flexible functions defining boundaries of the form of a $d$-dimensional ball 
using ZigZagBoomerang
const Zig = ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using ZigZagBoomerang: next_rand_reflect, reflect!
using Random
using StructArrays
using StructArrays: components
using LinearAlgebra
using ZigZagBoomerang: SPriorityQueue, enqueue!, lastiterate

# coefficients of the quadratic equation coming for the condition \|x + θ*t - μ\|^2 = rsq 
function abc_eq2d(μ, rsq, x, θ,)
    a = θ[1]^2 + θ[2]^2
    b = 2*((x[1] - μ[1]) *θ[1] + (x[2] - μ[2])*θ[2]) 
    c = (x[1]-μ[1])^2 + (x[2]-μ[2])^2 - rsq 
    a, b, c
end


# joint reflection at the boundary for the Zig-Zag sampler
# dir = 1 you want to be outside the circle 
# dir = -1 you want to be inside the circle
function circle_boundary_reflection!(μ, rsq, dir, x, θ)
    for i in eachindex(x)
        if dir*θ[i]*(x[i]-μ[i]) < 0.0 
             θ[i]*=-1
        end 
    end
    θ
end


# dir = 1 you want to be outside the circle 
# dir = -1 you want to be inside the circle
function next_circle_hit(μ, rsq, dir, j, i, t′, u, P::SPDMP, args...) 
    if j <= length(u) # not applicable
        return 0, Inf
    else #hitting time to the ball with radius `radius`
        t, x, θ, θ_old, m, c, t_old, b = components(u)
        a1, a2, a3 = abc_eq2d(μ, rsq, x, θ) #solving quadradic equation
        dis = a2^2 - 4a1*a3 #discriminant
         # no solutions or TABU region  
        if dis <=  1e-7 || dir*((x[1] - μ[1])^2 + (x[2] - μ[2])^2  - rsq - 1e-7) < 0.0 
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

# either standard reflection, or bounce at the boundary or traverse the boundary
# α(μ, rsq, x, v) teleportation. Default value is no teleportation 
function circle_hit!(α, μ, rsq, dir, i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, θ_old, m, c, t_old, b = components(u)

    if i > length(u) #hitting boundary
        smove_forward!(G, i, t, x, θ, m, t′, F)
        if abs((x[1] - μ[1])^2 + (x[2] - μ[2])^2  - rsq) > 1e-7 # make sure to hit be on the circle
            dump(u)
            error("not on the circle")
        end
        if α == nothing
            θ .= circle_boundary_reflection!(μ, rsq, dir, x, θ)
        else
            xnew, θnew = α(μ, rsq, x, θ) #    α(x) = -x + 2*μ
            disc =  ϕ(x) - ϕ(xnew) # magnitude of the discontinuity
            if  disc < 0.0 || rand() > 1 - exp(-disc) # teleport
            # if false # never teleport
                # jump on the other side drawing a line passing through the center of the ball
                x .= xnew # improve by looking at G[3]
                θ .= θnew # improve by looking at G[3]
            else    # bounce off 
                θ .= circle_boundary_reflection!(μ, rsq, dir, x, θ)   
            end
        end
        return true, neighbours(G1, i)
    else 
        error("action not available for clock $i")
    end
end

# draw circle
function draw_circ(μ, rsq)
    r = sqrt(rsq)
    θ = LinRange(0,2π, 1000)
    μ[1] .+ r*sin.(θ), μ[2] .+ r*cos.(θ)
end
