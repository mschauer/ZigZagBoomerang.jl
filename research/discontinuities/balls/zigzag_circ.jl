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

# coefficients of the quadratic equation coming for the condition \|x_i(t) + x_1(t)|^2 = rsq 
function abc_eq2d(i, x, v, ϵ, d)
    ii = d*i+1:d*(i+1)
    a = norm(v[ii] - v[1:d])^2
    b = 2*sum((x[ii] - x[1:d]).*(v[ii] - v[1:d]))
    c = norm(x[ii] - x[1:d])^2 - ϵ^2 
    a, b, c
end


# joint reflection at the boundary for the Zig-Zag sampler
function circle_boundary_reflection!(i, x, v, ϵ, d)
    ii = d*i+1:d*(i+1)
    v[1:d] .*= -1
    v[ii] .*= -1
    v
end


function next_circle_hit(i, x, v, ϵ, d)
    ii = d*i+1:d*(i+1)  
    a1, a2, a3 = abc_eq2d(i, x, v, ϵ, d)
    dis = a2^2 - 4a1*a3 #discriminant
    # no solutions or TABU region  
    if dis <=  1e-7 || ((x[1] - μ[1])^2 + (x[2] - μ[2])^2  - rsq -  1e-7) < 0.0 
    if dis <=  1e-7 || (norm(x[ii] - x[1:d])  - ϵ^2 -  1e-7) < 0.0 
        return 0, Inf 
    else #pick the first positive event time
        hitting_time = min((-a2 - sqrt(dis))/(2*a1),(-a2 + sqrt(dis))/(2*a1))
        if hitting_time <= 0.0
            return 0, Inf
        end
        return 0, t′ + hitting_time #hitting time
    end 
end

# either standard reflection, or bounce at the boundary or traverse the boundary
function circle_hit!(i, x, v, ϵ, d; α =nothing)
    ii = d*i+1:d*(i+1)  
    smove_forward!(G, i, t, x, v, m, t′, F)
    if abs(norm(x[1:d] - x[ii])^2  - ϵ^2) > 1e-7 # make sure to hit be on the circle
        error("not on the circle")
    end
    if α == nothing
        v .= circle_boundary_reflection!(i, x, v, ϵ, d)
    else
        xnew, vnew = α(x, v, ϵ) #    α(x) = -x[ii] + 2*x[1:d], θ[ii]
        disc =  ϕ(x) - ϕ(xnew) # magnitude of the discontinuity
        if  disc < 0.0 || rand() > 1 - exp(-disc) # teleport
        # if false # never teleport
            # jump on the other side drawing a line passing through the center of the ball
            x[ii] .= xnew # improve by looking at G[3]
            v[ii] .= vnew # improve by looking at G[3]
        else    # bounce off 
            v .= circle_boundary_reflection!(i, x, v, ϵ, d)
        end
    end
    return x, v
end

# draw circle in 2d
function draw_circ(μ, rsq)
    r = sqrt(rsq)
    θ = LinRange(0,2π, 1000)
    μ[1] .+ r*sin.(θ), μ[2] .+ r*cos.(θ)
end
