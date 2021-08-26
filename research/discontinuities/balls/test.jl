using CairoMakie
using Makie.GeometryBasics

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

### initialize 2 particles
n = 1
N = 2*(n+1)
x = ones(N) + randn(N)*0.01
v = ones(N).*0.5
ϵ = 0.5
odd = 1:2:N-1
even = 2:2:N
x[1:2] .= [-1.0,-1.0]  
v[3:4] .= [-1.0, -1.0]
odd = 1:2:N
scatter(x[odd], x[even], color = [:blue, :red])

# compute hitting
t = next_circle_hit(1, x, v, ϵ)
# move forward
x .= x .+ v.*t

scatter!(x[odd], x[even], color = [:blue, :red])
poly!(Circle(Point2f0(y[1:2]...), ϵ), color = (:pink,0.4))
current_figure()
# is at the boundary?
println("at the boundary? $(norm2(x[1:2] - x[3:4])  - ϵ^2 < 1e-7)")  
println("particle at : $(x[3:4])")
println("teleporting")
#option 1 => teleport
xnew, vnew = -x[3:4] + 2*x[1:2], v[3:4]
x[3:4] .= xnew 
v[3:4] .= vnew
scatter!(x[odd], x[even], color = (:green, 0.5))
current_figure()
println("still at the boundary? $(norm2(x[1:2] - x[3:4])  - ϵ^2 < 1e-7)")  

# teleport back
xnew, vnew = -x[3:4] + 2*x[1:2], v[3:4]
x[3:4] .= xnew 
v[3:4] .= vnew
v .= circle_boundary_reflection!(1, x, v, ϵ, 2)

# either standard reflection, or bounce at the boundary or traverse the boundary
function circle_hit!(i, x, v, ϵ; α =nothing)
    bounce = true
    d = 2
    ii = 2*i+1:2*(i+1)  
    if abs(norm2(x[1:2] - x[ii])  - ϵ^2) > 1e-7 # make sure to hit be on the circle
        error("not at the boundary. distance equal to $(abs(norm(x[1:2] - x[ii])  - ϵ))")
    end
    if α == nothing
        error("should not be here")
        v .= circle_boundary_reflection!(i, x, v, ϵ, d)
    else
        xnew= -x[ii] + 2*x[1:d]
        disc =  0.5 # magnitude of the discontinuity
        if  disc < 0.0 || rand() > 1 - exp(-disc) # teleport
            # jump on the other side drawing a line passing through the center of the ball
            x[ii] .= xnew 
            v[ii] .= vnew 
            bounce = false
        else    # bounce off 
            v .= circle_boundary_reflection!(i, x, v, ϵ, d)
        end
    end
    return x, v, bounce
end








