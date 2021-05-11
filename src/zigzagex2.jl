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

T = 100.0
d = 2
seed = (UInt(1),UInt(1))

# maybe new relfects and new rand_reflect? 
using ZigZagBoomerang: next_rand_reflect, reflect!


# coefficients of the quadratic equation coming for the condition \|x + θ*t - μ\|^2 > rsq
function abc_eq2d(x, θ, μ, rsq)
    a = θ^2[1] + θ^2[2]
    b = 2*((x[1] -μ[2]) *θ[1] + (x[2] - μ[2])*θ[2]) 
    c = (x[1]-μ[1])^2 + (x[2]-μ[2])^2 - rsq 
    a, b, c
end

# joint reflection at the boundary
function boundary_reflection!(x, θ, μ, rsq)
    for i in eachindex(x[1:end-1])
        if θ[i]*(x[i]-μ[i])>0 
             θ[i]*=-1
        end 
    end
    θ
end

# choose center of ball and squared radius
# μ = [0.0, 0.0]
# rsq = 2.0
# abc_eq2d(x, θ) = abc_eq2d(x, θ, μ, rsq) 
# boundary_reflection!(x, v) = boundary_reflection!(x, v, μ, rsq)

function next_circle_hit(j, i, t′, u, P::SPDMP, μ, rsq, args...) 
    t, x, θ, c, t_old, b = components(u)
    G, G1, G2 = P.G, P.G1, P.G2
    t_old[j] = t′
    if j <= length(x) # standard reflection time
        # b[j] = ab(G1, j, x, θ, c, F)
        # new_time = poisson_time(b[j], rand(P.rng))
        
    else #hitting time to the ball with radius `radius`
        a1,a2,a3 = abc_eq2d(x, θ, μ, rsq) #solving quadradic equation
        dis = a2^2 - 4a1*a3 #discriminant
        if dis < 0.0 # no solutions
            new_time = Inf 
        else #pick the first event time
            new_time = min((-a2 - sqrt(dis))/2*a1,(-a2 + sqrt(dis))/2*a1) #hitting time
        end 
    end
    return 0, t[j] + new_time
end

# either standard reflection, or bounce at the boundary or traversing the boundary
function reflect_traverse!(i, t′, u, P::SPDMP, μ, rsq, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, t_old, b = components(u)
    if i == 3 #hiting boundary
        @assert (x[1] - μ[1] + θ[1]*(t′ - t[1]))^2 + (x[2] - μ[2] + θ[2]*(t′ - t[2]))^2  - rsq < 1e-7 # make sure to hit be on the circle
        smove_forward!(G, i, t, x, θ, m, t′, F)
        t[i] = t′
        disc =  ϕ(x) - ϕ(-x + 2c) # magnitude of the discontinuity
        if  disc < 0.0 || rand() > exp(-disc) # traverse the ball
                # jump on the other side drawing a line passing through the center of the ball
                x .= -x + 2 .*μ
        else    # bounce off 
                θ = boundary_reflection!(x,v)    
        end
        return true, neighbours(G1, i)
    else 
        return rand_reflect!(i, t′, u, P::SPDMP, args...)
    end
end

Random.seed!(1)

using SparseArrays

# last row with zeros (fake coordinate)
Γ = sparse(Matrix([1.0 0.0 0.0; 
                    0.0 1.0 0.0;
                    0.0 0.0 0.0]))


ϕ(x) =  -0.5*x'*Γ*x  # negated log-density
∇ϕ(x, i, Γ) = Zig.idot(Γ, i, x) # sparse computation

# t, x, θ, c, t_old, b 
t0 = 0.0
t = fill(t0, 3)
x = rand(3) 
θ = θ0 = rand([-1.0, 1.0], 3)
F = ZigZag(Γ, x*0)


c = (zero(x) .+ 0.1) .*(eachindex(x) .!= 3)
G = [[i] => [rowvals(F.Γ)[nzrange(F.Γ, i)]..., 3] for i in eachindex(θ0[1:end-1])]
push!(G, [3] => [1, 2, 3])
G1 = G
G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]

b = [ab(G1, i, x, θ, c, F) for i in eachindex(θ)]
  
u0 = StructArray(t=t, x=x, θ=θ, c=c, t_old=copy(t), b=b)


rng = Rng(seed)
t_old = copy(t)
adapt = false
factor = 1.7
P = SPDMP(G, G1, G2, ∇ϕ, F, rng, adapt, factor)

action! = (reflect_traverse!)
next_action = FunctionWrangler((next_event))
