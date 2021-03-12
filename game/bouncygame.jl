
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using ZigZagBoomerang
const ZZB = ZigZagBoomerang
Pkg.activate(@__DIR__)
cd(@__DIR__)

using ZigZagBoomerang: Trace, sevent, waiting_time_ref, freezing_time!, ab, 
     smove_forward!, grad_correct!, λ, sλ̄, reflect_sticky!, freezing_time,
     refresh_sticky_vel!, sep
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using FileIO
using Statistics
using Makie, AbstractPlotting
using ForwardDiff
const ρ0 = 0.0
const d = 2
const dist = 1.5
const R = 4.0
 r = -R:0.05:R

released = true

function bouncy_inner!(canvas, X, Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
    Flow::Union{BouncyParticle, Boomerang}, κ, args...; strong_upperbounds = false, factor=1.5, adapt=false)

# f[i] is true if i is free
# frez[i] is the time to freeze, if free, the time to unfreeze, if frozen
while true
    global released
 
    !released &&
        if !ispressed(canvas.scene, Keyboard.space) 
            released = true
        end
    remove(x)
        

    X[] = [point(x)]
    sleep(0.005)
    yield()
    tᶠ, i = findmin(tfrez) # could be implemented with a queue
    tt, j = findmin([tref, tᶠ, t′])
    τ = tt - t
    t, x, θ = smove_forward!(τ, t, x, θ, f, Flow)
    # move forward
    if j == 1 # refreshments of velocities
        θ, θf = refresh_sticky_vel!(θ, θf, f, Flow)
        tref = t + waiting_time_ref(Flow) # regenerate refreshment time
        b = ab(x, θ, c, Flow) # regenerate reflection time
        told = t
        t′ = t + poisson_time(b, rand())
        tfrez = freezing_time!(tfrez, t, x, θ, f, Flow)
        for i in eachindex(f) # make function later...
            if !f[i]
                tfrez[i] = t - log(rand())/(κ[i]*abs(θf[i]))
            end
        end
    elseif j == 2 # get frozen or unfrozen in i
        if f[i] # if free
            if abs(x[i]) > 1e-8
                tfrez[i] = t + freezing_time(x[i], θ[i], Flow.μ[i], Flow) # wrong zero of curve 
                error("x[i] = $(x[i]) !≈ 0 at $(tfrez[i])")
            end
            x[i] = -0*θ[i]
            θf[i], θ[i] = θ[i], 0.0 # stop and save speed
            f[i] = false # change tag
            tfrez[i] = t - log(rand())/(κ[i]*abs(θf[i])) # sticky time
            # tfrez[i] = t - log(rand()) # option 2
            if !(strong_upperbounds) #not strong upperbounds, draw new waiting time
                b = ab(x, θ, c, Flow) # regenerate reflection time
                told = t
                t′ = t + poisson_time(b, rand())
            end
        else # is frozen ->  unfreeze
            @assert x[i] == 0 && θ[i] == 0
            θ[i], θf[i] = θf[i], 0.0 # restore speed
            f[i] = true # change tag
            tfrez[i] = t + freezing_time(x[i], θ[i], Flow.μ[i], Flow)
            b = ab(x, θ, c, Flow) # regenerate reflection time
            told = t
            t′ = t + poisson_time(b, rand())
        end
    else #   t′ usual bouncy particle / boomerang step
        ∇ϕx = ∇ϕ!(∇ϕx, x, args...)
        ∇ϕx = grad_correct!(∇ϕx, x, Flow)
        l, lb = λ(∇ϕx, θ, Flow), sλ̄(b, t - told) # CHECK if depends on f
        num += 1
        if ispressed(canvas.scene, Keyboard.space) && released # reflect!
            #rand()*lb <= l 
            acc += 1
            if l > lb
                !adapt && error("Tuning parameter `c` too small.")
                c *= factor
            end
            θ = reflect_sticky!(∇ϕx, x, θ, f, Flow)
            b = ab(x, θ, c, Flow) # regenerate reflection time
            told = t
            t′ = t + poisson_time(b, rand())
            tfrez = freezing_time!(tfrez, t, x, θ, f, Flow)
            released = false
        else # nothing happened
            b = ab(x, θ, c, Flow)
            told = t
            t′ = t + poisson_time(b, rand())
            -R < x[1] < R && -R < x[2] < R && continue
        end
    end
    push!(Ξ, sevent(t, x, θ, f, Flow))
    return t, x, θ, t′, tref, tfrez, told, f, θf, (acc, num), c, b
end

end


function bouncy(canvas, X, ∇ϕ!, t0, x0, θ0, T, c, Flow::Union{BouncyParticle, Boomerang},
    κ, args...;  strong_upperbounds = false, adapt=false, factor=2.0)
    X[] .= [point(x0)]
    t, x, θ, ∇ϕx = t0, deepcopy(x0), deepcopy(θ0), deepcopy(θ0)
    told = t0
    θf = 0*θ # tags
    f = [true for _ in eachindex(x)]
    Ξ = Trace(t0, x0, θ0, f, Flow)
    push!(Ξ, sevent(t, x0, θ0, f, Flow))
    tref = waiting_time_ref(Flow) #refreshment times
    tfrez = zero(x)
    tfrez = freezing_time!(tfrez, t0, x0, θ0, f, Flow) #freexing times
    num = acc = 0
    b = ab(x, θ, c, Flow)
    t′ = t + poisson_time(b, rand()) # reflection time
    while t < T  && -R < x[1] < R && -R < x[2] < R
        t, x, θ, t′, tref, tfrez, told, f, θf, (acc, num), c, b = bouncy_inner!(canvas, X, Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
                Flow, κ, args...; strong_upperbounds = strong_upperbounds, factor = factor, adapt = adapt)
    end
    return Ξ, (t, x, θ), (acc, num), c
end



phi(x, y, rho) =  1/(2*pi*sqrt(1-rho^2))*exp(-0.5*(x^2 + y^2 - 2x*y*rho)/(1-rho^2))
logdensity(x, y) = log(phi(x - dist, y - dist, 0.0) + phi(x + dist, y + dist, 0.0))
potential(xy) = -5*exp(logdensity(xy[1], xy[2]))


function gradϕ(x,i) 
    -ForwardDiff.partials(logdensity(ForwardDiff.Dual{}(x[1], 1.0*(i==1)), ForwardDiff.Dual{}(x[2], 1.0*(i==2))))[]
end

function gradϕ!(y, x) 
    y[1] = gradϕ(x,1)
    y[2] = gradϕ(x,2)
    y
end




κ = [Inf, Inf]
t0 = 0.0
x0 = [1.0, 0.]
θ0 = [1.0, 1.0]
c = 100.0
T = 10000.0
X = Node([point(x0)])

trace, (tT, xT, θT), (acc, num) = sspdmp(gradϕ!, t0, x0, θ0, T, c, BouncyParticle(sparse(I(d)), 0*x0, 0.1), κ; adapt=false)
ts, xs = sep(collect(discretize(trace, T/500)))



global ys = xs
YS = Node(point.(ys))
function remove(x)
    for i in eachindex(ys)
        if norm(x - ys[i]) < 0.3
            deleteat!(ys, i)
            length(ys) == 0 && error("You won")

            YS[] = point.(ys)
            return true
        end
    end
    return false
end
        
 
point(x) = Point3f0(x[1], x[2], 0.2+potential(x))
canvas = Figure(resolution=(1500,1500))
lscene = LScene(canvas[1, 1], scenekw = (camera = cam3d!, raw = false))



surface!(lscene, r, r, [potential((x1, x2)) for x1 in r, x2 in r], show_axis=false)
scatter!(lscene, X, color=:red, markersize=200)
scatter!(lscene, YS, color=:blue, markersize=120)


display(canvas)

sleep(4.0)
trace, (tT, xT, θT), (acc, num) = bouncy(canvas, X, gradϕ!, t0, x0, θ0, T, c, BouncyParticle(sparse(I(d)), 0*x0, 0.0, 0.), κ; adapt=false)



scatter!(lscene, X, color=:red, marker=:xcross, markersize=500)
display(canvas)