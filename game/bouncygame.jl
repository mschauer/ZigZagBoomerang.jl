
using Pkg
#Pkg.activate(joinpath(@__DIR__, ".."))

Pkg.activate(@__DIR__)
cd(@__DIR__)
using ZigZagBoomerang
const ZZB = ZigZagBoomerang

using ZigZagBoomerang: Trace, sevent, waiting_time_ref, freezing_time!, ab, 
     smove_forward!, grad_correct!, λ, sλ̄, reflect_sticky!, freezing_time,
     refresh_sticky_vel!, sep
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using FileIO
using Dates
using Statistics
@time using Makie, AbstractPlotting
println(now())
using ForwardDiff
const ρ0 = 0.0
const d = 2
const dist = 1.5
const R = 4.0
 r = -R:0.05:R

mutable struct Level
    gradϕ!
    logdensity
    potential
    ys
    minpoints
    T
    Target
    User
    θ0
    x0
end

mutable struct Game
    kill_on_boundary
    pressed
    released
    ys
    canvas
    score
    speed
    timeleft
    auto
end

function bouncy_inner!(canvas, X, Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
    Flow::Union{BouncyParticle, Boomerang}, κ, args...; strong_upperbounds = false, factor=1.5, adapt=false)

    while true
        global score
        ispressed(canvas.scene, Keyboard.q) && error("end")
        for (key, pressed) in GAME.pressed
            if pressed && !ispressed(canvas.scene, key) 
                GAME.pressed[key] = false
            end
        end
        for (key, released) in GAME.released
            if released && ispressed(canvas.scene, key) 
                GAME.released[key] = false
            end
        end

        remove(x)
        if ispressed(canvas.scene, Keyboard.a) && !GAME.pressed[Keyboard.a]
            auto[] = !auto[]
            GAME.pressed[Keyboard.a] = true
        end
        X[] = [point(x)]
        sleep(0.002)
        yield()
        tᶠ, i = findmin(tfrez) # could be implemented with a queue
        tt, j = findmin([tref, tᶠ, t′])
        τ = tt - t
        t, x, θ = smove_forward!(τ, t, x, θ, f, Flow)
        # move forward
        if j == 1 # refreshments of velocities
            if ispressed(canvas.scene, Keyboard.c) || ispressed(canvas.scene, Keyboard.a)
                θ, θf = refresh_sticky_vel!(θ, θf, f, Flow)
            elseif ispressed(canvas.scene, Keyboard.equal) 
                θ *= 1.1
                θf *= 1.1
            elseif ispressed(canvas.scene, Keyboard.minus) 
                θ *= 0.9
                θf *= 0.9
            end
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
            speed[] = norm(θ)
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
                GAME.released[Keyboard.s] = false
            elseif !ispressed(canvas.scene, Keyboard.s)  && !GAME.released[Keyboard.s] # is frozen ->  unfreeze
                @assert x[i] == 0 && θ[i] == 0
                θ[i], θf[i] = θf[i], 0.0 # restore speed
                f[i] = true # change tag
                tfrez[i] = t + freezing_time(x[i], θ[i], Flow.μ[i], Flow)
                b = ab(x, θ, c, Flow) # regenerate reflection time
                told = t
                t′ = t + poisson_time(b, rand())
                GAME.released[Keyboard.s] = true
            else 
                tfrez[i] = t - log(rand())/(κ[i]*abs(θf[i])) # sticky time
            end
        else #   t′ usual bouncy particle / boomerang step
            ∇ϕx = ∇ϕ!(∇ϕx, x, args...)
            ∇ϕx = grad_correct!(∇ϕx, x, Flow)
            l, lb = λ(∇ϕx, θ, Flow), sλ̄(b, t - told) # CHECK if depends on f
            num += 1
            if ((auto[] || !inbounds(x)) && rand()*lb <= l ) ||  ispressed(canvas.scene, Keyboard.space) && !GAME.pressed[Keyboard.space] # reflect!
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
                GAME.pressed[Keyboard.space] = true
                if !inbounds(x)
                    score[] = max(0, score[]-1)
                end
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
    while t < T  && (!GAME.kill_on_boundary || inbounds(0.9x))
        t, x, θ, t′, tref, tfrez, told, f, θf, (acc, num), c, b = bouncy_inner!(canvas, X, Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
                Flow, κ, args...; strong_upperbounds = strong_upperbounds, factor = factor, adapt = adapt)
        timeleft[] = T - t
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
function remove(x)
    global score
    for i in eachindex(ys)
        if norm(x - ys[i]) < 0.15
            deleteat!(ys, i)
            length(ys) == 0 && error("You won")
            score[] = score[] + 1
            YS[] = point.(ys)
            return true
        end
    end
    return false
end
        

point(x) = Point3f0(x[1], x[2], 0.1+potential(x))
inbounds(x) = -R < x[1] < R && -R < x[2] < R
while true

κ = [100.0, 100.0]
t0 = 0.0
x0 = [1.0, 0.]
θ0 = [1.0, 1.0]
c = 100.0
T = 10000.0
X = Node([point(x0)])

trace, (tT, xT, θT), (acc, num) = sspdmp(gradϕ!, t0, x0, θ0, T, c, BouncyParticle(sparse(I(d)), 0*x0, 0.1), 0.01*κ; adapt=false)
ts, xs = sep(collect(discretize(trace, T/500)))

global score = Node(0)
global SCORE = lift(x->"Score "*string(x), score)
global speed = Node(norm(θ0))
global timeleft = Node(T)
global SPEED = lift(x->"Speed "*string(round(x, digits=2)), speed)
global TIMELEFT = lift(y->"Time left "*string(round(y, digits=0)),  timeleft)
global auto = Node(false)
global AUTO = lift(x -> x ? "Autopilot on" : "Manual", auto)
global MESSAGE = Node("Get ready\nSPACE to reflect")
global ys = [x for x in xs if inbounds(x)]
global YS = Node(point.(ys))
global ELEMENTS = [TIMELEFT, SPEED, SCORE, AUTO]
canvas = Figure(resolution=(1500,1500))


global GAME = Game(false, Dict(Keyboard.space => false, Keyboard.s => false, Keyboard.a => false), Dict(Keyboard.s => false), ys, canvas, score, speed, timeleft, auto)

canvas[2:2,1] =  ax0 = Axis(canvas, aspect=DataAspect())
text!(ax0, MESSAGE, textsize=2)
hidedecorations!(ax0)
canvas[3:10,1] =  ax1 = Axis(canvas, aspect=DataAspect())
hidedecorations!(ax1)
canvas[1,1:5] =  ax = Axis(canvas, title = "SPACE - reflect, Q - quit, S - stick, C - Crank-Nicolson, A - auto pilot, +- - speed", aspect=DataAspect())
text!(ax, "Bouncy Particle Sampler Game", textsize=2)
hidedecorations!(ax)
lscene = LScene(canvas[2:10, 2:5], scenekw = (camera = cam3d!, raw = false))


using Colors
#surface!(lscene, r, r, [potential((x1, x2)) for x1 in r, x2 in r], colormap=:atlantic, show_axis=false)
contour3d!(lscene, r, r, (x,y)->potential((x,y)), levels=30, linewidth=3.0)

scatter!(lscene, X, color=:red, markersize=200)
meshscatter!(lscene, YS, color=:orange, markersize=0.05)
#text!(lscene.scene, SCORE, position=(1.5R, 0R, 1), strokecolor=:black, strokewidth=3.0, color=RGB(0.6, 0.2, 0.0), textsize=1.0, rotation= .75pi)
for i in eachindex(ELEMENTS)
    text!(ax1, ELEMENTS[i], strokecolor=:black, strokewidth=.0, color=RGB(0.6, 0.2, 0.0), position=(0, -2*i), textsize=2.0)
end

display(canvas)

println("Find your Makie window and reflect with the SPACE key, quit with Q")
sleep(1.0)
println("\nStick to the axes keeping S pressed. Explore different energy levels with + and -")
println("Turn on autopilot with A")

sleep(1.0)
MESSAGE[] = ""
trace, (tT, xT, θT), (acc, num) = bouncy(canvas, X, gradϕ!, t0, x0, θ0, T, c, BouncyParticle(sparse(I(d)), 0*x0, 1.0, 0.97), κ; adapt=false)
#trace, (tT, xT, θT), (acc, num) = bouncy(canvas, X, gradϕ!, t0, x0, θ0, T, c, Boomerang(sparse(I(d)), 0*x0, 0.0, 0.), κ; adapt=false)



scatter!(lscene, X, color=:red, marker=:xcross, markersize=500)

MESSAGE[] = "$(score[]) posterior\nsamples.\n- game over -"
sleep(2.0)
end