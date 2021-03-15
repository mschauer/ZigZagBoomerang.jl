
using Pkg
#Pkg.activate(joinpath(@__DIR__, ".."))

Pkg.activate(@__DIR__)
cd(@__DIR__)
using Revise
using ZigZagBoomerang
const ZZB = ZigZagBoomerang

using ZigZagBoomerang: Trace, sevent, waiting_time_ref, freezing_time!, ab, 
     smove_forward!, grad_correct!, λ, sλ̄, reflect_sticky!, freezing_time,
     refresh_sticky_vel!, sep
using DataStructures
using LinearAlgebra
#using Observables
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
const dist = 1.
const R = 4.0
 r = -R:0.05:R
#=
mutable struct Level
    story
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
=#
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
    potential
end
ZigZagBoomerang.λ(∇ϕx, θ, F::Union{BouncyParticle, Boomerang}) = ZZB.pos(dot(∇ϕx, θ)) + 0.0

function bouncy_inner!(game, X, Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
    Flow::Union{ZigZag, BouncyParticle, Boomerang}, κ, args...; strong_upperbounds = false, factor=1.5, adapt=false)
    canvas = game.canvas
    factorized = Flow isa ZigZag
    while true
        ispressed(canvas.scene, Keyboard.q) && error("end")
        for (key, pressed) in game.pressed
            if pressed && !ispressed(canvas.scene, key) 
                game.pressed[key] = false
            end
        end
        for (key, released) in game.released
            if released && ispressed(canvas.scene, key) 
                game.released[key] = false
            end
        end

        finished = remove(game, x)
        finished |= ispressed(canvas.scene, Keyboard.enter)
        if ispressed(canvas.scene, Keyboard.a) && !game.pressed[Keyboard.a]
            game.auto[] = !game.auto[]
            game.pressed[Keyboard.a] = true
        end
        X[] = [point(POTENTIAL[], x)]
        sleep(0.002)
        yield()
        tᶠ, i = findmin(tfrez) # could be implemented with a queue
        tt, j = findmin([tref, tᶠ, t′...])
        τ = tt - t
        t, x, θ = smove_forward!(τ, t, x, θ, f, Flow)
        # move forward
        if j == 1 # refreshments of velocities
            if (!game.auto[] && ispressed(canvas.scene, Keyboard.c)) || (game.auto[] && !ispressed(canvas.scene, Keyboard.c))
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
            t′ = t + poisson_times(b)
            tfrez = freezing_time!(tfrez, t, x, θ, f, Flow)
            for i in eachindex(f) # make function later...
                if !f[i]
                    tfrez[i] = t - log(rand())/(κ[i]*abs(θf[i]))
                end
            end
            game.speed[] = norm(θ)
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
                    t′ = t .+ poisson_times(b)
                end
                game.released[Keyboard.s] = false
            elseif !ispressed(canvas.scene, Keyboard.s)  && !game.released[Keyboard.s] # is frozen ->  unfreeze
                @assert x[i] == 0 && θ[i] == 0
                θ[i], θf[i] = θf[i], 0.0 # restore speed
                f[i] = true # change tag
                tfrez[i] = t + freezing_time(x[i], θ[i], Flow.μ[i], Flow)
                b = ab(x, θ, c, Flow) # regenerate reflection time
                told = t
                t′ = t .+ poisson_times(b)
                game.released[Keyboard.s] = true
            else 
                tfrez[i] = t - log(rand())/(κ[i]*abs(θf[i])) # sticky time
            end
        elseif factorized
            i = j - 2

            ∇ϕx = ∇ϕ!(∇ϕx, x, args...)
            l, lb = ZZB.sλ(∇ϕx[i], i, x, θ, Flow), ZZB.sλ̄(b[i], t - told)
            num += 1
            
            left = (θ[1]*θ[2] > 0) ⊻ (i == 2)
            

            if ((game.auto[] || !inbounds(game, x)) && rand()*lb <= l ) ||  ( left && ispressed(canvas.scene, Keyboard.left) && !game.pressed[Keyboard.left]) || 
                ( !left && ispressed(canvas.scene, Keyboard.right) && !game.pressed[Keyboard.right])
                if ispressed(canvas.scene, Keyboard.right)
                    game.pressed[Keyboard.right] = true
                end
                if ispressed(canvas.scene, Keyboard.left)
                    game.pressed[Keyboard.left] = true
                end
                acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small. l/lb = $(l/lb)")
                    acc = num = 0
                    adapt!(c, i, factor)
                end
                θ = ZZB.reflect!(i, ∇ϕx[i], x, θ, Flow)
                b = ab(x, θ, c, Flow)
                told = t
                t′ = t .+ poisson_times(b) 
                tfrez = freezing_time!(tfrez, t, x, θ, f, Flow)    
   #             if !inbounds(game, x)
    #                game.score[] = max(0, game.score[]-1)
     #           end
            else
                b = ab(x, θ, c, Flow)
                told = t
                t′ = t .+ poisson_times(b) 
                finished || continue
            end
        else #   t′ usual bouncy particle / boomerang step
            ∇ϕx = ∇ϕ!(∇ϕx, x, args...)
            ∇ϕx = grad_correct!(∇ϕx, x, Flow)
            l, lb = λ(∇ϕx, θ, Flow), sλ̄(b, t - told) # CHECK if depends on f
            num += 1
            if ((game.auto[] || !inbounds(game, x)) && rand()*lb <= l ) ||  (ispressed(canvas.scene, Keyboard.space) && !game.pressed[Keyboard.space]) # reflect!
                if ispressed(canvas.scene, Keyboard.space)
                    game.pressed[Keyboard.space] = true
                end
                acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ = reflect_sticky!(∇ϕx, x, θ, f, Flow)
                b = ab(x, θ, c, Flow) # regenerate reflection time
                told = t
                t′ = t + poisson_times(b)
                tfrez = freezing_time!(tfrez, t, x, θ, f, Flow)

#                if !inbounds(game, x)
 #                   game.score[] = max(0, game.score[]-1)
  #              end
            else # nothing happened
                b = ab(x, θ, c, Flow)
                told = t
                t′ = t + poisson_times(b)
                finished || continue
            end
        end
        
        push!(Ξ, sevent(t, x, θ, f, Flow))
        return finished, t, x, θ, t′, tref, tfrez, told, f, θf, (acc, num), c, b
    end

end
ZZB.Trace(t0, x0, θ0, f, Flow::ZigZag) = ZZB.Trace(t0, x0, θ0, f, BouncyParticle(Flow.Γ, Flow.μ, 0.0))
function ZZB.sevent(t, x, θ, f, Z::ZigZag)
    t, copy(x), copy(θ), copy(f)
end
ZZB.freezing_time!(tfrez, t0, x0, θ0, f, Flow::ZigZag) = ZZB.freezing_time!(tfrez, t0, x0, θ0, f,  BouncyParticle(Flow.Γ, Flow.μ, 0.0))
function ZZB.ab(x, θ, c, Z::ZigZag)
    a = [c + (ZZB.idot(Z.Γ, i, x)  - ZZB.idot(Z.Γ, i, Z.μ))'*θ[i] for i in eachindex(x)]
    b = [c + θ[i]'*ZZB.idot(Z.Γ, i, θ) for i in eachindex(x)]
    a, b
end
poisson_times((a, b)::Tuple{Array{Float64,1},Array{Float64,1}}) = ZZB.poisson_time.(a, b, rand(length(a)))
poisson_times((a, b)) = ZZB.poisson_time(a, b, rand())

function bouncy(game, X, ∇ϕ!, t0, x0, θ0, T, c, Flow::Union{ZigZag, BouncyParticle, Boomerang},
    κ, args...;  strong_upperbounds = false, adapt=false, factor=2.0)
    factorized = Flow isa ZigZag
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
    t′ = t .+ poisson_times(b) # reflection time
    finished = false 
    while !finished && t < T  && (!game.kill_on_boundary || inbounds(game, 0.9x))
        finished, t, x, θ, t′, tref, tfrez, told, f, θf, (acc, num), c, b = bouncy_inner!(game, X, Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
                Flow, κ, args...; strong_upperbounds = strong_upperbounds, factor = factor, adapt = adapt)
        game.timeleft[] = T - t
    end
    return Ξ, (t, x, θ), (acc, num), c
end



phi(x, y, rho = 0.0) =  1/(2*pi*sqrt(1-rho^2))*exp(-0.5*(x^2 + y^2 - 2x*y*rho)/(1-rho^2))
sphi(x, y, σ1, σ2 = σ1, ρ=0.0) = phi(x/σ1, y/σ2, ρ)/(σ1*σ2)
logdensity1(x, y) = log(sphi(x, y, 0.9))
logdensity2(x, y) = log(0.5sphi(x - dist, y - dist, 0.6) + 0.5sphi(x + dist, y + dist, 0.6))
#potential(xy) = -0.1exp(logdensity(xy[1], xy[2])-log(phi(xy[1], xy[2])))
getpotential(logdensity, boom=false) = function (xy) 
     -exp(logdensity(xy[1], xy[2]) - boom*log(phi(xy[1], xy[2])))
end
getpotentialxy(logdensity, boom=false) = function (x,y) 
    -exp(logdensity(x,y) - boom*log(phi(x, y)))
end

potentials = [getpotentialxy(logdensity1), getpotentialxy(logdensity2)]
potential(x, y) = potentials(POTENTIAL[], x, y)
gradϕ(logdensity) = function (x,i) 
    -ForwardDiff.partials(logdensity(ForwardDiff.Dual{}(x[1], 1.0*(i==1)), ForwardDiff.Dual{}(x[2], 1.0*(i==2))))[]
end

gradϕ!(logdensity, g = gradϕ(logdensity)) = function (y, x) 
    y[1] = g(x,1)
    y[2] = g(x,2)
    y
end
function remove(game, x)
    ys = game.ys[]
    for i in eachindex(ys)
        if norm(x - ys[i]) < 0.15
            deleteat!(ys, i)
            #Makie.notify(game.ys)
            game.ys[] = game.ys[] 
            length(ys) == 0 && return true
            game.score[] = game.score[] + 1
            return false
        end
    end
    return false
end
        

point(x) = Point3f0(x[1], x[2], 0.1+potential(x))
inbounds(x) = -R < x[1] < R && -R < x[2] < R

point(potential, x) = Point3f0(x[1], x[2], 0.1+potential(x[1],x[2]))
point(i::Int, x) = Point3f0(x[1], x[2], 0.1+potentials[i](x[1],x[2]))
inbounds(game, x) = -R < x[1] < R && -R < x[2] < R

κ = [100.0, 100.0]
t0 = 0.0
x0 = [1.0, 0.]
θ0 = [1.0, 1.0]
c = 100.0
T = 1000.0

trace, (tT, xT, θT), (acc, num) = sspdmp(gradϕ!(logdensity2), t0, x0, θ0, T, c, BouncyParticle(sparse(I(d)), 0*x0, 0.1), 0.01*κ; adapt=false)
ts, xs = sep(collect(discretize(trace, T/500)))

X = Node([point(x0)])
score = Node(0)
SCORE = lift(x->"Score "*string(x), score)
speed = Node(norm(θ0))
timeleft = Node(T)
SPEED = lift(x->"Speed "*string(round(x, digits=2)), speed)
TIMELEFT = lift(y->"Time left "*string(round(y, digits=0)),  timeleft)
auto = Node(false)
AUTO = lift(x -> x ? "Autopilot on" : "Manual", auto)
MESSAGE = Node(" "^10*"Get ready - SPACE to reflect"*" "^10*"\n Q to quit")
ys = Node([x for x in xs if inbounds(1.1*x)])
POTENTIAL = Node{Any}(getpotentialxy(logdensity1))
M = lift(p -> [p(x, y) for x in r, y in r], POTENTIAL)
YS = lift((i, ys)->point.(i, ys), POTENTIAL, ys)
ELEMENTS = [TIMELEFT, SPEED, SCORE, AUTO]
MARKER = Node(:circle)
TITLE = Node(" "^20*"Bouncy Particle Sampler Game"*" "^20)
   
canvas = Figure(resolution=(1500,1500))
canvas[2:2,1] =  ax0 = Axis(canvas, aspect=DataAspect())
hidedecorations!(ax0)
canvas[3:10,1] =  ax1 = Axis(canvas, aspect=DataAspect())
hidedecorations!(ax1)
canvas[1,1:5] =  ax = Axis(canvas, title = "SPACE - reflect, Q - quit, S* - stick, C* - Crank-Nicolson, A - auto pilot, +-* - speed (*hold), RETURN - continue to next lvl.", aspect=DataAspect())
text!(ax, TITLE, align=(:center, :bottom), textsize=.85)
text!(ax, MESSAGE, align=(:center, :bottom), textsize=0.65, position=(-0, -2.2))
hidedecorations!(ax)
lscene = LScene(canvas[2:10, 2:5], scenekw = (camera = cam3d!, raw = false))



using Colors
#surface!(lscene, r, r, [potential((x1, x2)) for x1 in r, x2 in r], colormap=:atlantic, show_axis=false)
contour3d!(lscene, r, r, M, levels=30, linewidth=3.0)

scatter!(lscene, X, marker=MARKER, color=:red, markersize=200)
meshscatter!(lscene, YS, color=:orange, markersize=0.05)
#text!(lscene.scene, SCORE, position=(1.5R, 0R, 1), strokecolor=:black, strokewidth=3.0, color=RGB(0.6, 0.2, 0.0), textsize=1.0, rotation= .75pi)
for i in eachindex(ELEMENTS)
    text!(ax1, ELEMENTS[i], strokecolor=:black, strokewidth=.0, color=RGB(0.6, 0.2, 0.0), position=(0, -2*i), textsize=2.0)
end

display(canvas)


println("Find your Makie window and reflect with the SPACE key, quit with Q")
# Continuous trajectories with \nsuch countour reflections for the velocity are characteristic for the BPS.
level = [
    (title="Bouncy particle sampler (BPS)", message="Stear the Bouncy Particle and collect the point. Change direction with a\n 'contour reflection' with SPACE. ", logdensity=logdensity1, F=BouncyParticle(sparse(I(d)), 0*x0, 2.0, 0.995), x0=[0.0,1.0], θ0=[1.0,0.0], T=100.0, ys=[[2.0,2.0]])
    (title="Bouncy particle sampler 2", message="Collect some more posterior points. Continue with RETURN.", logdensity=logdensity1, F=BouncyParticle(sparse(I(d)), 0*x0, 2.0, 0.995), x0=[0.0,1.0], θ0=[1.0,0.0], T=100.0, ys=0.9*[randn(2) for _ in 1:50])
    (title="Bouncy particle sampler 3", message="A double well potential! Collect the points.\nContinue with RETURN or try the autopilot with A.", logdensity=logdensity2, F=BouncyParticle(sparse(I(d)), 0*x0, 2.0, 0.995), x0=[0.0,1.0], θ0=[1.0,0.0], T=100.0, ys=[[0.6*(randn(2)-[dist,dist]) for _ in 1:50];[ 0.6*(randn(2) + [dist,dist]) for _ in 1:50]])
  
    (title="Zig-Zag", message="Try to collect the point with the Zig-Zag. Change direction with the arrow keys. \n Continue with RETURN.", logdensity=logdensity1, F=ZigZag(sparse(I(d)), 0*x0), x0=[0.0,1.0], θ0=[1.0,1.0], T=100.0, ys=[[2.0,2.0]])
    (title="Zig-Zag 2", message="A double well potential! Collect the points.\n Try also A to see how the actual Zig-Zag sampler explores this posterior landscape.", logdensity=logdensity2, F=ZigZag(sparse(I(d)), 0*x0),x0=[0.0,1.0], θ0=[1.0,1.0], T=1000.0, ys=[[0.6*(randn(2)-[dist,dist]) for _ in 1:50];[ 0.6*(randn(2) + [dist,dist]) for _ in 1:50]])

    (title="Boomerang", message="The Boomerang moves on circles and does 'contour reflections' with SPACE. Change your energy with - to reach the center.", logdensity=logdensity1, F=Boomerang(sparse(I(d)), 0*x0, 2.0, 0.995), x0=[0.0,1.0], θ0=[1.0,0.5], T=100.0, ys=[[2.0,2.0], [0.1,-.1]])
    (title="Sticky Bouncy particle sampler", message="Some of the points are on the coordinate axes.\nSTICK to the axis with S to collect them all (release to unstick).", logdensity=logdensity2, F=BouncyParticle(sparse(I(d)), 0*x0, 2.0, 0.995), x0=[0.1,1.0], θ0=[1.0,0.5], T=100.0, ys=[[0.6*rand(Bool,2).*(randn(2)-[dist,dist]) for _ in 1:50];[ 0.6*rand(Bool,2).*(randn(2) + [dist,dist]) for _ in 1:50]])
  
    ]

for i in 7:length(level)
    lvl = level[i]
    ∇ϕ! = gradϕ!(lvl.logdensity)
    x0 = lvl.x0
    θ0 = lvl.θ0
    ys[] = lvl.ys

    game = Game(false, Dict(Keyboard.space => false, Keyboard.s => false, Keyboard.left => false, Keyboard.right => false, Keyboard.a => false), Dict(Keyboard.s => false), ys, canvas, score, speed, timeleft, auto, getpotential(lvl.logdensity))
    POTENTIAL[] = getpotentialxy(lvl.logdensity)
    MARKER[] = :circle
    #sleep(1.5)
    MESSAGE[] = lvl.message
    TITLE[] = lvl.title
    

    trace, (tT, xT, θT), (acc, num) = bouncy(game, X, ∇ϕ!, t0, x0, θ0, lvl.T, c, lvl.F, κ; adapt=false)



    MARKER[] = :xcross

    MESSAGE[] = "$(score[]) posterior\nsamples."
    sleep(2.0)
end