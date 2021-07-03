using Base: Float64
using ZigZagBoomerang
const ZZB = ZigZagBoomerang
### animation 1: no reflection (lebsgue measure)
x0, v0 = -2.0, 1.0
dt = 0.01
# fake trace from -2.0 to +2.0
trace1 = [(t, x0 + t) for t in 0.0:dt:4.0]
# animate: todo

### animation 1b: no reflection (lebsgue measure) with stickyness at 0
x0, v0 = -2.0, 1.0
dt = 0.01
dtinv = Int(1/dt)
κ = 1.0
T = 10.0
wait = floor(-log(rand())/κ*dtinv)/dtinv #  10^-2
# fake trace from -2.0 to +2.0
trace1b = [[(t, x0 + t) for t in 0.0:dt:2.0]..., [(2.0 + t, 0.0)  for t in dt:dt:wait]...,[(2.0 + wait + t, 0.0+ t)  for t in dt:dt:T]... ]
# animate: todo



### animation 2: Boomerang sampler with rereshments but no reflections
B = Boomerang1d(0.0)
t = 0.0
T = 10.0
x = x0
v = v0
out2 = [(0.0, x, v)]
while t < T
    global x, v, t, out
    ref = t -log(rand())/1.0
    τ = min(ref, T)
    t, x, v = ZZB.move_forward(τ, t, x, v, B)
    push!(out2, (t,x,v))
end
trace2 = ZigZagBoomerang.discretize(out2, B, dt)
# animate: todo

### animation 3: Zig-Zag with reflection for a multimodal density
∇ϕ(x) = (x-1.0)/1.0 + (x + 1.0)/1.0
c = 3.0
out3, acc = ZZB.pdmp(∇ϕ, x0, v0, T, 1.2π, ZigZag1d())
trace3 = ZigZagBoomerang.discretize(out3, B, dt)


### animation 4: same as animation 3 + stickiness at 0
# include("C:\\Users\\sebas\\.julia\\dev\\ZigZagBoomerang\\scripts\\sticky\\ss_1d_zigzag.jl")
const ZZB = ZigZagBoomerang
function freezing_time(x, θ)
    if θ*x > 0
        return Inf
    else
        return -x/θ
    end
end

# k determines the weight on the Dirac measure. The smaller k, the higher the wegiht
function ss_pdmp(∇ϕ, x, θ, T, c, k, Flow::ZZB.ContinuousDynamics; adapt=false, factor=2.0)
    t = zero(T)
    Ξ = [(t, x, θ)]
    num = acc = 0
    a, b = ZZB.ab(x, θ, c, Flow)
    t_ref = t + ZZB.waiting_time_ref(Flow)
    t′ =  t + poisson_time(a, b, rand())
    tˣ = t + freezing_time(x, θ)
    while t < T
        if  tˣ < min(t_ref, t′)
            t, x, θ = ZZB.move_forward(tˣ - t, t, x, θ, Flow) # go to 0
            @assert -0.0001 < x < 0.0001 #check
            #t, x , θ = tˣ, 0.0, θ #go to 0
            push!(Ξ, (t, x, 0.0))
            t = t - log(rand())/k #wait exponential time
            push!(Ξ, (t, x, θ))
            tˣ  = Inf
        elseif t_ref < t′
            t, x, θ = ZZB.move_forward(t_ref - t, t, x, θ, Flow)
            θ = sqrt(Flow.Σ)*randn()
            t_ref = t +  ZZB.waiting_time_ref(Flow)
            tˣ = t + freezing_time(x, θ)
            push!(Ξ, (t, x, θ))
        else
            τ = t′ - t
            t, x, θ = ZZB.move_forward(τ, t, x, θ, Flow)
            l, lb = ZZB.λ(∇ϕ, x, θ, Flow), ZZB.λ_bar(τ, a, b)
            num += 1
            if rand()*lb < l
                acc += 1
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ = -θ  # In multi dimensions the change of velocity is different:
                        # reflection symmetric on the normal vector of the contour
                tˣ = t + freezing_time(x, θ)
                push!(Ξ, (t, x, θ))
            end
        end
        a, b = ZZB.ab(x, θ, c, Flow)
        t′ = t + poisson_time(a, b, rand())
    end
    return Ξ, acc/num
end

∇ϕ(x) = (x-1.0)/1.0 + (x + 1.0)/1.0
k = 1.0 # rate of stickying time
c = 3.0
k = 1.0
out4, acc = ss_pdmp(∇ϕ, x0, v0, T, c, k, ZigZag1d())
trace4 = ZigZagBoomerang.discretize(out4, B, dt)

### animation 5: as 4) + boundaries
function hitting_bounds(x, θ, b)
    b1, b2 = b
    if θ*(x-b1) > 0
        return -(x-b2)/θ
    else
        return -(x-b1)/θ
    end
end

function ss_pdmp_b(b, ∇ϕ, x, θ, T, c, k, Flow::ZZB.ContinuousDynamics; adapt=false, factor=2.0)
    b1, b2 = b
    t = zero(T)
    Ξ = [(t, x, θ)]
    num = acc = 0
    a, b = ZZB.ab(x, θ, c, Flow)
    t_ref = t + ZZB.waiting_time_ref(Flow)
    t′ =  t + poisson_time(a, b, rand())
    tˣ = t + freezing_time(x, θ)
    tb = t + hitting_bounds(x, θ, b)
    while t < T
        if  tˣ < min(t_ref, t′, tb)
            t, x, θ = ZZB.move_forward(tˣ - t, t, x, θ, Flow) # go to 0
            @assert -0.0001 < x < 0.0001 #check
            #t, x , θ = tˣ, 0.0, θ #go to 0
            push!(Ξ, (t, x, 0.0))
            t = t - log(rand())/k #wait exponential time
            push!(Ξ, (t, x, θ))
            tˣ  = Inf
        elseif  tb < min(t_ref, t′)
            t, x, θ = ZZB.move_forward(tb - t, t, x, θ, Flow) # go to 0
            #t, x , θ = tˣ, 0.0, θ #go to 0
            @assert (b1-0.0001 < x < b1 + 0.0001 || b2 - 0.0001 < x < b2 + 0.0001 )  #check
            θ > 0.0 ? x = b2 : x = b1
            θ = -θ    
            push!(Ξ, (t, x, θ))
            tˣ  =  t + hitting_bounds(x, θ, b)
        elseif t_ref < t′
            t, x, θ = ZZB.move_forward(t_ref - t, t, x, θ, Flow)
            θ = sqrt(Flow.Σ)*randn()
            t_ref = t +  ZZB.waiting_time_ref(Flow)
            tˣ = t + freezing_time(x, θ)
            push!(Ξ, (t, x, θ))
        else
            τ = t′ - t
            t, x, θ = ZZB.move_forward(τ, t, x, θ, Flow)
            l, lb = ZZB.λ(∇ϕ, x, θ, Flow), ZZB.λ_bar(τ, a, b)
            num += 1
            if rand()*lb < l
                acc += 1
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ = -θ  # In multi dimensions the change of velocity is different:
                        # reflection symmetric on the normal vector of the contour
                tˣ = t + freezing_time(x, θ)
                push!(Ξ, (t, x, θ))
            end
        end
        a, b = ZZB.ab(x, θ, c, Flow)
        t′ = t + poisson_time(a, b, rand())
    end
    return Ξ, acc/num
end

∇ϕ(x) = (x-1.0)/1.0 + (x + 1.0)/1.0
k = 1.0 # rate of stickying time
c = 3.0
k = 1.0
b = (-2.0, 2.0) #buondaries
out5, acc = ss_pdmp_b(∇ϕ, x0, v0, T, c, k, ZigZag1d())
trace5 = ZigZagBoomerang.discretize(out4, B, dt)