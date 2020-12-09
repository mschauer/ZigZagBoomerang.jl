# target a univariate measure μ(dx) ∝ (x-\mu)^2/2\sigma^2 + δ_0 (x) dx
using ZigZagBoomerang
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


ϕ(x) = x^2/2 #not used
∇ϕ(x) = x

# Example: ZigZag
x0, θ0 = -1.0, 1.0
T = 300.0
c = 1.0
k = 1.0
out1, acc = ss_pdmp(∇ϕ, x0, θ0, T, c, k, ZigZag1d())

# Boomerang
function freezing_time(x, θ)
    if θ*x > 0
        return Inf
    else
        return -x/θ
    end
end
out2, acc = ss_pdmp(∇ϕ, x0, θ0, T, c, k, Boomerang1d(0.1))
@show acc
using Makie, CairoMakie
p1 = lines(eventtime.(out1), eventposition.(out1))

## TEST probability of x being 0 euqal to total time divided by time spent on 0
function prob_of_zero(out1)
    k = 0
    freeze =false
    for event in out1
        if freeze == true
            k += event[1]
        end
        if event[3] == 0.0
            freeze = true
            k -= event[1]
        else
            freeze = false
        end
    end
    k/out1[end][1]
end

const μ = 1.0
ϕ(x) = (x-μ)^2/2 #not used
∇ϕ(x) = x - μ
# Example: ZigZag
x0, θ0 = -1.0, 1.0
T = 2000.0
c = 10.01
k = 0.1
@show acc
out1, acc = ss_pdmp(∇ϕ, x0, θ0, T, c, k, ZigZag1d())
using Makie, CairoMakie
p1 = lines(eventtime.(out1), eventposition.(out1), linewidth = 0.3)
prob_of_zero(out1)

f0 = 1/(sqrt(2*pi))*exp(-1/2*μ^2)
1/(k/f0 + 1)
