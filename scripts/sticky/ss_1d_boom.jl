# target a univariate measure μ(dx) ∝ (x-\mu)^2/2\sigma^2 + δ_0 (x) dx
using ZigZagBoomerang
const ZZB = ZigZagBoomerang

###Boomerang freexing time with μstar = 0.0
function ZZB.freezing_time(x, θ)
    if θ*x >= 0.0
        return π - atan(x/θ)
    else
        return atan(-x/θ)
    end
end
###Boomerang freexing time with μstar = 0.0
function ZZB.freezing_time(x, θ, F::Boomerang1d)
    if F.μ == 0.0
        if x == 0.0
            return 2π
        else
            return ZZB.freezing_time(x, θ)
        end
    elseif abs(F.μ) - sqrt((x - F.μ)^2 + θ^2) >= 0.0
            return Inf
    end
    a = acos(F.μ/(sqrt((x - F.μ)^2 + θ^2)))
    b = atan(θ/(x - F.μ))
    if (x - F.μ) < 0.0
        if x < 0.0  #  x'<0, x<0
            return a + b
        elseif θ > 0 # x'<0, x>0, θ>0
            return 2π - a + b
        else   # x'<0, x>0, θ<0
            return - a + b
        end
    elseif x > 0   # x' > 0, x > 0
        return π - a + b
    elseif θ < 0   # x' > 0, x < 0, θ < 0
        return π + a + b
    else    # x' > 0, x < 0, θ > 0
        return -π + a + b
    end
end

function freezing_time_at0(v, B::Boomerang1d)
    if B.μ == 0.0
        return pi
    end
    a = acos(B.μ/(sqrt((B.μ)^2 + v^2)))
    b = atan(v/(-B.μ))
    if B.μ > 0
        if v > 0
            return 2π - a + b
        else
            return a + b
        end
    elseif v > 0
        return π - a + b
    else
        return π + a + b
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
    tˣ = t + ZZB.freezing_time(x, θ, Flow)
    while t < T
        if  tˣ < min(t_ref, t′) # it s freezing
            t, x, θ = ZZB.move_forward(tˣ - t, t, x, θ, Flow) # go to 0
            @assert -0.0001 < x < 0.0001 #check
            #t, x , θ = tˣ, 0.0, θ #go to 0
            push!(Ξ, (t, x, 0.0))
            t = t - log(rand())/k #wait exponential time
            if t_ref < t
                θ = sqrt(Flow.Σ)*randn()
                t_ref = t +  ZZB.waiting_time_ref(Flow)
            end
            push!(Ξ, (t, x, θ))
            tˣ  = t + freezing_time_at0(θ, Flow)
        elseif t_ref < t′
            t, x, θ = ZZB.move_forward(t_ref - t, t, x, θ, Flow)
            θ = sqrt(Flow.Σ)*randn()
            t_ref = t +  ZZB.waiting_time_ref(Flow)
            tˣ = t + ZZB.freezing_time(x, θ, Flow)
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
                tˣ = t + ZZB.freezing_time(x, θ, Flow)
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
x0, θ0 = -randn(), randn()
T = 1000.0
c = 0.0
k = 0.5
μ = 1.0
ref = 0.5
B = Boomerang1d(μ, ref)
c = 0.0
out1, acc = ss_pdmp(∇ϕ, x0, θ0, T, c, k, B)
out1
dt = 0.1
xx = ZigZagBoomerang.discretize(out1, B, dt)
using Makie, CairoMakie
using Plots
p2 = Makie.lines(xx.t, xx.x, linewidth=0.4)

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
prob_of_zero(out1)



A = zeros(1, 200)
C = zeros(1, 200)
μ, k = 0.0, 0.01
    for i in 1:1
        global μ += 0.0
        for j in 1:200
            global k += 0.03
            x0, θ0 = -randn(), randn()
            T = 100000.0
            c = 0.0
            ref = 0.5
            B = Boomerang1d(μ, ref)
            c = 0.0
            out1, acc = ss_pdmp(∇ϕ, x0, θ0, T, c, k, B)
            A[i,j] = prob_of_zero(out1)
            #C[i,j] = 1/(1 + k*sqrt(2*pi)*exp(0.5*(-μ)^2))
            C[i,j] = 1/(1 + k*sqrt(2pi)*exp(0.5*(-μ)^2)/sqrt(2/pi))
        end
    end
b = sum([(A[1,j] - sum(A[1,:])/200)*(C[1,j] - sum(C[1,:])/200) for j in 1:200])/
            sum([(A[1,j] - sum(A[1,:])/200)^2 for j in 1:200])
a = sum(C[1,:])/200 - sum(A[1,:])/200*b
#b1 =  sum([A[1,j]*C[1,j] for j in 1:100])/
#            sum([(A[1,j])^2 for j in 1:100])
pp = Makie.plot([A[1,j] for j in 1:200], [C[1,j] for j in 1:200], markersize=3)
Makie.lines!(0.1:0.1:1.0, [a + b*x for x in 0.1:0.1:1.0])
#Makie.lines!(0.1:0.1:0.7, [b1*x for x in 0.1:0.1:0.7])
save("figures/boomerang_prob0.png", title(pp, "Boomerang 1d, varying k"))



for i in 2:10
    Makie.plot!(a, [A[i,j] for j in 1:10], [C[i,j] for j in 1:10])
end
a
