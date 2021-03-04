using ZigZagBoomerang
using Random
using Makie, CairoMakie, Trajectories
function prob_of_zero(out1, i)
    k = 0.0
    first = true
    t⁻ = 0.0
    for event in out1.events
        if (event[2][i] == 0.0 && event[3][i] == 0.0)
            if first
                t⁻ = event[1]
                first = false
            else
                continue
            end
        elseif (event[2][i] == 0.0 && event[3][i] != 0.0)
            k += event[1] - t⁻
            first = true
        else
            continue
        end
    end
    if (out1.events[end][2][i] == 0.0 && out1.events[end][3][i] == 0.0)
            return (k + out1.events[end][1] - t⁻)/out1.events[end][1]
    else
        return (k)/out1.events[end][1]
    end
end

Random.seed!(1)
∇ϕ!(∇ϕx, x) = x
d = 1
B = BouncyParticle(1.0, d)
t0, x0, θ0 = 0.0, randn(d), randn(d)
T = 50.0
c = 0.001
κ = 1.0
strong_upperbounds = true
adapt = false
out1, uT, acc  = sticky_pdmp(∇ϕ!, t0, x0, θ0, T, c, B,
        κ, strong_upperbounds = strong_upperbounds, adapt = adapt)
dt = 0.1
xx = discretize(out1.events, B, dt)
p1 = Makie.lines(xx.t, getindex.(xx.x,1), linewidth=0.8)
p1 = Makie.lines(getindex.(xx.x, 1), getindex.(xx.x, 2))
save("figures/boomerang_prob0.png", title(p1, "sticky Bps"))


## CHOOSE MY OWN BOUNDS
const ZZ = ZigZagBoomerang
using LinearAlgebra, Trajectories, Random

#####
#Gaussian bounds with buffer
struct Mybound
    c
end

function ZZ.ab(x, θ, c::Mybound, B::BouncyParticle)
    (c.c + dot(x - B.μ, θ), dot(θ, θ))
end

#λ̄ is the same

Random.seed!(1)
d = 100
μ = fill(3.0, d)
λref = 1.0
B = BouncyParticle(I(d), μ, λref, ρ = 0.0)
∇ϕ!(∇ϕx, x) = x - μ
t0, x0, θ0 = 0.0, randn(d), randn(d)
T = 1000.0
c = Mybound(0.001)
κ = 0.5
strong_upperbounds = false
adapt = false
out1, uT, acc  = sticky_pdmp(∇ϕ!, t0, x0, θ0, T, c, B,
        κ, strong_upperbounds = strong_upperbounds, adapt = adapt)
out1.events
dt = 0.1
xx = trajectory(discretize(out1, dt))
p1 = Makie.lines(getindex.(xx.x, 1), getindex.(xx.x, 2), linewidth=0.6, alpha = 0.7)
getindex.(xx.x, 1)
p1 = Makie.lines(xx.t, getindex.(xx.x,1), linewidth=0.8)
pi


### TEST VARYING MU AND KAPPA,
# tp(κ, μ) = 1/(1 + κ*sqrt(2pi)*exp(0.5*(-μ)^2)) # option 1
 tp(κ, μ) = 1/(1 + κ*sqrt(pi/2)*sqrt(2pi)*exp(0.5*(-μ)^2)) # option 2


Random.seed!(0)
c = Mybound(0.0001)
T = 10000.0
λref = 0.01
d = 100
t0, x0, θ0 = 0.0, randn(d), randn(d)
μ1 = 0.0:0.5:2.0
κ = 0.1:0.1:1.0
mean(x) = sum(x)/length(x)
A = zeros(length(μ1), length(κ))
backup = []
for i in eachindex(μ1)
    for j in eachindex(κ)
        global λref, d, T, A, c, μ1, κ, count
        count += 1
        println("lambda ref is : $λref")
        println("μ is : $(μ1[i])")
        μ = deepcopy(fill(μ1[i], d))
        t0, x0, θ0 = 0.0, randn(d), randn(d)
        println("the dimentionality is : $d")
        println("kappa is : $(κ[j])")
        ∇ϕ!(∇ϕx, x) = x - μ
        B = BouncyParticle(I(d), μ, λref, ρ = 0.0)
        out1, uT, acc  = sticky_pdmp(∇ϕ!, t0, x0, θ0, T, c, B,
            κ[j], strong_upperbounds = false, adapt = false)
        prob0 = [prob_of_zero(out1, kk) for kk in eachindex(x0)]
        push!(backup, prob0)
        A[i,j] = mean(prob0)
    end
end

using Makie, CairoMakie, ColorSchemes
mygradcol = ColorSchemes.deep

p1 = Scene()
for i in eachindex(μ1)
    global μ1, κ
    p1 = plot!(κ, [A[i,j] for j in eachindex(κ)], color = mygradcol[Int(floor(i*256/5))], label = "mu = 0.0")
    Makie.lines!(κ, tp.(κ, μ1[i]), color = mygradcol[Int(floor(i*256/5))])
end
p1

# Makie.plot(backup[end])
save("figures/bps_test3_accurate.png", title(p1, "SBPS O2 test, mu = $(μ1[1])-$(μ1[end]), d = $(d), refresh = $(λref)"))


################## DOES NOT WORK WITH/WITHOUT REFRESHMENTS
struct Mybound
    c
end

function ZZ.ab(x, θ, c::Mybound, B::BouncyParticle)
    (c.c + dot(x - B.μ, θ), dot(θ, θ))
end

# dd = 10:100:510
dd = 500
A = zeros(dd)
T = 1000.0
λref = 0.1
μ1 = 0.0
μ = zeros(dd)
κ = 1.0
c = Mybound(0.001)
∇ϕ!(∇ϕx, x) = x - μ
t0, x0, θ0 = 0.0, randn(dd), randn(dd)
B = BouncyParticle(I(dd), μ, λref, ρ = 0.0)
out1, uT, acc  = sticky_pdmp(∇ϕ!, t0, x0, θ0, T, c, B,
    κ, strong_upperbounds = false, adapt = false)
println("done...")
[A[k] = prob_of_zero(out1, k) for k in eachindex(x0)]
p1 = scatter(A)
c = [tp(κ, μ1) for _ in A]
Makie.lines!(eachindex(A), c)
save("figures/bps_test_dim_accurate.png", title(p1, "sticky BPS varying dimensions"))

####################################### WORKS
#### d = 2
c = Mybound(0.001)
T = 10000.0
d = 2
using Random
Random.seed!(0)
μ = fill(3.0, d)
κ = 0.00:0.05:0.2
λref = 1.0
A = zeros(length(κ))
B = BouncyParticle(I(d), μ, λref, ρ = 0.0)
for j in eachindex(κ)
    global λref, d, T, A, c, κ, μ, B
    t0, x0, θ0 = 0.0, randn(d), randn(d)
    ∇ϕ!(∇ϕx, x) = x - μ
    out1, uT, acc  = sticky_pdmp(∇ϕ!, t0, x0, θ0, T, c, B,
        κ[j], strong_upperbounds = false, adapt = false)
    A[j] = sum([prob_of_zero(out1, i) for i in eachindex(x0)])/length(x0)
end


using Makie, CairoMakie, ColorSchemes
mygradcol = ColorSchemes.deep
plot(κ, A)
c = tp.(0.0:0.01:0.2, 3.0)
lines!(0.0:0.01:0.2, c)


######################################### WORKS
Random.seed!(0)
κ = 0.0
t0, x0, θ0 = 0.0, randn(d), randn(d)
∇ϕ!(∇ϕx, x) = x - μ
c = Mybound(0.001)
out1, uT, acc  = sticky_pdmp(∇ϕ!, t0, x0, θ0, T, c, B,
    κ, strong_upperbounds = false, adapt = false)
A = sum([prob_of_zero(out1, i) for i in eachindex(x0)])/length(x0)
dt = 0.1
xx = trajectory(discretize(out1, dt))
p1 = Makie.lines(getindex.(xx.x, 1), getindex.(xx.x, 2), linewidth=0.6, alpha = 0.7)
