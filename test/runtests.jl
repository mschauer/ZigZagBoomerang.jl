using ZigZagBoomerang
using Test
using Statistics
using Random
using LinearAlgebra

using ZigZagBoomerang

using ZigZagBoomerang: poisson_time

Random.seed!(1)

# testing poisson time sampler

a, b = 1.1, 0.0
n = 5000
Λ0(a, b, T) = a*T + b*(T)^2/2
P(a, b, T) = 1 - exp(-Λ0(a, b, T))
T = 0.7

for (a, b, pt) in ((1.1, 0.0, NaN), (1.1, 0.3, NaN), (0.0, 0.3, NaN), (1.1, -0.5, NaN),
    (-0.5, 1, P(0, 1, T-0.5)), (-1, -2, 0.0))
    p = mean(poisson_time(a, b, rand()) < T for i in 1:n)
    if isnan(pt)
        pt = P(a, b, T)
    end
#    @show p, pt
    @test abs(p - pt) < 2/sqrt(n)
end


# negative log-density with respect to Lebesgue
# ϕ(x) =  (x - π)^2/2 # not needed

# gradient of ϕ(x)
∇ϕ(x) = x - π

x0, θ0 = 0.01, 1.0
T = 5000.0
out1, _ = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 10.0, ZigZag())
B = Boomerang(2.0, 0.5)
out2, _ = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 4.0, B)

@testset "ZigZag" begin
    @test T/10 < length(out1) < T*10
    est = 1/T*sum((eventposition.(out1)[1:end-1] + eventposition.(out1)[2:end])/2 .* diff(eventtime.(out1)))
    @test abs(est-pi) < 2/sqrt(length(out1))
    dt = 0.01
    traj = ZigZagBoomerang.discretization(out1, ZigZag(), dt)
    est = mean(traj.x)
    @test abs(est-pi) < 2/sqrt(length(out1))

end

@testset "Boomerang" begin
    @test T/10 < length(out2) < T*10
    dt = 0.01
    traj = ZigZagBoomerang.discretization(out2, B, dt)
    est = mean(traj.x)
    @test abs(est-pi) < 2/sqrt(length(out2))
end


# Local ZigZag
using SparseArrays
d = 8
S = I + 0.5sprandn(d, d, 0.1)
Γ = S*S'

G = [i => rowvals(Γ)[nzrange(Γ, i)] for i in 1:d]

∇ϕ(x, i) = dot(Γ[:,i], x) # sparse computation



t0 = 0.0
x0 = rand(d)
θ0 = rand([-1,1], d)


c = 1.01*[norm(Γ[:, i], 2) for i in 1:d]

Z = LocalZigZag(Γ, x0*0)
T = 1000.0

Ξ, _ = @time pdmp(G, ∇ϕ, t0, x0, θ0, T, c, Z)

t, x, θ = deepcopy((t0, x0, θ0))
xs = [x0]
ts = [t0]
dt = 0.5
for ξ in Ξ
    global t, x, θ
    local i
    i, ti, xi = ξ
    while t + dt < ti
        t, x, θ = ZigZagBoomerang.move_forward!(dt, t, x, θ, Z)
        push!(ts, t)
        push!(xs, copy(x))
    end
    r = dt - (ti - t)
    @test r > 0
    t, x, θ = ZigZagBoomerang.move_forward!(ti - t, t, x, θ, Z)
    @test x[i] ≈ xi atol=1e-7
    θ[i] = -θ[i]
    t, x, θ = ZigZagBoomerang.move_forward!(ti - t, t, x, θ, Z)
    push!(ts, t)
    push!(xs, copy(x))
end
@test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 0.08
#display(round.(cov(xs) - inv(Matrix(Γ)), digits=3))
#display(round.(cov(xs), digits=3))
#display(round.( inv(Matrix(Γ)), digits=3))
