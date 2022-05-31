
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
    @test abs(p - pt) < 2/sqrt(n)
end


# negative log-density with respect to Lebesgue
# ϕ(x) =  (x - π)^2/2 # not needed
# gradient of ϕ(x)
const σ2 = 1.3
∇ϕ(x) = (x - π/2)/σ2
∇ϕhat(x) = (x - π/2)/σ2 + 0.1(rand()-0.5)


x0, θ0 = 1.01, -1.5
T = 10000.0
out1, _ = ZigZagBoomerang.pdmp(∇ϕhat, x0, θ0, T, 2.5, ZigZag1d())

@testset "ZigZag1d" begin
    @test T/10 < length(out1) < T*10
    est = 1/T*sum((eventposition.(out1)[1:end-1] + eventposition.(out1)[2:end])/2 .* diff(eventtime.(out1)))
    @test abs(est - pi/2) < 2/sqrt(length(out1))
    dt = 0.01
    traj = ZigZagBoomerang.discretize(out1, ZigZag1d(), dt)
    @test abs(-(extrema(diff(traj.t[1:end÷3]))...)) < 1e-10
    est = mean(traj.x)
    @test abs(est - pi/2) < 2/sqrt(length(out1))
    est2 = var(traj.x)
    @test abs(est2 - σ2) < 2/sqrt(length(out1))
end


x0, θ0 = 1.41, +0.1
B = Boomerang1d(0.1)
out2, _ = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 2.0, B)


@testset "Boomerang1d" begin
    @test T/10 < length(out2) < T*10
    dt = 0.01
    traj = ZigZagBoomerang.discretize(out2, B, dt)
    @test abs(-(extrema(diff(traj.t[1:end÷3]))...)) < 1e-10
    est = mean(traj.x)
    @test abs(est - pi/2) < 5/sqrt(length(out2))
    est2 = var(traj.x)
    @test abs(est2 - σ2) < 11/sqrt(length(out2))
end

B = Boomerang1d(1.1, 1.2, 0.5)
out2, _ = ZigZagBoomerang.pdmp(∇ϕhat, x0, θ0, T, 10.0, B)


@testset "Boomerang1dNoncentredSub" begin
    @test T/10 < length(out2) < T*10
    dt = 0.01
    traj = ZigZagBoomerang.discretize(out2, B, dt)
    @test abs(-(extrema(diff(traj.t[1:end÷3]))...)) < 1e-10
    est = mean(traj.x)
    @test abs(est - pi/2) < 5/sqrt(length(out2))
    est2 = var(traj.x)
    @test abs(est2 - σ2) < 5/sqrt(length(out2))
end
