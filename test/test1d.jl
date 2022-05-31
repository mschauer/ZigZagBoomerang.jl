
@testset "1D" begin
Random.seed!(3)
# negative log-density with respect to Lebesgue
# ϕ(x) =  (x - μ)^2/(2σ2) # not needed
# gradient of ϕ(x)
σ2 = 1.3
μ = π/3
∇ϕ(x) = (x - μ)/σ2
∇ϕhat(x) = (x - μ)/σ2 + 0.1(rand()-0.5)


x0, θ0 = 1.01, -1.5
T = 8000.0
out1, _ = ZigZagBoomerang.pdmp(∇ϕhat, x0, θ0, T, 10.0, ZigZag1d())

@testset "ZigZag1d" begin
    @test T/10 < length(out1) < T*10
    est = 1/T*sum((eventposition.(out1)[1:end-1] + eventposition.(out1)[2:end])/2 .* diff(eventtime.(out1)))
    @test abs(est - μ) < 2/sqrt(length(out1))
    dt = 0.01
    traj = ZigZagBoomerang.discretize(out1, ZigZag1d(), dt)
    @test abs(-(extrema(diff(traj.t[1:end÷3]))...)) < 1e-10
    est = mean(traj.x)
    @test abs(est - μ) < 2/sqrt(length(out1))
    est2 = var(traj.x)
    @test abs(est2 - σ2) < 2.5/sqrt(length(out1))
end


x0, θ0 = 1.41, +0.5
B = Boomerang1d(1.0)
out2, _ = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 1.6, B)

@testset "Boomerang1d" begin
    @test T/10 < length(out2) < T*10
    dt = 0.01
    traj = ZigZagBoomerang.discretize(out2, B, dt)
    @test abs(-(extrema(diff(traj.t[1:end÷3]))...)) < 1e-10
    est = mean(traj.x)
    @test abs(est - μ) < 5/sqrt(length(out2))
    est2 = var(traj.x)
    @test abs(est2 - σ2) < 5/sqrt(length(out2))
end

B = Boomerang1d(1.1, 1.2, 0.5)
out2, _ = ZigZagBoomerang.pdmp(∇ϕhat, x0, θ0, T, 10.0, B)


@testset "Boomerang1dNoncentredSub" begin
    @test T/10 < length(out2) < T*10
    dt = 0.01
    traj = ZigZagBoomerang.discretize(out2, B, dt)
    @test abs(-(extrema(diff(traj.t[1:end÷3]))...)) < 1e-10
    est = mean(traj.x)
    @test abs(est - μ) < 5/sqrt(length(out2))
    est2 = var(traj.x)
    @test abs(est2 - σ2) < 5/sqrt(length(out2))
end
end