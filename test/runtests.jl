using Test
using Statistics
using Random
using LinearAlgebra
using ZigZagBoomerang
using ZigZagBoomerang: poisson_time
Random.seed!(1)
sep(x) = first.(x), last.(x)

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
T = 8000.0
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
B = Boomerang1d(1.0)
out2, _ = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 1.6, B)


@testset "Boomerang1d" begin
    @test T/10 < length(out2) < T*10
    dt = 0.01
    traj = ZigZagBoomerang.discretize(out2, B, dt)
    @test abs(-(extrema(diff(traj.t[1:end÷3]))...)) < 1e-10
    est = mean(traj.x)
    @test abs(est - pi/2) < 10/sqrt(length(out2))
    est2 = var(traj.x)
    @test abs(est2 - σ2) < 50/sqrt(length(out2))
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


# Local ZigZag
using SparseArrays
d = 8
S = 1.3I + 0.5sprandn(d, d, 0.1)
const Γ = S*S'

∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation


@testset "ZigZag" begin

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0, 1.0], d)


    c = .6*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(0.9Γ, x0*0)
    T = 1000.0

    trace, _, acc = @time pdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))

    @show acc[1]/acc[2]

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end


@testset "SZigZag" begin

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0,-0.5,0.5,1.0], d)


    c = .7*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(0.9Γ, x0*0)
    T = 1000.0

    trace, _, acc = @time spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))

    @show acc[1]/acc[2]

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end



@testset "FactBoomerang" begin

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0,1.0], d)

    #Γ0 = sparse(I, d, d)
    c = 10.5*[norm(Γ[:, i], 2) for i in 1:d]
    Γ0 = copy(Γ)
    for i in 1:d
        #Γ0[d,d] = 1
    end
    Z = FactBoomerang(0.9Γ0, x0*0, 0.5)
    T = 3000.0

    trace, _, acc = @time pdmp(∇ϕ, t0, x0, θ0, T, c, Z, Z.Γ)
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))

    @show acc[1]/acc[2]

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2/sqrt(T)
end

@testset "SFactBoomerang" begin

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0,1.0], d)

    #Γ0 = sparse(I, d, d)
    c = 10.5*[norm(Γ[:, i], 2) for i in 1:d]
    Γ0 = copy(Γ)
    for i in 1:d
        #Γ0[d,d] = 1
    end
    Z = FactBoomerang(0.9Γ0, x0*0, 0.5)
    T = 3000.0

    trace, _, acc = @time spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Z.Γ)
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))

    @show acc[1]/acc[2]

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2/sqrt(T)
end

@testset "Boomerang" begin
    t0 = 0.0
    θ0 = randn(d)
    x0 = randn(d)
    c = 5.0
    Γ0 = copy(Γ)
    B = Boomerang(Γ0, x0*0, 0.5)
    ∇ϕ!(y, x) = mul!(y, Γ, x)
    T = 3000.0
    out, acc = @time pdmp(∇ϕ!, t0, x0, θ0, T, c, B)
    dt = 0.1
    xs = ZigZagBoomerang.discretize(out, B, dt)
    @test mean(abs.(mean(xs.x))) < 2/sqrt(T)
    @test mean(abs.(cov(xs.x) - inv(Matrix(Γ0)))) < 2/sqrt(T)
end

@testset "Bouncy Particle Sampler" begin
    t0 = 0.0
    θ0 = randn(d)
    x0 = randn(d)
    c = 1.0
    Γ0 = copy(Γ)
    B = Bps(Γ0, x0*0, 0.5)
    ∇ϕ!(y, x) = mul!(y, Γ, x)
    T = 3000.0
    out, acc = @time pdmp(∇ϕ!, t0, x0, θ0, T, c, B)
    dt = 0.1
    xs = ZigZagBoomerang.discretize(out, B, dt)
    @test mean(abs.(mean(xs.x))) < 2/sqrt(T)
    @test mean(abs.(cov(xs.x) - inv(Matrix(Γ0)))) < 2/sqrt(T)
end

@testset "ZigZag (independent)" begin

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0, 1.0], d)


    c = 10.0*[norm(Γ[:, i], 2) for i in 1:d]
    Γ0 = sparse(I, d, d)
    Z = ZigZag(Γ0, x0*0)

    T = 1000.0

    trace, _, acc = @time pdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))

    @show acc[1]/acc[2]

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end

@testset "FactBoomerang1" begin
    ϕ(x) = [cos(π*x[1]) + x[1]^2/2] # not needed
    # gradient of ϕ(x)
    ∇ϕ(x) = [-π*sin(π*x[1]) + x[1]]
    ∇ϕ(x, i) = -π*sin(π*x[1]) + x[1] # (REPLACE IT WITH AUTOMATIC DIFFERENTIATION)
    c = [3.5π]
    λref = 1.0
    n = 1
    x0 = randn(n)
    θ0 = randn(n)
    t0 = 0.0
    T = 10000.0
    Γ = sparse(Matrix(1.0I, n, n))
    B = FactBoomerang(Γ, x0*0, λref)
    trace, _,  acc = pdmp(∇ϕ, t0, x0, θ0, T, c, B)
    m = mean(last.(collect(trace)))
    dt = 0.1
    ts, xs = sep(collect(discretize(trace, dt)))
    @show acc[2]/acc[1]
    @test mean(xs)[1] < 2.5/sqrt(T)
end
