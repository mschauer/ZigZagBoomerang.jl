using ZigZagBoomerang
using Test
using Statistics
using Random

using ZigZagBoomerang

Random.seed!(1)
# negative log-density with respect to Lebesgue
# ϕ(x) =  (x - π)^2/2 # not needed

# gradient of ϕ(x)
∇ϕ(x) = x - π


x0, θ0 = 0.01, 1.0
T = 5000.0
out1 = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 10.0, ZigZag())
out2 = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 4.0, Boomerang(0.5))



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
    traj = ZigZagBoomerang.discretization(out2, Boomerang(NaN), dt)
    est = mean(traj.x)
    @test abs(est-pi) < 2/sqrt(length(out2))
end
