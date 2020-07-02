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
out2 = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 4.0, Boomerang(1.0, 0.1))

@show length(out1)

@testset "ZigZag" begin
    est = 1/T*sum((eventposition.(out1)[1:end-1] + eventposition.(out1)[2:end])/2 .* diff(eventtime.(out1)))
    @test abs(est-pi) < 2/sqrt(length(out1))
end
