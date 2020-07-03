
using ZigZagBoomerang
using Random
Random.seed!(1)
# negative log-density with respect to Lebesgue
ϕ(x) = cos(π*x) + x^2/2 # not needed

# gradient of ϕ(x)
∇ϕ(x) = -π*sin(π*x) + x # (REPLACE IT WITH AUTOMATIC DIFFERENTIATION)


# Example: ZigZag
x0, θ0 = randn(), 1.0
T = 300.0
out1, acc = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 1.2π, ZigZag())
@show acc

# Example: Boomerang
B = Boomerang(0.0, 0.2)
out2, acc = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 3.5π, B)
@show acc

using Makie
p1 = Makie.lines(eventtime.(out1), eventposition.(out1))
save("zigzag.png", p1)

dt = 0.01
xx = ZigZagBoomerang.discretization(out2, B, dt)
p2 = Makie.lines(xx.t, xx.x)
save("boomerang.png", p2)
