
using ZigZagBoomerang

# negative log-density with respect to Lebesgue
ϕ(x) = cos(2pi*x) + x^2/2 # not needed

# gradient of ϕ(x)
∇ϕ(x) = -2*pi*sin(2*π*x) + x # (REPLACE IT WITH AUTOMATIC DIFFERENTIATION)


# Example: ZigZag
x0, θ0 = randn(), 1.0
T = 100.0
out1 = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 2π, ZigZag())

# Example: Boomerang
out2 = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 2π, Boomerang(0.5))


using Makie
p1 = Makie.lines(eventtime.(out1), eventposition.(out1))
save("zigzag.png", p1)

dt = 0.01
xx = ZigZagBoomerang.discretization(out2, Boomerang(NaN), dt)
p2 = Makie.lines(xx.t, xx.x)
save("boomerang.png", p2)
