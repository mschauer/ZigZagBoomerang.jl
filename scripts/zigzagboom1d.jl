
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
out1, acc = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 1.2π, ZigZag1d())
@show acc

# Example: Boomerang
B = Boomerang1d(0.2)
out2, acc = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 3.5π, B)
@show acc

using Makie
p1 = Makie.lines(eventtime.(out1), eventposition.(out1))
save("figures/zigzag.png", title(p1, "ZigZag 1d"))

dt = 0.01
xx = ZigZagBoomerang.discretize(out2, B, dt)
p2 = Makie.lines(xx.t, xx.x, linewidth=0.4)
save("figures/boomerang.png", title(p2, "Boomerang 1d"))

dt = 0.1
xx = ZigZagBoomerang.discretize(out2, B, dt)
p2 = Makie.lines(xx.t[1:end÷3], xx.x[1:end÷3], linewidth=0.2)
Makie.scatter!(p2, xx.t[1:end÷3], xx.x[1:end÷3], markersize=0.3)
save("figures/boomerangshort.png", title(p2, "Boomerang 1d"))
