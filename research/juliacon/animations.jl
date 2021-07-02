using Base: Float64
using ZigZagBoomerang
const ZZB = ZigZagBoomerang
### animation 1: no reflection (lebsgue measure)
x0, v0 = -2.0, 1.0
dt = 0.01
# fake trace from -2.0 to +2.0
trace1 = [x0 + t for t in 0.0:dt:4.0]
# animate: todo


### animation 2: Boomerang sampler with rereshments but no reflections
B = Boomerang1d(0.0)
t = 0.0
T = 10.0
out = [(0.0, x0, v0)]
while t < T
    global x0, v0, t, out
    x = x0
    ref = t -log(rand())/1.0
    τ = min(ref, T)
    t, x, v = ZZB.move_forward(τ, t, x, v, B)
    push!(out, (t,x,v))
end
trace2 = ZigZagBoomerang.discretize(out, B, dt)


