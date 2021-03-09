
Random.seed!(1)
const d = 2
const Point = SArray{Tuple{d},Float32,1,d}
using Makie
T = (0:0.001f0:2)[2:end]
n = 200
σ = 0.01f0
μ = 0.2f0
a = 4*0.8f0
b = 4*0.15f0
B = sparse((-μ - a + b)*I, n, n)
B1 = sparse(-1f0*I, n, n)
B2 = sparse(1f0*I, n, n)

for i in 1:n
    k1 = rand(1:n-1)
    k1 += (i <= k1)
    k2 = rand(1:n-2)
    k2 += (min(i, k1) <= k2)
    k2 += (max(i, k1) <= k2)
    @assert i ≠ k1
    @assert i ≠ k2
    @assert k1 ≠ k2

    B[i, k1] = a
    B[i, k2] = -b

    B1[i,k1] = 1
    B2[i, k2] = -1

end
#x0 = 0.1*randn(Point, n)
x = copy(x0)
xs = [x]
ts = [0.0]
t = 0.0
xp = Node(x)
r = .5
p1 = scatter(xp, markersize=3, color=1:n)
xlims!( -r, r)
ylims!( -r, r)

display(p1)
nlz(x) = x/norm(x)
for i in eachindex(T)
    global t, x, xs
    local dt = T[i] - t
    @assert dt > 0
    x = x + (B*x)*dt + σ*sqrt(dt)*randn(Point, n)
     xp[] = x
    t = T[i]
    yield()
    sleep(0.0001)
    push!(ts, t)
    push!(xs, copy(x))
end
error("stop")
p1 = scatter(xp, markersize=3, color=1:n)
xlims!( -r, r)
ylims!( -r, r)
using Base.Iterators: take
framerate=30
record(p1, "boids_animation.mp4", 1:200; framerate = framerate) do i
    xp[] = xs[i*10]
end