using ZigZagBoomerang, Distributions, ForwardDiff, LinearAlgebra, SparseArrays, StructArrays

# Problem
d = 2
n = 10
xtrue = [-3.0, 3.0]
data = rand(Normal(xtrue[1], xtrue[2]), n)
g(x) = sum(logpdf(Normal(x[1], x[2]), dt) for dt in data)

# Negative partial derivative maker
function negpartiali(f, d)
   id = collect(I(d))
   ith = [id[:,i] for i in 1:d]
   function (x, i, args...)
       sa = StructArray{ForwardDiff.Dual{}}((x, ith[i]))
       δ = -f(sa, args...).partials[]
       return δ
   end
end


d = 2
t0 = 0.0
x0 = [2.0, 5.0]
θ0 = rand([-1.0,1.0], d)
c = 1.0*ones(length(x0))
Z = ZigZag(sparse(I(n)), x0*0);
T = 200.0
κ = 200_000.0

u0 = ZZB.stickystate(x0)
G = [i=>collect(1:d) for i in 1:d]
target = ZZB.StructuredTarget(G, negpartiali(g, d))
barriers = [ZZB.StickyBarriers((-Inf, Inf), (:sticky, :sticky), (κ, κ)) , ZZB.StickyBarriers((0.0, Inf), (:reflect, :sticky), (κ, κ)) ]

flow = ZZB.StickyFlow(Z)
strong = false
c = 20*[1.0 for i in 1:d]
adapt = true
factor = 1.5

T = 2000.0



G1 = [i => [i] for i in 1:d]
G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]
upper_bounds = ZZB.StickyUpperBounds(G1, G2, 1.0sparse(I(d)), strong, adapt, c, factor)
end_time = ZZB.EndTime(T)
trace, _, _, acc = @time ZZB.stickyzz(u0, target, flow, upper_bounds, barriers, end_time)
println("acc ", acc.acc/acc.num)

dt = 0.5
global ts1, xs1 = sep(collect(trace))








using GLMakie
fig1 = fig = Figure()
r = 1:length(ts1)
ax = Axis(fig[1,1], title = "trace")
lines!(ax, ts1[r], getindex.(xs1[r], 1))
lines!(ax, ts1[r], getindex.(xs1[r], 2))

ax = Axis(fig[1,2], title = "phase")
lines!(ax, getindex.(xs1[r], 1), getindex.(xs1[r], 2))

save(joinpath(@__DIR__, "positivity.png"), fig1)
display(fig1)
