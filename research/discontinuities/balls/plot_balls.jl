# plot initial configuration
using CairoMakie
using Makie.GeometryBasics
x0 = deepcopy(x) # initial position
odd = 1:2:2*N-1
even = 2:2:2*N

fig = Figure(backgroundcolor = RGBf0(0.98, 0.98, 0.98),
resolution = (700, 700),)
ax1 = Axis(fig[1,1])
limits!(ax1, -3, 3, -3, 3)
# scatter!(ax1, x[odd], x[even],  marker = '✈', markersize = 20, )
scatter!(ax1, x0[odd], x0[even],  marker = 'o', markersize = 20, color = :black)
poly!(Circle(Point2f0(x[1:2]...), ϵ), color = (:pink,0.4))
Label(fig[1, 1, Top()], "Initial Configuration", valign = :bottom,
    padding = (0, 0, 2, 0))
current_figure()
save("initial_conf.png", fig)

