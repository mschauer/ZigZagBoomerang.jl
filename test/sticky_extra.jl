
using GLMakie
fig1 = fig = Figure()

ax = Axis(fig[1,1], label = "trace")
lines!(ax, ts, getindex.(xs, 1))

ax = Axis(fig[1,2], label = "phase")
lines!(ax, getindex.(xs, 1), getindex.(xs, 2))

ax = Axis(fig[2,1], label = "trace new")
lines!(ax, ts2, getindex.(xs2, 1))

ax = Axis(fig[2,2], label = "phase new")
lines!(ax, getindex.(xs2, 1), getindex.(xs2, 2))
fig


ax = Axis(fig[3,1], label = "trace reflect")
lines!(ax, ts3, getindex.(xs3, 1))

ax = Axis(fig[3,2], label = "phase reflect")
lines!(ax, getindex.(xs3, 1), getindex.(xs3, 2))
fig