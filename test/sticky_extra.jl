
using GLMakie
fig1 = fig = Figure()

ax = Axis(fig[1,1], label = "trace")
lines!(ax, ts, getindex.(xs, 1))

ax = Axis(fig[1,2], label = "phase")
lines!(ax, getindex.(xs, 1), getindex.(xs, 2))
