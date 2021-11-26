
using GLMakie
fig1 = fig = Figure()
r = 1:200
ax = Axis(fig[1,1], label = "trace")
lines!(ax, ts1[r], getindex.(xs1[r], 1))

ax = Axis(fig[1,2], label = "phase")
lines!(ax, getindex.(xs1[r], 1), getindex.(xs1[r], 2))

ax = Axis(fig[2,1], label = "trace new")
lines!(ax, ts2[r], getindex.(xs2[r], 1))

ax = Axis(fig[2,2], label = "phase new")
lines!(ax, getindex.(xs2[r], 1), getindex.(xs2[r], 2))

ax = Axis(fig[3,1], label = "trace reflect")
lines!(ax, ts3[r], getindex.(xs3[r], 1))

ax = Axis(fig[3,2], label = "phase reflect")
lines!(ax, getindex.(xs3[r], 1), getindex.(xs3[r], 2))
fig