
using GLMakie
fig1 = fig = Figure()
r = 1:10000
ax = Axis(fig[1,1], title = "trace")
lines!(ax, ts1[r], getindex.(xs1[r], 1))

ax = Axis(fig[1,2], title = "phase")
lines!(ax, getindex.(xs1[r], 1), getindex.(xs1[r], 2))

ax = Axis(fig[2,1], title = "trace new")
lines!(ax, ts2[r], getindex.(xs2[r], 1))

ax = Axis(fig[2,2], title = "phase new")
lines!(ax, getindex.(xs2[r], 1), getindex.(xs2[r], 2))

ax = Axis(fig[3,1], title = "trace reflect")
lines!(ax, ts3[r], getindex.(xs3[r], 1))

ax = Axis(fig[3,2], title = "phase reflect")
lines!(ax, getindex.(xs3[r], 1), getindex.(xs3[r], 2))
fig