using GLMakie
fig1 = fig = Figure()
r = 1:min(10000, length(ts1))
ax = Axis(fig[1,1], title = "trace")
lines!(ax, ts1[r], getindex.(xs1[r], 1))

ax = Axis(fig[1,2], title = "phase")
lines!(ax, getindex.(xs1[r], 1), getindex.(xs1[r], 2))
