using Makie

fig = Figure(resolution = (1200, 900))

ax = Axis(fig[1, 1])
ax2 = Axis(fig[2, 1])

sl_x = Slider(fig[3, 1], range = 1:length(xs), startvalue = 1)

r = 16384-100:16384+100
line = @lift xs[$(sl_x.value)][r]
line2 = @lift xs2[$(sl_x.value)][r]

lines!(ax, r, line, color = :blue)
lines!(ax2, r, line2, color = :black, linewidth=0.5)

limits!(ax, extrema(r)..., -20, 20)
limits!(ax2, extrema(r)..., -20, 20)

fig

