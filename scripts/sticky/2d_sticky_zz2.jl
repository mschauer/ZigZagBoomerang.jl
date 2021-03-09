using Revise
using CairoMakie, ZigZagBoomerang, SparseArrays, LinearAlgebra
using Random

function zz_sticky_events(ts0, xs0, i)
    first = true
    y = []
    t = Float64[]
    for (s, x) in zip(ts0, xs0)
        if x[i] == 0.0  && first == true
            push!(y, x)
            push!(t, s)
            first = false
        elseif x[i] == 0.0 && first == false
            continue
        else
            first = true
        end
    end
    return t, y
end

Random.seed!(10)
function ϕ(x, i, μ)
    x[i] - μ[i]
end
κ = .5
n = 2
μ = zeros(n)
x0 = [-1.0, -0.35]
θ0 = [1.0, -1.0]

T = 10.0
c = 0.1*ones(n)
@time trace0, _ = ZigZagBoomerang.sspdmp(ϕ, 0.0, x0, θ0, T, c, ZigZag(sparse(1.0I,n,n), zeros(n)), [κ,κ], μ)
ts0, xs0 = splitpairs(trace0)
# AbstractPlotting.available_marker_symbols()

fig  = Figure(resolution = (1000, 500))
p1 = fig[1, 1] = ax1 = Axis(fig, title = "ZigZag")

co = [(:black, 0.5) for i in 1:length(xs0)-1]
#co[8] = co[4] = co[3] = (:red, 0.5)
#co[12] = co[4] = (:blue, 0.5)
fr1 = findall(x->x[1]==0, diff(xs0))
fr2 = findall(x->x[2]==0, diff(xs0))
co[fr1] .=  [(:red, 0.5)]
co[fr2] .=  [(:blue, 0.5)]

dupl(x) = [x[1:end-1]'; x[2:end]'][:] 
segs = dupl(xs0)
#lines!(getindex.(xs0,1), getindex.(xs0,2), color=co, markersize=0.1)
linesegments!(first.(segs), last.(segs), linewidth=1.5,color=co)
arrows!(first.(xs0)[1:1],last.(xs0)[1:1],first.(xs0)[2:2].-first.(xs0)[1:1],last.(xs0)[2:2].-last.(xs0)[1:1], arrowsize = 10.0, lengthscale = 0.5)

t1, y1 = zz_sticky_events(ts0, xs0, 1)
t2, y2 = zz_sticky_events(ts0, xs0, 2)
scatter!(getindex.(y1,1), getindex.(y1,2),
    color = (:red, 0.4), strokecolor = (:red, 0.6), markersize = 15, marker = :star4)
scatter!(getindex.(y2,1), getindex.(y2,2),
    color = (:blue, 0.4), strokecolor = (:blue, 0.6), markersize = 15, marker = :star4)
scatter!([xs0[1][1]], [xs0[1][2]], color = (:black, 1.0), strokecolor = (:black, 1.0), markersize = 5.5)
#text!("x(0), y(0)", textsize = 0.25, position = (xs0[1][1]- 0.05, xs0[1][2] + 0.05))
p1.xlabel = "x"
p1.ylabel = "y"
hidespines!(p1)
#p1.aspect = DataAspect()

p2 = fig[1,2] = ax2 = Axis(fig, title = "Coordinates")
linkyaxes!(ax1, ax2)

co1 = [(:black, 0.5) for i in 1:length(xs0)-1]
co2 = [(:black, 0.5) for i in 1:length(xs0)-1]

fr1 = findall(x->x[1]==0, diff(xs0))
fr2 = findall(x->x[2]==0, diff(xs0))
co1[fr1] .= [(:red, 0.5)]
co2[fr2] .= [(:blue, 0.5)]
#p2.aspect = DataAspect()


times = dupl(ts0)
linesegments!(times, getindex.(segs,1), color = co1, label = "x1", leg = true)
linesegments!(times, getindex.(segs,2), color = co2, linestyle = :dash, label = "x2")

scatter!(t1, getindex.(y1,1),
    color = (:red, 0.4), strokecolor = (:red, 0.6), markersize = 15, marker = :star4)
scatter!(t2, getindex.(y2,2),
    color = (:blue, 0.4), strokecolor = (:blue, 0.6), markersize = 15, marker = :star4)
hidespines!(p2)
p2.xlabel = "t"
p2.ylabel = "x,y"
scatter!(ts0[1:1], first.(xs0[1:1]),
    color = (:red, 0.4), strokecolor = (:red, 1.0), markersize = 5.5)
scatter!(ts0[1:1], getindex.(xs0[1:1],2),
    color = (:blue, 0.4), strokecolor = (:blue, 1.0), markersize = 5.5)



save("./figures/2d_sticky_samplers.png", fig)
fig