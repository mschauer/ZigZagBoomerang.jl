using Revise
using Makie, CairoMakie, ZigZagBoomerang, SparseArrays, LinearAlgebra
using Random

function zz_sticky_events(xs0, i)
    first = true
    y = []
    for x in xs0
        if x[i] == 0.0  && first == true
            push!(y, x)
            first = false
        elseif x[i] == 0.0 && first == false
            continue
        else
            first = true
        end
    end
    return y
end

    Random.seed!(14)
    function ϕ(x, i, μ)
        x[i] - μ[i]
    end
    κ = 2.0
    n = 2
    μ = zeros(n)
    x0 = randn(n)
    θ0 = rand((-1.0,1.0), n)
    T = 10.0
    c = 0.1*ones(n)
    @time trace0, _ = ZigZagBoomerang.sspdmp(ϕ, 0.0, x0, θ0, T, c, κ, ZigZag(sparse(1.0I,n,n), zeros(n)), μ)
    ts0, xs0 = splitpairs(trace0)
    p1  = Scene()
        lines!(getindex.(xs0,1), getindex.(xs0,2), color=(:black, 0.4), markersize=0.1)
        y1 = zz_sticky_events(xs0, 1)
        y2 = zz_sticky_events(xs0, 2)
        scatter!(getindex.(y1,1), getindex.(y1,2),
            color = (:red, 0.3), strokecolor = (:red, 0.3), markersize = 15, marker = 'x')
        scatter!(getindex.(y2,1), getindex.(y2,2),
            color = (:blue, 0.3), strokecolor = (:blue, 0.3), markersize = 15, marker = 'x')
        scatter!([xs0[1][1]], [xs0[1][2]], color = (:black, 1.0), strokecolor = (:black, 1.0), markersize = 2.0)
        text!("x(0),y(0)",textsize = 0.15, position = (xs0[1][1]- 0.05, xs0[1][2] + 0.05))


p2 = lines(ts0, getindex.(xs0,1), color = (:red, 0.5), label = "x1", leg = true)
    lines!(p2, ts0, getindex.(xs0,2), color = (:blue, 0.5), label = "x2")
    xlabel!(p2, "t")
    ylabel!(p2, "")
p = vbox(p1,p2)

save("./figures/2d_sticky_samplers.png", title(p, "2d sticky Zig-Zag"))
