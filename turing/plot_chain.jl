using Makie, GLMakie 
markersize = 1.2
function plot_chain(t, x, FULL=false; skip=0, color=:black, title="")
    r = eachindex(t)[1:1+skip:end]
    fig3 = Figure(resolution=(2000,2000), title=title)
    e = 6
    for i in 1:e^2
        u = CartesianIndices((e,e))[i]
        if u[1] == u[2]
            FULL && Makie.lines(fig3[u[1],u[2]], t, getindex.(x, u[1]),  color=color)
            !FULL && Makie.scatter(fig3[u[1],u[2]], t[r], getindex.(x, u[1])[r], markersize=markersize,  color=color)
        elseif u[1] < u[2]
            FULL && Makie.lines(fig3[u[1],u[2]], getindex.(x, u[1]),  getindex.(x, u[2]), linewidth=markersize, color=(color,1.0))
            !FULL && Makie.scatter(fig3[u[1],u[2]], getindex.(x, u[1])[r],  getindex.(x, u[2])[r], linewidth=markersize, color=(color,1.0), markersize= markersize, strokewidth=0)
        end
    end
    fig3
end
function plot_chain!(fig3, t, x, FULL=false; color=:black, skip=0)
    r = eachindex(t)[1:1+skip:end]
    e = 6
    for i in 1:e^2
        u = CartesianIndices((e,e))[i]
        if u[1] == u[2]
            FULL && Makie.lines!(fig3[u[1],u[2]], t, getindex.(x, u[1]), color=color)
            !FULL && Makie.scatter!(fig3[u[1],u[2]], t[r], getindex.(x, u[1])[r], markersize=markersize, color=color)
        elseif u[1] < u[2]
            FULL && Makie.lines!(fig3[u[1],u[2]], getindex.(x, u[1]),  getindex.(x, u[2]), linewidth=markersize, color=(color,1.0))
            !FULL && Makie.scatter!(fig3[u[1],u[2]], getindex.(x, u[1])[r],  getindex.(x, u[2])[r], linewidth=markersize, color=(color,1.0), markersize= markersize, strokewidth=0)
        end
    end
    fig3
end
