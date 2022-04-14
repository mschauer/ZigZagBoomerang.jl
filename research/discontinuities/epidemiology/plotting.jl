using GLMakie, Colors
function plot_forard_model!(fig, k, it, nobs, robs, T, N)
    ax = Axis(fig[1,k])
    ax.limits = (0.9, N+0.1, 0.0-0.1, T+0.1)
    hlines!(ax, [T], color = (:red, 0.5), linestyle = :dot)
    col = distinguishable_colors(N, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    linetype(i::Int64) = i == 1 ? :solid : (i==2 ? :dashdot : :dot )
    for i in 1:N
        if it[i] < T
             lines!(ax, [i, i] , [it[i], nobs[i]],  color = col[i], linestyle = linetype(2), linewidth = 2)
             scatter!(ax,[i], [it[i]], color = col[i], marker = :xcross,  markersize = 20)
             hlines!(ax, it[i], color = (:black, 0.7), linestyle = :dot,  linewidth = 1)
        end
        if nobs[i] < T
             scatter!(ax,[i], [nobs[i]], color = col[i], marker = :star8,  markersize = 20)
             lines!(ax, [i, i] ,[nobs[i], robs[i]], color = col[i], linestyle = linetype(1),  linewidth = 2)    
        end
        if robs[i] < T
            scatter!(ax,[i], [robs[i]], color = col[i], marker = :circle,  markersize = 20)
        end
    end
    return fig
end

function reshape(xx)
    y = zeros(length(xx), length(xx[1]))
    for (i,x) in zip(eachindex(xx), xx)
        y[i, :] = x
    end
    return y
end

function plot_trace_zz!(fig, k, u0, Ξ, N, T)
    xx = getindex.(Ξ,3)
    xtrace = reshape(xx)
    tt = getindex.(Ξ,1)
    clock = tt[end]
    ax1 = Axis(fig[1,k])
    hlines!(ax1, [T], color = (:red, 0.5), linestyle = :dot)
    col = distinguishable_colors(N, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    linetype(i::Int64) = i == 0 || i == 1 ? :solid : (i==2 ? :dashdot : :dot )
    for i in 1:size(xtrace,2)
        if ind0[i] == 0
            continue
        else
            agent = ind0[i] 
            tp = tag0[i]
            lines!(ax1, tt, xtrace[:, i], color = col[agent], linestyle = linetype(tp))
        # scatter!(ax, tt, xtrace[:, i], color = col[agent])
        end
    end
    ax1.limits = (0.0, clock, 0.0-0.1, T+0.1)
    # display(fig)
    return fig
end