function smove_forward!(G, i, t, x, θ, t′, Z::Union{Bps, ZigZag})
    nhd = neighbours(G, i)
    for i in nhd
        t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
    end
    t, x, θ
end
function smove_forward!(t, x, θ, t′, Z::Union{Bps, ZigZag})
    for i in eachindex(x)
        t[i], x[i] = t′, x[i] + θ[i]*(t′ - t[i])
    end
    t, x, θ
end
function smove_forward!(G, i, t, x, θ, t′, B::Union{Boomerang, FactBoomerang})
    nhd = neighbours(G, i)
    for i in nhd
        τ = t′ - t[i]
        t[i], x[i], θ[i] = t′, (x[i] - B.μ[i])*cos(τ) + θ[i]*sin(τ) + B.μ[i],
                    -(x[i] - B.μ[i])*sin(τ) + θ[i]*cos(τ)
    end
    t, x, θ
end

function event(i, t::Vector, x, θ, Z::Union{ZigZag,FactBoomerang})
    t[i], i, x[i], θ[i]
end


#
# function λ(∇ϕ, i, x, θ, Z::ZigZag, args...)
#     pos(∇ϕ(x, i, args...)*θ[i])
# end
# function ab(G, i, x, θ, c, Z::ZigZag)
#     a = c[i] + θ[i]*(idot(Z.Γ, i, x)  - idot(Z.Γ, i, Z.μ))
#     b = θ[i]*idot(Z.Γ, i, θ)
#     a, b
# end
# λ_bar(G, i, x, θ, c, Z::ZigZag) = pos(ab(G, i, x, θ, c, Z)[1])
# function event(i, t, x, θ, Z::Union{ZigZag,FactBoomerang})
#     t, i, x[i], θ[i]
# end



function spdmp_inner!(Ξ, G, ∇ϕ, t, x, θ, Q, c, (acc, num),
     F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)
    while true
        (refresh, i), t′ = dequeue_pair!(Q)
        t, x, θ = smove_forward!(G, i, t, x, θ, t′, F)
        if refresh
            θ[i] = sqrt(F.Γ[i,i])\randn()
            #renew refreshment
            enqueue!(Q, (true, i)=> t[i] + poisson_time(F.λref, 0.0, rand()))
            #update reflections
            Q[(false, i)] = t[i] + poisson_time(ab(G, i, x, θ, c, F)..., rand())
            for j in neighbours(G, i)
                j == i && continue
                Q[(false, j)] = t[i] + poisson_time(ab(G, j, x, θ, c, F)..., rand())
            end
            push!(Ξ, event(i, t, x, θ, F))
            return t, x, θ, t′, (acc, num), c
        else
            l, lb = λ(∇ϕ, i, x, θ, F, args...), λ_bar(G, i, x, θ, c, F)
            num += 1
            if rand()*lb < l
                acc += 1
                if l >= lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c[i] *= factor
                end
                θ = reflect!(i, θ, x, F)
                for j in neighbours(G, i)
                    Q[(false, j)] = t[j] + poisson_time(ab(G, j, x, θ, c, F)..., rand())
                end
                push!(Ξ, event(i, t, x, θ, F))
                return t, x, θ, t′, (acc, num), c
            end
            enqueue!(Q, (false, i) => t[i] + poisson_time(ab(G, i, x, θ, c, F)..., rand()))
        end
    end
end

function spdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, args...;
        factor=1.5, adapt=false)
    #sparsity graph
    t′ = t0
    t = fill(t′, size(θ0)...)
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    x, θ = copy(x0), copy(θ0)
    num = acc = 0
    Q = PriorityQueue{Tuple{Bool, Int64},Float64}()
    for i in eachindex(θ)
        enqueue!(Q, (false, i)=>poisson_time(ab(G, i, x, θ, c, F)..., rand()))
        if hasrefresh(F)
            enqueue!(Q, (true, i)=>poisson_time(F.λref, 0.0, rand()))
        end
    end
    Ξ = Trace(t0, x0, θ0, F)
    while t′ < T
        t, x, θ, t′, (acc, num), c = spdmp_inner!(Ξ, G, ∇ϕ, t, x, θ, Q, c, (acc, num), F, args...; factor=factor, adapt=adapt)
    end
    #t, x, θ = smove_forward!(t, x, θ, T, F)
    Ξ, (t, x, θ), (acc, num), c
end
