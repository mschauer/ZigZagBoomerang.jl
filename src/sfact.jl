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


function spdmp_inner!(Ξ, G, ∇ϕ, t, x, θ, Q, c, (acc, num),
     F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)
    n = length(x)
    while true
        ii, t′ = peek(Q)
        refresh = ii > n
        i = ii - refresh*n
        t, x, θ = smove_forward!(G, i, t, x, θ, t′, F)
        if refresh
            θ[i] = sqrt(F.Γ[i,i])\randn()
            #renew refreshment
            Q[(n + i)] = t[i] + poisson_time(F.λref)
            #update reflections
            for j in neighbours(G, i)
                Q[j] = t[i] + poisson_time(ab(G, j, x, θ, c, F)..., rand())
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
                θ = reflect!(i, x, θ, F)
                for j in neighbours(G, i)
                    Q[j] = t[j] + poisson_time(ab(G, j, x, θ, c, F)..., rand())
                end
                push!(Ξ, event(i, t, x, θ, F))
                return t, x, θ, t′, (acc, num), c
            end
            Q[i] = t[i] + poisson_time(ab(G, i, x, θ, c, F)..., rand())
        end
    end
end

"""
    spdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, args...;
        factor=1.5, adapt=false)

Version of spdmp which assumes that `i` only depends on coordinates
`x[j] for j in neighbours(G, i)`.
"""
function spdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag,FactBoomerang}, args...;
        factor=1.5, adapt=false)
    #sparsity graph
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    x, θ = copy(x0), copy(θ0)
    num = acc = 0
    Q = SPriorityQueue{Int,Float64}()
    for i in eachindex(θ)
        enqueue!(Q, i =>poisson_time(ab(G, i, x, θ, c, F)..., rand()))
    end
    if hasrefresh(F)
        for i in eachindex(θ)
            enqueue!(Q, (n + i)=>poisson_time(F.λref))
        end
    end
    Ξ = Trace(t0, x0, θ0, F)
    while t′ < T
        t, x, θ, t′, (acc, num), c = spdmp_inner!(Ξ, G, ∇ϕ, t, x, θ, Q, c, (acc, num), F, args...; factor=factor, adapt=adapt)
    end
    #t, x, θ = smove_forward!(t, x, θ, T, F)
    Ξ, (t, x, θ), (acc, num), c
end
