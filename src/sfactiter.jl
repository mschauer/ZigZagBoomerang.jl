import Base.iterate
export FactSampler, trace
abstract type PDMPSampler
end

struct FactSampler{TF,T∇ϕ,Tc,Tu0,Targs} <: PDMPSampler
    F::TF
    ∇ϕ::T∇ϕ
    c::Tc
    u0::Tu0

    args::Targs

    factor::Float64
    structured::Bool
    adapt::Bool
end

function FactSampler(∇ϕ, u0, c, F::Union{ZigZag,FactBoomerang}, args...;
    factor=1.8, structured=false, adapt=false)
    return FactSampler(F, ∇ϕ, c, u0, args, factor, structured, adapt)
end

function iterate(FS::FactSampler)
    t0, (x0, θ0) = FS.u0
    F = FS.F
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    t_old = copy(t)
    G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    G2 = [i => setdiff(union((G[j].second for j in G[i].second)...), G[i].second) for i in eachindex(G)]
    x, θ = copy(x0), copy(θ0)
    num = acc = 0
    Q = SPriorityQueue{Int,Float64}()
    b = [ab(G, i, x, θ, FS.c, F) for i in eachindex(θ)]
    for i in eachindex(θ)
        enqueue!(Q, i => poisson_time(b[i], rand()))
    end
    if hasrefresh(F)
        for i in eachindex(θ)
            enqueue!(Q, (n + i) => waiting_time_ref(F))
        end
    end
    iterate(FS, ((t => (x, θ)), t_old, (acc, num), Q, b, G, G2))
end


function iterate(FS::FactSampler, (u, t_old, (acc, num), Q, b, G, G2))
    t, (x, θ) = u
    n = length(x)
    t, (x, θ) = u
    ev, t, x, θ, t′, (acc, num), _,  b, t_old = spdmp_inner!(G, G2, FS.∇ϕ, t, x, θ, Q,
    FS.c, b, t_old, (acc, num), FS.F, FS.args...; structured=FS.structured, factor=FS.factor, adapt=FS.adapt)
    u = t => (x, θ)
    return (t′ => ev), (u, t_old, (acc, num), Q, b, G, G2)
end

function trace(FS::FactSampler, T)
    t0, (x0, θ0) = FS.u0
    Ξ = Trace(t0, x0, θ0, FS.F)
    ϕ = iterate(FS)
    ϕ === nothing && error("No events")
    ev, state = ϕ
    while true
        ϕ = iterate(FS, state)
        ϕ === nothing && return Ξ
        (t, ev), state = ϕ
        t > T && return Ξ
        push!(Ξ, ev)
    end
end
