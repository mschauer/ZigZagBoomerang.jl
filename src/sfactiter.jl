import Base.iterate
export FactSampler, trace


struct FactSampler{TF,T∇ϕ,Tc,Tu0,TG,Trng,Targs} <: PDMPSampler
    F::TF
    ∇ϕ::T∇ϕ
    c::Tc
    u0::Tu0
    G::TG
    rng::Trng

    args::Targs

    factor::Float64
    adapt::Bool
end

function FactSampler(∇ϕ, u0, c, G, F::Union{ZigZag,FactBoomerang}, args...;
    factor=1.8, adapt=false, seed=Seed())
    return FactSampler(F, ∇ϕ, c, u0, G, Rng(seed), args, factor, adapt)
end
function FactSampler(∇ϕ, u0, c, F::Union{ZigZag,FactBoomerang}, args...;
    factor=1.8, adapt=false)
    return FactSampler(F, ∇ϕ, c, u0, nothing, Rng(), args, factor, adapt)
end

function iterate(FS::FactSampler)
    t0, (x0, θ0) = FS.u0
    F = FS.F
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    t_old = copy(t)
    G1 = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    G = FS.G
    if G === nothing
        G = G1
    end
    @assert all(a.second ⊇ b.second for (a,b) in zip(G, G1))
    G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]
    x, θ = copy(x0), copy(θ0)
    num = 0
    acc = zeros(Int, n)
    Q = SPriorityQueue{Int,Float64}()
    b = [ab(G1, i, x, θ, FS.c, F) for i in eachindex(θ)]
    for i in eachindex(θ)
        enqueue!(Q, i => poisson_time(b[i], rand(FS.rng)))
    end
    if hasrefresh(F)
        enqueue!(Q, (n + 1) => waiting_time_ref(F))
    end
    iterate(FS, ((t => (x, θ)), t_old, (acc, num), Q, b, G, G1, G2))
end


function iterate(FS::FactSampler, (u, t_old, (acc, num), Q, b, G, G1, G2))
    t, (x, θ) = u
    n = length(x)
    ev, t, x, θ, t′, (acc, num), _,  b, t_old = spdmp_inner!(FS.rng, G, G1, G2, FS.∇ϕ, t, x, θ, Q,
    FS.c, b, t_old, (acc, num), FS.F, FS.args...; factor=FS.factor, adapt=FS.adapt)
    u = t => (x, θ)
    return (t′ => ev), (u, t_old, (acc, num), Q, b, G, G1, G2)
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
