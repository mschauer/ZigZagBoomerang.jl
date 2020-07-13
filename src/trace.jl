abstract type Trace end
"""
    FactTrace

See [`Trace`](@ref).
"""
struct FactTrace{FT,T,S,S2,R} <: Trace
    F::FT
    t0::T
    x0::S
    θ0::S2
    events::R
end

"""
    FactTrace

See [`Trace`](@ref).
"""
struct PDMPTrace{FT,T,S,S2,R} <: Trace
    F::FT
    t0::T
    x0::S
    θ0::S2
    events::R
end

"""
    Trace(t0::T, x0, θ0, F::Union{ZigZag,FactBoomerang})

Trace object for exact trajectory of pdmp samplers. Returns an iterable `FactTrace` object.
Note that iteration iterates pairs `t => x` where the vector `x` is modified
inplace, so copies have to be made if the `x` is to be saved.
`collect` applied to a trace object automatically copies `x`.
[`discretize`](@ref) returns a discretized version.
"""
Trace(t0::T, x0, θ0, F::Union{ZigZag,FactBoomerang}) where {T} = FactTrace(F, t0, x0, θ0, Tuple{T,Int,eltype(x0),eltype(θ0)}[])
Trace(t0::T, x0::U, θ0::U2, F::Union{Bps,Boomerang}) where {T, U, U2} = PDMPTrace(F, t0, x0, θ0, Tuple{T,U,U2}[])

Base.length(FT::Trace) = 1 + length(FT.events)

function Base.iterate(FT::FactTrace)
    t, x, θ = FT.t0, copy(FT.x0), copy(FT.θ0)
    t => x, (t, x, θ, 1)
end

function Base.iterate(FT::PDMPTrace)
    t, x, θ = FT.t0, FT.x0, FT.θ0
    t => x, (t, x, θ, 1)
end

function Base.iterate(FT::FactTrace, (t, x, θ, k))
    k > length(FT.events) && return nothing
    k == length(FT.events) && return t => x, (t, x, θ, k + 1)
    t2, i, xi, θi = FT.events[k]
    t, x, θ = move_forward!(t2 - t, t, x, θ, FT.F)
    t = t2
    x[i] = xi
    θ[i] = θi
    return t => x, (t, x, θ, k + 1)
end

function Base.iterate(FT::PDMPTrace, (t, x, θ, k))
    k > length(FT.events) && return nothing
    k == length(FT.events) && return t => x, (t, x, θ, k + 1)
    t, x, θ = FT.events[k]
    return t => x, (t, x, θ, k + 1)
end


Base.push!(FT::Trace, ev) = push!(FT.events, ev)
Base.collect(FT::FactTrace) = collect(t=>copy(x) for (t, x) in FT)
Base.collect(FT::PDMPTrace) = collect(t=>x for (t, x) in FT)

struct Discretize{T, S}
    FT::T
    dt::S
end
Base.IteratorSize(::Discretize) = Iterators.SizeUnknown()

"""
    discretize(trace::FactTrace, dt)

Discretize `trace` with step-size dt. Returns iterable object
iterating pairs `t => x`.

Iteration changes the vector `x` inplace,
`collect` creates copies.
"""
discretize(FT, dt) = Discretize(FT, dt)

function Base.iterate(D::Discretize{<:FactTrace})
    FT = D.FT
    t, x, θ = FT.t0, copy(FT.x0), copy(FT.θ0)
    t => x, (t, x, θ, 1)
end

function Base.iterate(D::Discretize{<:PDMPTrace})
    FT = D.FT
    t, x, θ = FT.t0, copy(FT.x0), copy(FT.θ0)
    t => x, (t, x, θ, 1)
end

function Base.iterate(D::Discretize{<:FactTrace{T}}, (t, x, θ, k)) where {T<:Union{FactBoomerang,ZigZag}}
    dt = D.dt
    FT = D.FT
    while true
        k > length(FT.events) && return nothing
        ti, i, xi, θi = FT.events[k]
        if t + dt < ti
            t, x, θ = move_forward!(dt, t, x, θ, FT.F)
            return t => x, (t, x, θ, k)
        else # move not more than to ti to change direction
            Δt = ti - t
            dt = dt - Δt
            t, x, θ = move_forward!(Δt, t, x, θ, FT.F)
            t = ti
            x[i] = xi
            θ[i] = θi
            k = k + 1
        end
    end
end


function Base.iterate(D::Discretize{<:PDMPTrace{T}}, (t, x, θ, k)) where {T<:Union{Boomerang,Bps}}
    dt = D.dt
    FT = D.FT
    while true
        k > length(FT.events) && return nothing
        tn = first(FT.events[k])
        if t + dt < tn
            t, x, θ = move_forward!(dt, t, x, θ, FT.F)
            return t => x, (t, x, θ, k)
        else # move not more than to ti to change direction
            Δt = tn - t
            dt = dt - Δt
            t, x, θ = FT.events[k]
            k = k + 1
        end
    end
end


function Base.collect(D::Discretize{<:FactTrace})
    collect(t=>copy(x) for (t, x) in D)
end
function Base.collect(D::Discretize{<:PDMPTrace})
    collect(t=>copy(x) for (t, x) in D)
end
