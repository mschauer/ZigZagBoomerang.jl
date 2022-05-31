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
struct PDMPTrace{FT,T,S,S2,S3,R} <: Trace
    F::FT
    t0::T
    x0::S
    θ0::S2
    f::S3
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
Trace(t0::T, x0, θ0, F::Union{ZigZag,FactBoomerang,JointFlow}) where {T} = FactTrace(F, t0, x0, θ0, Tuple{T,Int,eltype(x0),eltype(θ0)}[])
Trace(t0::T, x0::U, θ0::U2, F::Union{BouncyParticle,Boomerang}) where {T, U, U2} = PDMPTrace(F, t0, x0, θ0, ones(Bool, length(x0)), Tuple{T,U,U2,Nothing}[])
Trace(t0::T, x0::U, θ0::U2, f::U3, F::Union{BouncyParticle,Boomerang}) where {T, U, U2, U3} = PDMPTrace(F, t0, x0, θ0, f, Tuple{T,U,U2,U3}[])

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
    t, x, θ, _ = FT.events[k]
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
    discretize(trace::Trace, dt)

Discretize `trace` with step-size dt. Returns iterable object
iterating pairs `t => x`.

Iteration changes the vector `x` inplace,
`collect` creates necessary copies.
"""
discretize(FT, dt) = Discretize(FT, dt)

function Base.iterate(D::Discretize{<:FactTrace})
    FT = D.FT
    t, x, θ = FT.t0, copy(FT.x0), copy(FT.θ0)
    t => x, (t, x, θ, 1)
end

function Base.iterate(D::Discretize{<:PDMPTrace})
    FT = D.FT
    t, x, θ, f = FT.t0, copy(FT.x0), copy(FT.θ0), copy(FT.f)
    t => x, (t, x, θ, f, 1)
end

function Base.iterate(D::Discretize{<:FactTrace{T}}, (t, x, θ, k)) where {T<:Union{FactBoomerang,ZigZag,JointFlow}}
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


#function Base.iterate(D::Discretize{<:PDMPTrace{T}}, (t, x, θ, f, k)) where {T<:Union{Boomerang,BouncyParticle}}
function Base.iterate(D::Discretize{<:PDMPTrace}, (t, x, θ, f, k))
    dt = D.dt
    FT = D.FT
    while true
        k > length(FT.events) && return nothing
        tn = first(FT.events[k])
        if t + dt < tn
            t, x, θ = smove_forward!(dt, t, x, θ, f, FT.F)
            return t => x, (t, x, θ, f, k)
        else # move not more than to ti to change direction
            Δt = tn - t
            dt = dt - Δt
            t, xnew, θ, f_ = FT.events[k]
            x .= xnew
            if !isnothing(f_)
                f = f_
            end
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

export inclusion_prob

function inclusion_prob(trace::ZigZagBoomerang.Trace)
    x = copy(trace.x0)
    θ = copy(trace.θ0)
    y = 0*x
    T = trace.events[end][1]
    t2 = trace.t0
    t = fill(t2, length(x))
    k = 1
    while k <= length(trace.events)
        t2, i, xi, θi = trace.events[k]
        k += 1
        y[i] += (x[i] ≠ 0 | xi ≠ 0)*(t2-t[i])/T
        t[i] = t2
        x[i] = xi
        θ[i] = θi
    end
    y
end



function Statistics.mean(trace::ZigZagBoomerang.Trace)
    x = copy(trace.x0)
    θ = copy(trace.θ0)
    y = 0*x
    T = trace.events[end][1]
    t2 = trace.t0
    t = fill(t2, length(x))
    k = 1
    scale = 1/(2T)
    while k <= length(trace.events)
        t2, i, xi, θi = trace.events[k]
        k += 1
        y[i] += (x[i]+xi)*(t2-t[i])*scale
        t[i] = t2
        x[i] = xi
        θ[i] = θi
    end
    y
end


function cummean(trace::FactTrace)
    x = copy(trace.x0)
    θ = copy(trace.θ0)
    y = 0*x
    T = trace.events[end][1]
    t2 = trace.t0
    ys = [(t=[t2], y=[xi]) for xi in x]
  
 
    t = fill(t2, length(x))
    k = 1
    while k <= length(trace.events)
        t2, i, xi, θi = trace.events[k]
        k += 1
        y[i] += (x[i]+xi)*(t2-t[i]) 
        t[i] = t2
        x[i] = xi
        θ[i] = θi
        push!(ys[i].t, t[i])
        push!(ys[i].y, y[i]/(2t[i]))
    end
    ys
end
export cummean


function Statistics.mean(trace::PDMPTrace)
    x = copy(trace.x0)
    θ = copy(trace.θ0)
    y = 0*x
    ys = fill(y, 0)
    T = trace.events[end][1]
    t2 = trace.t0
    t = t2
    k = 1
    while k <= length(trace.events)
        t2, x2, _ =trace.events[k]
        k += 1
        y  += (x + x2)*(t2-t) 
        t = t2
        x  = x2
    end
    y/T
end

function cummean(trace::PDMPTrace)
    x = copy(trace.x0)
    θ = copy(trace.θ0)
    y = 0*x
    ys = fill(y, 0)
    T = trace.events[end][1]
    t2 = trace.t0
    t = t2
    k = 1
    while k <= length(trace.events)
        t2, x2, _ =trace.events[k]
        k += 1
        y  += (x + x2)*(t2-t) 
        t = t2
        x  = x2
        push!(ys, y/(2t))
    end
    ys
end


"""
    subtrace(tr, J)

Compute the trace of a subvector `x[J]`,
returns a trace object.
"""
function subtrace(tr, J)
    @assert issorted(J)
    F = tr.F  
    t0 = tr.t0
    x0 = collect(tr.x0[j] for j in J)
    θ0 = collect(tr.θ0[j] for j in J)

    str = Trace(t0, x0, θ0, F)
    events = str.events
    for ev in tr.events 
        r = searchsorted(J, ev[2])
        isempty(r) && continue
        push!(events, (ev[1], r[1], ev[3:end]...))
    end
    str
end