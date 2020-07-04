struct ZigZagTrace{T,S,S2,R}
    t0::T
    x0::S
    θ0::S2
    events::R
end
ZigZagTrace(t0::T, x0, θ0) where {T} = ZigZagTrace(t0, x0, θ0, Tuple{T,Int,eltype(x0),eltype(θ0)}[])
Base.length(Z::ZigZagTrace) = 1 + length(Z.events)

function Base.iterate(Z::ZigZagTrace)
    t, x, θ = Z.t0, copy(Z.x0), copy(Z.θ0)
    t => x, (t, x, θ, 1)
end

function Base.iterate(Z::ZigZagTrace, (t, x, θ, k))
    k > length(Z.events) && return nothing
    k == length(Z.events) && return t => x, (t, x, θ, k + 1)
    t2, i, xi, θi = Z.events[k]
    x .+= (t2 - t)*θ
    t = t2
    x[i] = xi
    θ[i] = θi
    return t => x, (t, x, θ, k + 1)
end

Base.push!(Z::ZigZagTrace, ev) = push!(Z.events, ev)
Base.collect(Z::ZigZagTrace) = collect(t=>copy(x) for (t, x) in Z)

struct Discretize{T, S}
    Z::T
    dt::S
end
Base.IteratorSize(::Discretize) = Iterators.SizeUnknown()
discretize(Z, dt) = Discretize(Z, dt)

function Base.iterate(D::Discretize{<:ZigZagTrace})
    Z = D.Z
    t, x, θ = Z.t0, copy(Z.x0), copy(Z.θ0)
    t => x, (t, x, θ, 1)
end

function Base.iterate(D::Discretize{<:ZigZagTrace}, (t, x, θ, k))
    dt = D.dt
    Z = D.Z
    while true
        k > length(Z.events) && return nothing
        ti, i, xi, θi = Z.events[k]
        if t + dt < ti
            t += dt
            x .+= dt .* θ
            return t => (x), (t, x, θ, k)
        else # move not more than to ti to change direction
            Δt = ti - t
            dt = dt - Δt
            x .+= Δt .* θ
            t = ti
            x[i] = xi
            θ[i] = θi
            k = k + 1
        end
    end
end
Base.collect(D::Discretize{<:ZigZagTrace}) = collect(t=>copy(x) for (t, x) in D)
