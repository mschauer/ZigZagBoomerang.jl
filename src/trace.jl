struct ZigZagTrace{T,S,S2,R}
    t0::T
    x0::S
    θ0::S2
    events::R
end

struct FactBoomTrace{T,S,S2,R}
    t0::T
    x0::S
    θ0::S2
    events::R
end


ZigZagTrace(t0::T, x0, θ0) where {T} = ZigZagTrace(t0, x0, θ0, Tuple{T,Int,eltype(x0),eltype(θ0)}[])
FactBoomTrace(t0::T, x0, θ0) where {T} = FactBoomTrace(t0, x0, θ0, Tuple{T,Int,eltype(x0),eltype(θ0)}[])

Base.length(F::Union{ZigZagTrace, FactBoomTrace}) = 1 + length(F.events)




function Base.iterate(F::Union{ZigZagTrace, FactBoomTrace})
    t, x, θ = F.t0, copy(F.x0), copy(F.θ0)
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

function Base.iterate(Z::FactBoomTrace, (t, x, θ, k))
    k > length(Z.events) && return nothing
    k == length(Z.events) && return t => x, (t, x, θ, k + 1)
    t2, i, xi, θi = Z.events[k]
    # x_new = (x .- B.μ)*cos(t2 - t) .+ θ*sin(t2 - t) .+ B.μ
    # θ .= -(x .- B.μ)*sin(t2 - t) .+ θ*cos(t2 - t)
    x_new = (x)*cos(t2 - t) .+ θ*sin(t2 - t)
    θ .= -(x)*sin(t2 - t) .+ θ*cos(t2 - t)
    x .= x_new
    t = t2
    x[i] = xi
    θ[i] = θi
    return t => x, (t, x, θ, k + 1)
end





Base.push!(Z::Union{ZigZagTrace, FactBoomTrace}, ev) = push!(Z.events, ev)
Base.collect(Z::Union{ZigZagTrace, FactBoomTrace}) = collect(t=>copy(x) for (t, x) in Z)

struct Discretize{T, S}
    Z::T
    dt::S
end
Base.IteratorSize(::Discretize) = Iterators.SizeUnknown()
discretize(Z, dt) = Discretize(Z, dt)

function Base.iterate(D::Discretize{T}) where {T <: Union{ZigZagTrace, FactBoomTrace}}
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

function Base.iterate(D::Discretize{<:FactBoomTrace}, (t, x, θ, k))
    dt = D.dt
    Z = D.Z
    while true
        k > length(Z.events) && return nothing
        ti, i, xi, θi = Z.events[k]
        if t + dt < ti
            t += dt
            # x_new = (x .- B.μ)*cos(dt) .+ θ*sin(dt) .+ B.μ
            # θ .= -(x .- B.μ)*sin(dt) .+ θ*cos(dt)
            x_new = (x)*cos(dt) .+ θ*sin(dt)
            θ .= -(x)*sin(dt) .+ θ*cos(dt)
            x .= x_new
            return t => (x), (t, x, θ, k)
        else # move not more than to ti to change direction
            Δt = ti - t
            dt = dt - Δt
            # x_new = (x .- B.μ)*cos(Δt) .+ θ*sin(Δt) .+ B.μ
            # θ .= -(x .- B.μ)*sin(Δt) .+ θ*cos(Δt)
            x_new = (x)*cos(Δt) .+ θ*sin(Δt)
            θ .= -(x)*sin(Δt) .+ θ*cos(Δt)
            x .= x_new
            t = ti
            x[i] = xi
            θ[i] = θi
            k = k + 1
        end
    end
end
function Base.collect(D::Discretize{T}) where {T <: Union{ZigZagTrace, FactBoomTrace}}
    collect(t=>copy(x) for (t, x) in D)
end
