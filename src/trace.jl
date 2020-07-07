
struct FactTrace{FT,T,S,S2,R}
    F::FT
    t0::T
    x0::S
    θ0::S2
    events::R
end


Trace(t0::T, x0, θ0, F::Union{ZigZag,FactBoomerang}) where {T} = FactTrace(F, t0, x0, θ0, Tuple{T,Int,eltype(x0),eltype(θ0)}[])

Base.length(FT::FactTrace) = 1 + length(FT.events)

function Base.iterate(FT::FactTrace)
    t, x, θ = FT.t0, copy(FT.x0), copy(FT.θ0)
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





Base.push!(FT::FactTrace, ev) = push!(FT.events, ev)
Base.collect(FT::FactTrace) = collect(t=>copy(x) for (t, x) in FT)

struct Discretize{T, S}
    FT::T
    dt::S
end
Base.IteratorSize(::Discretize) = Iterators.SizeUnknown()
discretize(FT, dt) = Discretize(FT, dt)

function Base.iterate(D::Discretize{<:FactTrace})
    FT = D.FT
    t, x, θ = FT.t0, copy(FT.x0), copy(FT.θ0)
    t => x, (t, x, θ, 1)
end

function Base.iterate(D::Discretize{<:FactTrace{T}}, (t, x, θ, k)) where {T<:ZigZag}
    dt = D.dt
    FT = D.FT
    while true
        k > length(FT.events) && return nothing
        ti, i, xi, θi = FT.events[k]
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

function Base.iterate(D::Discretize{<:FactTrace{T}}, (t, x, θ, k)) where {T<:FactBoomerang}
    dt = D.dt
    FT = D.FT
    while true
        k > length(FT.events) && return nothing
        ti, i, xi, θi = FT.events[k]
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
function Base.collect(D::Discretize{<:FactTrace})
    collect(t=>copy(x) for (t, x) in D)
end
