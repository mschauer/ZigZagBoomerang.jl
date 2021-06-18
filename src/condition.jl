
struct ConditionalTrace{T, S}
    FT::T
    plane::S
end
Base.IteratorSize(::ConditionalTrace) = Iterators.SizeUnknown()

"""
    conditional_trace(trace::Trace, (p, n))

ConditionalTrace `trace` on hyperplane `H` through p with norml n. Returns iterable object
iterating pairs `t => x` such that `x ∈ H`.

Iteration changes the vector `x` inplace,
`collect` creates necessary copies.
"""
conditional_trace(FT, f) = ConditionalTrace(FT, f)
function hittingtime(x, θ, (p, n), FT::Union{ZigZag, BouncyParticle})
    dot(p - x, n)/dot(θ, n)
end


function Base.iterate(D::ConditionalTrace{<:FactTrace})
    FT = D.FT
    t, x, θ = FT.t0, copy(FT.x0), copy(FT.θ0)
    iterate(D, (t, x, θ, 1))
end

function Base.iterate(D::ConditionalTrace{<:PDMPTrace})
    FT = D.FT
    t, x, θ, f = FT.t0, copy(FT.x0), copy(FT.θ0), copy(FT.f)
    iterate(D, (t, x, θ, f, 1))
end

function Base.iterate(D::ConditionalTrace{<:FactTrace{T}}, (t, x, θ, k)) where {T<:Union{FactBoomerang,ZigZag,JointFlow}}
    (p, n) = D.plane
    FT = D.FT
    while true
        k > length(FT.events) && return nothing
        τ = hittingtime(x, θ, (p, n), FT.F)
        ti, i, xi, θi = FT.events[k]
        if τ > 0 && t + τ <= ti
            t, x, θ = move_forward!(τ, t, x, θ, FT.F)
            return t => x, (t, x, θ, k)
        else # move to ti to change direction
            t, x, θ = move_forward!(ti - t, t, x, θ, FT.F)
            t = ti
            x[i] = xi
            θ[i] = θi
            k = k + 1
        end
    end
end


function Base.iterate(D::ConditionalTrace{<:PDMPTrace{T}}, (t, x, θ, f, k)) where {T<:Union{Boomerang,BouncyParticle}}
    (p,n) = D.plane
    FT = D.FT
    while true
        k > length(FT.events) && return nothing
        tn = first(FT.events[k])
        τ = hittingtime(x, θ, (p, n), FT.F)
        if τ > 0 && t + τ < tn
            t, x, θ = smove_forward!(τ, t, x, θ, f, FT.F)
            return t => x, (t, x, θ, f, k)
        else # move not more than to ti to change direction
            Δt = tn - t
            t, x, θ = smove_forward!(Δt, t, x, θ, f, FT.F)
            t, x, θ, f_ = FT.events[k]
            if !isnothing(f_)
                f = f_
            end
            k = k + 1
        end
    end
end


function Base.collect(D::ConditionalTrace{<:FactTrace})
    collect(t=>copy(x) for (t, x) in D)
end
function Base.collect(D::ConditionalTrace{<:PDMPTrace})
    collect(t=>copy(x) for (t, x) in D)
end
