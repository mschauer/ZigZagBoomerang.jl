using Trajectories
# For plotting: discretize of the circular dynamics

"""
    discretize(x::Vector, Flow::Union{ZigZag1d, Boomerang1d}, dt)

Transform the output of the algorithm (a skeleton of points) to
a trajectory. Simple 1-d version.
"""
function discretise(x::Vector, Flow::Union{ZigZag1d, Boomerang1d}, dt0)
    k = 1
    _, ξ, θ = x[k]
    τ = x[k+1][1]
    clock = 0.0
    Ω = trajectory([clock => ξ])
    dt = dt0
    while k < length(x)-1
        while clock + dt <= τ
            if θ == 0.0
                clock += dt
            else
                clock, ξ, θ = move_forward(dt, clock, ξ, θ, Flow)
            end
            push!(Ω, clock => ξ)
            dt = dt0
        end
        Δt = τ - clock
        dt = dt - Δt
        if θ == 0.0
            clock = τ
        else
            clock, ξ, θ = move_forward(τ - clock, clock, ξ, θ, Flow)
        end
        k += 1
        _, ξ, θ = x[k]
        τ = x[k+1][1]
    end
    push!(Ω, clock => ξ)
    Ω
end

## THIS FUNCTION IS DEPRECATED. See src/trace.jl
"""
    discretize(x::Vector, Flow::Union{BouncyParticle, Boomerang}, dt)

Transform the output of the algorithm (a skeleton of points) to
a trajectory. multi-dimensional version.

Old version that would not work with the sticky Boomerang sampler not centered in 0
"""
function sdiscretise(x::Vector, Flow::Union{BouncyParticle, Boomerang}, dt0)
    k = 1
    _, ξ, θ, f = x[k]
    τ = x[k+1][1]
    clock = 0.0
    Ω = trajectory([clock => deepcopy(ξ)])
    dt = dt0
    while k < length(x)-1
        while clock + dt <= τ
            clock, ξ, θ = smove_forward!(dt, clock, ξ, θ, f, Flow)
            push!(Ω, clock => deepcopy(ξ))
            dt = dt0
        end
        Δt = τ - clock
        dt = dt - Δt
        clock, ξ, θ = smove_forward!(τ - clock, clock, ξ, θ, f, Flow)
        k += 1
        _, ξ, θ, f = x[k]
        τ = x[k+1][1]
    end
    Ω
end
