using Trajectories
# For plotting: discretization of the circular dynamics

"""
    discretization(x::Vector{Skeleton}, Flow::Boomerang, dt)

Tansform the output of the algorithm (a skeleton of points) to
a trajectory.
"""
function discretization(x::Vector, Flow, dt0)
    k = 1
    _, ξ, θ = x[k]
    τ = x[k+1][1]
    clock = 0.0
    Ω = trajectory([clock => ξ])
    dt = dt0
    while k < length(x)-1
        while clock + dt <= τ
            clock, ξ, θ = move_forward(dt, clock, ξ, θ, Flow)
            push!(Ω, clock => ξ)
            dt = dt0
        end
        Δt = τ - clock
        dt = dt - Δt
        clock, ξ, θ = move_forward(τ - clock, clock, ξ, θ, Flow)
        k += 1
        _, ξ, θ = x[k]
        τ = x[k+1][1]
    end
    Ω
end
