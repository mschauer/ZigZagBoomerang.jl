#=
# Inventory

Boundaries, Reference #Q

State space
# u = (x, v, f)

Flow # 

Gradient of negative log-density ϕ    

Graph

Skeleton 
=#
using Test

struct StickyBarriers{Tx,Trule,Tκ}
    x::Tx # Intervals
    rule::Trule # set velocity to 0 and change label from free to frozen
    κ::Tκ 
end    

struct StickyFlow
end

struct StickyUpperBounds{TG,TΓ,Tstrong,Tc,Tfact}
    G::TG
    Γ::TΓ
    strong::Tstrong
    adapt::Bool
    c::Tc
    factor::Tfact
end

struct StructuredTarget{TG,T∇ϕ}
    G::TG
    ∇ϕ::T∇ϕ
end

StructuredTarget(Γ::SparseMatrixCSC, ∇ϕ) = StructuredTarget([i => rowvals(Γ)[nzrange(Γ, i)] for i in axes(Γ, 1)], ∇ϕ)


struct AcceptanceDiagnostics
    acc::Int
    num::Int
end

function stickystate(x0)
    d = length(x0)
    v0 = rand((-1.0, 1.0), d)
    t0 = zeros(d)
    u0 = (t0, x0, v0) 
end

struct EndTime
    T::Float64
end
finished(end_time::EndTime, t) = t < end_time.T


function stickyzz(u0, target::StructuredTarget, flow::StickyFlow, upper_bounds::StickyUpperBounds, barriers::Vector{<:StickyBarriers}, end_condition)
    # Initialize
    (t0, x0, v0) = u0
    t′ = maximum(t0)
    # priority queue
    Q = []
    # Skeleton
    Ξ = []

    # Diagnostics
    acc = AcceptanceDiagnostics(0, 0)

    Ξ = @inferred sticky_main(Q, Ξ, t′, u0, target, flow, upper_bounds, barriers, end_condition, acc)

    return Ξ
end

function sticky_main(Q, Ξ, t′, u, target, flow, upper_bounds, barriers, end_condition, acc)
    while !finished(end_condition, t′) 
        t′ = stickyzz_inner!(Q, Ξ, t′, u, target, flow, upper_bounds, barriers, acc)
        #    (acc, num), c,  b, t_old = sspdmp_inner!(Ξ, G, G1, G2, ∇ϕ, t, x, θ, Q,
        # c, b, t_old, f, θf, (acc, num), F, κ, args...; 
        # , factor=factor,
        # adapt=adapt)
    end
    return Ξ
end
function stickyzz_inner!(Q, Ξ, t′, u, target, flow, upper_bounds, barriers, acc)
    return t′ + 1
end


