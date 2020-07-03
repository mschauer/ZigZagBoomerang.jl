using DataStructures
using Statistics
using SparseArrays

struct LocalZigZag{T,S} <: ContinuousDynamics
    Γ::T
    μ::S
end

"""
    neighbours(G, i)

Return extended neighbourhood of `i` including `i`
"""
neighbours(G::Vector{<:Pair}, i) = G[i].second

function move_forward!(τ, t, x, θ, Z::LocalZigZag)
    t += τ
    x .+= θ .* τ
    t, x, θ
end

function reflect!(i, θ, x, Z)
    θ[i] = -θ[i]
    θ
end
normsq(x::Real) = abs2(x)
normsq(x) = dot(x,x)


function λ(G, ∇ϕ, i, x, θ, Z::LocalZigZag)
    pos(∇ϕ(x, i)*θ[i])
end

#ab(G, i, x, θ, c, Z::LocalZigZag) = Z.Γ[:,i]'*(x-Z.μ)*θ[i] + c[i]*sqrt(sum(abs2(x[j]-Z.μ[j]) for j in neighbours(G, i))), 1.0
#ab(G, i, x, θ, c, Z::LocalZigZag) = c[i]*sqrt(sum(abs2(x[j]-Z.μ[j]) for j in neighbours(G, i))), 1.0
#ab(G, i, x, θ, c, Z::LocalZigZag) = c[i] + mean(θ[j]*(x[j] - Z.μ[j]) for j in neighbours(G, i)), 1.0
function ab(G, i, x, θ, c, Z::LocalZigZag)
    a = c[i] + θ[i]*sum(Z.Γ[i, :].nzval[ji]*(x[j] - Z.μ[j]) for (ji, j) in enumerate(neighbours(G, i)))
    b = θ[i]*sum(Z.Γ[i, :].nzval[ji]*θ[j] for (ji, j) in enumerate(neighbours(G, i)))
    a, b
end

λ_bar(G, i, x, θ, c, Z::LocalZigZag) = pos(ab(G, i, x, θ, c, Z::LocalZigZag)[1])


event(i, t, x, θ, Z::LocalZigZag) = (i, t, x[i])


function pdmp(G, ∇ϕ, t0, x0, θ0, T, c, Z::LocalZigZag; factor=1.5, adapt=false)

    t, x, θ = t0, copy(x0), copy(θ0)
    num = acc = 0

    Q = PriorityQueue{Int,Float64}();

    for i in eachindex(θ)
        enqueue!(Q, i=>poisson_time(ab(G, i, x, θ, c, Z)..., rand()))
    end

    Ξ = [event(1, t, x, θ, Z)][1:0]
    while t < T
        t, x, θ, (num, acc) = pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), Z; factor=1.5)
    end
    Ξ, (t, x, θ), (acc, num)
end

function pdmp_inner!(Ξ, G, ∇ϕ, x, θ, Q, t, c, (num, acc), Z::LocalZigZag; factor=1.5, adapt=false)

    i, t′ = dequeue_pair!(Q)
    if t′ - t < 0
        error("negative time")
    end
    t, x, θ = move_forward!(t′ - t, t, x, θ, Z)

    l, lb = λ(G, ∇ϕ, i, x, θ, Z), λ_bar(G, i, x, θ, c, Z)
    num += 1

    if rand()*lb < l
        acc += 1
        if l >= lb
            !adapt && error("Tuning parameter `c` too small.")
            c[i] *= factor
        end
        θ = reflect!(i, θ, x, Z)
        for j in neighbours(G, i)
            j == i && continue
            Q[j] = t + poisson_time(ab(G, j, x, θ, c, Z)..., rand())
        end
        push!(Ξ, event(i, t, x, θ, Z))
    end
    enqueue!(Q, i=>t + poisson_time(ab(G, i, x, θ, c, Z)..., rand()))
    t, x, θ, (num, acc)
end
