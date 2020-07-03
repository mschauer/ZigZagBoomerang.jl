using Revise
using ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
const ZZB = ZigZagBoomerang
struct LocalZigZag{QT} <: ZZB.ContinuousDynamics
    Q::QT
end
neighbours(G, i) = G[i].second


function move_forward!(τ, t, x, θ, Z::LocalZigZag)
    t += τ
    x .+= θ .* τ
    t, x, θ
end

function reflect!(i, θ, x, Z)
    θ[i] = -θ[i]
    θ
end

pos(x) = max(zero(x), x)
function λ(G, ∇ϕ, i, x, θ, Z::LocalZigZag)
    pos(∇ϕ(x, i)*θ[i])
end
function λ_bar(G, i, x, θ, c, Z::LocalZigZag)
    n = neighbours(G, i)
    if isempty(n)
        return pos(c[i]*norm(x[i]*θ[i]))
    else
        return pos(c[i]*sqrt(norm(x[n])^2 + abs2(x[i]))*θ[i])
    end
end


function pdmp_inner(G, ∇ϕ, x, θ, Q, t, c, Z::LocalZigZag; factor=1.5)

    i, t′ = dequeue_pair!(Q)
    if t′ - t < 0
        error("negative time")
    end
    t, x, θ = move_forward!(t′ - t, t, x, θ, Z)

    l, lb = λ(G, ∇ϕ, i, x, θ, Z), λ_bar(G, i, x, θ, c, Z)
    if rand()*lb < l
        if l >= lb
            !adapt && error("Tuning parameter `c` too small.")
            c[i] *= factor
        end
        θ = reflect!(i, θ, x, Z)
        if test
            for j in neighbours(G, i)
                j == i && continue
                enqueue!(Q, j=>t + randexp())
            end
        end
    end
    enqueue!(Q, i=>t + randexp())

end

G = [1=>2:10, 2=>3:5, 3=>5:10, 4=>[5], 5=>[], 6=>7:8, 7=>[], 8=>[], 9=>[10], 10=>[]]
S = Matrix(2.0I, n, n)
for (i, nb) in G
    @show i, nb
    if !isempty(nb) && i >= minimum(nb)
        error("not a tree")
    end
    if !isempty(nb)
        for j in nb
            S[i, j] = 1
        end
    end
end
x
Γ = S * S'
ϕ(x) = 0.5*x'*Γ*x

using ForwardDiff

∇ϕ(x) = ForwardDiff.gradient(ϕ, x)
∇ϕ(x, i) = ∇ϕ(x)[i]


n = 10
x = rand(n)
Q = PriorityQueue{Int,Float64}();

for i in 1:n
    enqueue!(Q, i=>randexp())
end

θ = rand([-1,1], n)
t = 0.0
c = ones(n)
Z = LocalZigZag(NaN)
pdmp_inner(G, ∇ϕ, x, θ, Q, t, c, Z; factor=1.5)
