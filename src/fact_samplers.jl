# implementation of factorised samplers

using DataStructures
using Statistics
using SparseArrays
using LinearAlgebra
using Graphs

"""
    neighbours(G::Vector{<:Pair}, i) = G[i].second

Return extended neighbourhood of `i` including `i`.
`G`: graphs of neightbourhoods
"""
neighbours(G::Vector{<:Pair}, i) = G[i].second
neighbours(G::SimpleGraph, i) = neighbors(G, i)
#need refreshments
hasrefresh(::FactBoomerang) = true
hasrefresh(Z::ZigZag) = Z.λref > 0




"""
    λ(∇ϕ, i, x, θ, Z::ZigZag)
`i`th Poisson rate of the `ZigZag` sampler
"""
function λ(∇ϕi, i, x, θ, Z::ZigZag)
    pos(∇ϕi'*θ[i])
end


"""
    λ(∇ϕi, i, x, θ, Z::FactBoomerang)
`i`th Poisson rate of the `FactBoomerang` sampler
"""
function λ(∇ϕi, i, x, θ, B::FactBoomerang)
    pos((∇ϕi - (x[i] - B.μ[i])*B.Γ[i,i])*θ[i])
end

loosen(c, x) = c + x #+ log(c+1)*abs(x)/100
"""
    ab(G, i, x, θ, c, Flow)

Returns the constant term `a` and linear term `b` when computing the Poisson times
from the upper upper bounding rates λᵢ(t) = max(a + b*t)^2. The factors `a` and `b`
can be function of the current position `x`, velocity `θ`, tuning parameter `c` and
the Graph `G`
"""
function ab(G, i, x, θ, c, Z::ZigZag, args...)
    a = loosen(c[i], (idot(Z.Γ, i, x)  - idot(Z.Γ, i, Z.μ))'*θ[i])
    b = loosen(c[i]/100, θ[i]'*idot(Z.Γ, i, θ))
    a, b
end



function ab(G, i, x, θ, c, Z::FactBoomerang)
    nhd = neighbours(G, i)
    z = sqrt(sum((x[j] - Z.μ[j])^2 + θ[j]^2 for j in nhd))
    z2 = (x[i]^2 + θ[i]^2)
    a = c[i]*sqrt(z2)*z + z2*Z.Γ[i,i]
    b = 0.0
    a, b
end

function adapt!(c, i, factor)
    c[i] *= factor
    c
end


function event(i, t, x, θ, Z::Union{ZigZag,FactBoomerang,JointFlow})
    t, i, x[i], θ[i]
end





