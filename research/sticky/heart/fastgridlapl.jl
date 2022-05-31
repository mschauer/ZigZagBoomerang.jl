using SparseArrays
using LinearAlgebra
function gridlaplacian(::Type{T}, m, n) where T
    linear = LinearIndices((1:m, 1:n))
    Is = Int[]
    Js = Int[]
    Vs = T[]
    for i in 1:m
        for j in 1:n
            for (i2, j2) in ((i + 1, j), (i, j + 1))
                if i2 <= m && j2 <= n
                    push!(Is, linear[i, j])
                    push!(Js, linear[i2, j2])
                    push!(Vs, -1)
                    push!(Is, linear[i2, j2])
                    push!(Js, linear[i, j])
                    push!(Vs, -1)
                    push!(Is, linear[i, j]) 
                    push!(Js, linear[i, j]) 
                    push!(Vs, 1)
                    push!(Js, linear[i2, j2])
                    push!(Is, linear[i2, j2])
                    push!(Vs, 1)
                end
            end
        end
    end
    sparse(Is, Js, Vs)
end
function gridlaplacian_old(T, m, n)
    S = sparse(T(0.0)I, n*m, n*m)
    linear = LinearIndices((1:m, 1:n))
    for i in 1:m
        for j in 1:n
            for (i2, j2) in ((i + 1, j), (i, j + 1))
                if i2 <= m && j2 <= n
                    S[linear[i, j], linear[i2, j2]] -= 1.
                    S[linear[i2, j2], linear[i, j]] -= 1.

                    S[linear[i, j], linear[i, j]] += 1.
                    S[linear[i2, j2], linear[i2, j2]] += 1.
                end
            end
        end
    end
    S
end

#@time A = gridlaplacian(Int, 40, 100) 
#@time B = gridlaplacian_old(Int, 40, 100)
#A == B