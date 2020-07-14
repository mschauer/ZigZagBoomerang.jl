using SparseArrays

"""
    pos(x)

Positive part of `x` (i.e. max(0,x)).
"""
pos(x) = max(zero(x), x)

"""
    idot(A, j, x) = dot(A[:, j], x)

Compute column-vector dot product exploiting sparsity of `A`.
"""
idot(A, j, x) = dot((@view A[:, j]), x)
function idot(A::SparseMatrixCSC, j, x)
    rows = rowvals(A)
    vals = nonzeros(A)
    s = zero(eltype(A))
    @inbounds for i in nzrange(A, j)
        s += vals[i]*x[rows[i]]
    end
    s
end

"""
    normsq(x)

Squared 2-norm.
"""
normsq(x::Real) = abs2(x)
normsq(x) = dot(x,x)

sep(x) = first.(x), last.(x)
