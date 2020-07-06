using SparseArrays
pos(x) = max(zero(x), x)

"""
    idot(A, j, x) = dot(A[:, j], x)

Compute exploiting sparsity.
"""
idot(A, j, x) = dot(A[:, j], x)
function idot(A::SparseMatrixCSC, j, x)
    rows = rowvals(A)
    vals = nonzeros(A)
    s = zero(eltype(A))
    @inbounds for i in nzrange(A, j)
        s += vals[i]*x[rows[i]]
    end
    s
end
