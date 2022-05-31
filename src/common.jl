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
    s = zero(eltype(x))
    @inbounds for i in nzrange(A, j)
        s += vals[i]'*x[rows[i]]
    end
    s
end


"""
    idot_moving!(A::SparseMatrixCSC, j, t, x, θ, t′, F)

Compute column-vector dot product exploiting sparsity of `A`.
Move all coordinates needed to their position at time `t′`
"""
function idot_moving!(A::SparseMatrixCSC, j, t, x, θ, t′, F)
    rows = rowvals(A)
    vals = nonzeros(A)
    s = zero(eltype(A))
    @inbounds for i in nzrange(A, j)
        smove_forward!(rows[i], t, x, θ, t′, F)
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

function sep(x)
    [first(xi) for xi in x], [copy(last(xi)) for xi in x]
end

"""
    splitpairs(tx) = t, x

Splits a vector of pairs into a pair of vectors.
"""
splitpairs(x) = first.(x) => last.(x)
export splitpairs
