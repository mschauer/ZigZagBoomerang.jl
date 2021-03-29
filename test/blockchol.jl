using LinearAlgebra
using SparseArrays

lchol(A) = LowerTriangular(Matrix(LinearAlgebra._chol!(copy(A), UpperTriangular)[1])')
lchol!(A) = LowerTriangular(Matrix(LinearAlgebra._chol!(A, UpperTriangular)[1])')


"""
    cholinverse!(L, x)
Solve L L'y =x using two backsolves,
L should be lower triangular
"""
function cholinverse!(L, x)
    naivesub!(L, x) # triangular backsolves
    naivesub!(UpperTriangular(L'), x)
    x
end



function naivesub!(At::Adjoint{<:Any,<:LowerTriangular}, b::AbstractVector, x::AbstractVector = b)
    A = At.parent
    n = size(A, 2)
    if !(n == length(b) == length(x))
        throw(DimensionMismatch("second dimension of left hand side A, $n, length of output x, $(length(x)), and length of right hand side b, $(length(b)), must be equal"))
    end
    @inbounds for j in n:-1:1
        iszero(A.data[j,j]) && throw(SingularException(j))
        xj = x[j] = A.data[j,j] \ b[j]
        for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
            b[i] -= A.data[j,i] * xj
        end
    end
    x
end

function naivesub!(At::Adjoint{<:Any,<:LowerTriangular}, B::AbstractMatrix, X::AbstractMatrix = B)
    A = At.parent
    n = size(A, 2)
    if !(n == size(B,1) == size(B,2) == size(X,1) == size(X,2))
        throw(DimensionMismatch())
    end
    @inbounds for k in 1:n
        for j in n:-1:1
            iszero(A.data[j,j]) && throw(SingularException(j))
            xjk = X[j,k] = A.data[j,j] \ B[j,k]
            for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                B[i,k] -= A.data[j,i] * xjk
            end
        end
    end
    X
end

function naivesub!(A::LowerTriangular, B::AbstractMatrix, X::AbstractMatrix = B)
    n = size(A,2)
    if !(n == size(B,1) == size(X,1))
        throw(DimensionMismatch())
    end
    if !(size(B,2) == size(X,2))
        throw(DimensionMismatch())
    end


    @inbounds for k in 1:size(B,2)
        for j in 1:n
            iszero(A.data[j,j]) && throw(SingularException(j))
            xjk = X[j,k] = A.data[j,j] \ B[j,k]
            for i in j+1:n
                B[i,k] -= A.data[i,j] * xjk
            end
        end
    end
    X
end


function naivesub!(A::UpperTriangular, B::AbstractMatrix, X::AbstractMatrix = B)
    n = size(A, 2)
    if !(n == size(B,1) == size(X,1))
        throw(DimensionMismatch())
    end
    if !(size(B,2) == size(X,2))
        throw(DimensionMismatch())
    end

    @inbounds for k in 1:size(B, 2)
        for j in n:-1:1
            iszero(A.data[j,j]) && throw(SingularException(j))
            xjk = X[j,k] = A.data[j,j] \ B[j,k]
            for i in j-1:-1:1 # counterintuitively 1:j-1 performs slightly better
                B[i,k] -= A.data[i,j] * xjk
            end
        end
    end
    X
end