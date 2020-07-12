
function sparse_design(d = (5,5,5), r = 5, m = 100)
    p = sum(d) + (sum(d)^2 - sum(d.^2))÷2 + r
    K = length(d)
    n = m*p
    D = (0, cumsum(d)...)
    A = zeros(Float64, n, p)
    for i in 1:n
        j = D[end]
        for k in 1:K
            A[i, rand(D[k]+1:D[k+1])] = rand() < (d[k]-1)/d[k]
            for k2 in 1:k-1
                for c in CartesianIndices((d[k],d[k2]))
                    j += 1
                    A[i, j] = 0.3*((A[i, D[k] + c[1]] == 1) & (A[i, D[k2] + c[2]] == 1))
                end
            end
        end
        for _ in 1:r
            j += 1
            A[i, j] = 0.2*randn()
        end
        @assert j == p
    end
    A = sparse(A)
end
