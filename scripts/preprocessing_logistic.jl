# full kth partial derivative
function ∇ϕ(x, k, A, y, precision)
    res  = 0.0
    for j in 1:size(A)[1]
        res += ∇ϕs(x, k, j, A, y)
    end
    res + x[k]*precision
end

# full in  place gradient
function ∇ϕ!(G, x, X, y, precision)
    for k in 1:length(x)
        G[k] = ∇ϕ(x, k, X, y, precision)
    end
    G
end

println("Preprocessing data")
ξref = randn(d)
precision = 1.0
norm(ξref - ξtrue)
results = optimize(x -> ϕ(x, X, y, precision), (G, x) -> ∇ϕ!(G, x, X, y, precision),  ξref)
ξref = Optim.minimizer(results)
