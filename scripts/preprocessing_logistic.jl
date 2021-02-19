# full kth partial derivative
function ∇ϕ(x, k, A, y)
    res  = 0.0
    for j in 1:size(A)[1]
        res += ∇ϕs(x, k, j, A, y)
    end
    res
end

# full in  place gradient
function ∇ϕ!(G, x, X, y)
    for k in 1:length(x)
        G[k] = ∇ϕ(x, k, X, y)
    end
    G
end

println("Preprocessing data")
ξref = randn(d)
println("Distance before optimization: $(norm(ξref - ξtrue))")
results = optimize(x -> ϕ(x, X, y), (G, x) -> ∇ϕ!(G, x, X, y),  ξref)
println("Optimization success: $(results.ls_success)")
ξref = Optim.minimizer(results)
println("Distance after optimization: $(norm(ξref - ξtrue))")
