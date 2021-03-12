function generate_data(ξ_true, p = 100, N = 100, zero_ratio = 0.0)
    #ξ_true of dim p
    println("Create a full design matrix and simulate data. The ratio N/p is $(N/p)")
    println("The ratio N/alpha is $(N/sum(ξ_true.!= 0))")
    X = [ones(N) randn(N,p-1)]
    ξ_true
    q = exp.(X*ξ_true)
    z = q ./(1 .+ q)
    y = Float64.(rand(N) .< z)
    return(X, y)
end
