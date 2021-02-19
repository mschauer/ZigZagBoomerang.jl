function generate_data(ξ_true, p = 100, N = 100, zero_ratio = 0.0)
    #ξ_true of dim p
    println("Create a full design matrix and simulate data. The ratio N/p is $(N/p)")
    if !(0 <= zero_ratio <= 1)
        error("The proportion of zero coefficients has to be a number between 0 and 1")
    end
    X = [ones(N) randn(N,p-1)]
    for i in eachindex(ξ_true)
        if zero_ratio >= rand()
            ξ_true[i] = 0
        end
    end
    ξ_true
    q = exp.(X*ξ_true)
    z = q ./(1 .+ q)
    y = Float64.(rand(N) .< z)
    return(X, y)
end
