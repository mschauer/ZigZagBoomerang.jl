import Random
function example_design_matrix(;num_rows = 50_000)
    num_categorical_features = 100
    num_continuous_features = 10
    X = zeros(Float64, num_rows, num_categorical_features + num_continuous_features)
    for j = 1:num_categorical_features # binary categorical features
        X[:, j] = rand(num_rows) .> 0.5
    end
    for j = (num_categorical_features+1):(num_categorical_features+num_continuous_features) # continuous features that we have normalized
        X[:, j] = randn(num_rows)
    end
    return X
end
