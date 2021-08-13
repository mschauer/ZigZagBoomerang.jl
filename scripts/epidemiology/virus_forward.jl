### Forward model to simulate the data

# number of individuals
N = 10

# infectivity metric 
d(i,j) = 1 - (abs(i-j)+1)/N

# poisson_time 
poisson_time(c) = -log(rand())/c

# infectivity pressure on individual j where
# `ϑ` is the baseline infectivity and
# `ξ` is the baseline susceptibility 
function beta(j, x, ξ, ϑ)
    β = 0.0
    for i in eachindex(x) 
        if (i != j) && x[i] == 1 
           β += d(i,j)*ξ[j]*ϑ[i]
        end
    end
    β
end

# Time window
function forward_simulation(x, ξ, ϑ, T)
    t = 0.0
    S = [(copy(x) => t)]
    while true
        τ, ii = Inf, 0
        for j in eachindex(x)
            if x[j] == 0
                β = beta(j, x, ξ, ϑ)
                τ_new = poisson_time(β)
                if τ_new < τ
                    τ = τ_new
                    ii = j
                end
            end
        end
        t += τ
        if t>T
            break
        end
        x[ii] = 1
        push!(S, copy(x) => t)
    end
    S
end


# baseline susceptibility 
ξ = rand(N)
# baseline infectivity
ϑ = rand(N)
# Initialize population 
x = 1:10 .== 5
# Final time
T = 10.0
forward_simulation(x, ξ, ϑ, T)