### Forward model to simulate the data
using Random
Random.seed!(1)
# number of individuals
N = 50

# infectivity metric 
d(i,j) = 1 - (abs(i-j)+1)/N

# poisson_time 
poisson_time(c) = -log(rand())/c


# infectivity pressure on individual j where
# `ϑ` is the baseline infectivity and
# `ξ` is the baseline susceptibility 

function beta(j, x, ξ, ϑ, γ)
    β = 0.0
    for i in eachindex(x) 
        if (i != j) && x[i] == 1  #infected but not notified
            β += d(i,j)*ξ[j]*ϑ[i]
        elseif (i != j) && x[i] == 2
            β += d(i,j)*ξ[j]*ϑ[i]*γ # infected and notified
        end #not infected or removed
    end
    β
end


function forward_simulation(x0, ξ, ϑ, γ, δ1, δ2, T)
    x = copy(x0)
    t = 0.0
    S = [(copy(x) => t)]
    nobs = fill(T, length(x0))
    robs = fill(T, length(x0))
    it = fill(T, length(x0))
    for i in eachindex(x0)
        if x0[i] == 1
            it[i] = 0.0
        end
    end
    while true
        τ, τ_new, ii = Inf, Inf, 0
        for j in eachindex(x)   
            if x[j] == 0    # from susceptible to infected
                β = beta(j, x, ξ, ϑ, γ)
                τ_new = poisson_time(β)
            elseif x[j] == 1 # from infected to notified
                τ_new = poisson_time(δ1)
            elseif x[j] == 2 #  from notified to removed
                τ_new = poisson_time(δ2)
            end
            if τ_new < τ
                τ = τ_new
                ii = j
            end
        end
        t += τ
        if t>T
            break
        end
        x[ii] += 1
        push!(S, copy(x) => t)
        if x[ii] == 1
            it[ii] = t
        elseif x[ii] == 2 #notification times
            nobs[ii] = t
        elseif x[ii] == 3 #removed times
            robs[ii] = t
        end
    end
    S, it, nobs, robs
end


# baseline susceptibility 
ξ = rand(N).*0.5
# baseline infectivity
ϑ = rand(N).*0.5
# reduction after notification 
γ = 0.1
# rate of 1 -> 2
δ1 = 0.1
# rate of 2 -> 3
δ2 = 0.1

# Initialize population 
x = (1:N .== 5)*1
# Final time
T = 20.0
S, it, nobs, robs = forward_simulation(x, ξ, ϑ, γ, δ1, δ2, T)

nobs
println(" (Unobserved) number of infected: at time $T:   $(sum(it .< T - eps()))")
println(" (Observed) number of notified at time $T:   $(sum(nobs .< T - eps()))")
println(" (Observed) number of removed at time $T:   $(sum(robs .< T - eps()))")