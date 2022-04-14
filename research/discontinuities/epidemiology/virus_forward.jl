### Forward model to simulate the data
using Random
Random.seed!(1)


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


function forward_simulation_(x0, ξ, ϑ, γ, δ1, δ2, T)
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


function new_input(it, nobs, robs, T)
    N = length(it) 
    x0 = [it; nobs; robs; 0.0; T]
    tag = [fill(1, N); fill(2, N); fill(3, N); 0; 0]
    ind = [eachindex(it);  eachindex(it);  eachindex(it); 0; 0]
    x1 = Vector{Float64}()
    v1 = Vector{Float64}()
    s1 = Vector{Bool}()
    tag′ = Vector{Int64}()
    ind′ = Vector{Int64}()
    for i in eachindex(x0)
        if x0[i] >= T 
            if tag[i] == 1
                push!(x1, T)
                push!(v1, 0.0)
                push!(s1, 1)
            elseif tag[i] == 0
                push!(x1, x0[i])
                push!(v1, 0.0)
                push!(s1, 0)         
            else # && tag[i] != 1
                continue
            end      
        elseif x0[i] < T 
            if tag[i] == 1
                push!(x1, x0[i])
                push!(v1, rand([-1.0, 1.0]))
                push!(s1, 0)
            else
                push!(x1, x0[i])
                push!(v1, 0.0)
                push!(s1, 0)
            end
        end
        push!(tag′, tag[i])
        push!(ind′, ind[i])
    end
    return x1, v1, s1, tag′, ind′
end


function forward_simulation(x0, ξ, ϑ, γ, δ1, δ2, T)
    S, it, nobs, robs = forward_simulation_(x0, ξ, ϑ, γ, δ1, δ2, T)
    x, v, s, tag, ind = new_input(it, nobs, robs, T)
    return S, it, nobs, robs, (x, v, s, tag, ind)
end