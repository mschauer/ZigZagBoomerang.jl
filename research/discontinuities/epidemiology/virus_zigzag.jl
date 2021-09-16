# simulate data
include("virus_forward.jl")
# observation notified, removed
nobs

function hit_time_b(i, x,v, nobs)
    if v[i] > 0
        return (nobs[i] - x[i])/v[i] # hitting right limit
    else # v[i] < 0 
        return x[i]/v[i] #hitting left limit
    end
end

# integrated lambdas
# ∫_0^T \lambda_{i,j}(t) dt
function Λ(i,j, x, nobs, robs)
    i == j && return 0.0
    if x[i] < x[j]
        res = ξ[j]*ϑ[i]*((min(nobs[i],x[j]) - x[i]) + γ*(min(robs[i],x[j]) - min(nobs[i],x[j])))
        @assert res > 0
    else
        res = 0.0
    end
    return res
end
#marginalized over `j`s
Λ(j, x, nobs, robs) = sum([Λ(i,j, x, nobs, robs) for i in 1:N]) 
#marginalizez over `i`s and `j`s
Λ(x, nobs, robs) = sum([Λ(i,j, x, nobs, robs) for (i,j) in zip(1:N, 1:N)]) 


# exponential density and distribution for delay between infection and notiafication 
f(x, γ) =  γ*exp(-γ*x)
Fc(x, γ) = 1 - exp(-γ*x)

lf(x, γ) = -γ*x # up to constant of proportionality
# ∇lf(x,v) = -γ

lFc(x,γ) = -γ*x 
# ∇lFc(x, γ) = -γ #ah funny


#logdensity as in equation (2.2)
function ∇ϕ(x, y, z, nobs, robs)
    Λ(x, nobs, robs) + γ*sum(z) 
end


#sticky times 
function sticky_time(i, x, nobs, robs)
    exp(-Λ(x, i, nobs, robs))
end


















