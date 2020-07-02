
"""
    poisson_time(a, b, u)

obtaining waiting time for inhomogeneous Poisson Process
with rate of the form λ(t) = (a + b*t)^+, `a`,`b` ∈ R, `u` uniform random variable
"""
function poisson_time(a, b, u)
    if b > 0
        if a < 0
            τ = sqrt(-log(u)*2.0/b) - a/b
        else # a[i]>0
            τ = sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        end
    elseif  b == 0
        if a > 0
            τ = -log(u)/a
        else # a[i] <= 0
            τ = Inf
        end
    else # b[i] < 0
        if a <= 0
            τ = Inf
        elseif -log(u) <= -a^2/b + a^2/(2*b)
            τ = - sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        else
            τ = Inf
        end
    end
end

"""
    h_poisson_time(a, u)
obtaining waiting time for homogeneous Poisson Process
with rate of the form λ(t) = a, `a` ≥ 0, `u` uniform random variable
"""
function poisson_time(a,u)
    -log(u)/a
end
