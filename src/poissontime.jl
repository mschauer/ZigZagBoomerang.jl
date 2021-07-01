
"""
    poisson_time(a, b, u)

Obtaining waiting time for inhomogeneous Poisson Process
with rate of the form λ(t) = (a + b*t)^+, `a`,`b` ∈ R, `u` uniform random variable
"""
function poisson_time(a, b, u)
    if b > 0
        if a < 0
            return sqrt(-log(u)*2.0/b) - a/b
        else # a[i]>0
            return sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        end
    elseif b == 0
        if a > 0
            return -log(u)/a
        else # a[i] <= 0
            return Inf
        end
    else # b[i] < 0
        if a <= 0
            return Inf
        elseif -log(u) <= -a^2/b + a^2/(2*b)
            return -sqrt((a/b)^2 - log(u)*2.0/b) - a/b
        else
            return Inf
        end
    end
end

"""
    poisson_time((a, b, c), u)

Obtaining waiting time for inhomogeneous Poisson Process
with rate of the form λ(t) = c + (a + b*t)^+, 
where `c`> 0 ,`a, b` ∈ R, `u` uniform random variable
"""
function poisson_time((a,b,c)::NTuple{3}, u)
    if b > 0
        if a < 0
            if !(-c*a/b + log(u) >= 0)
                return sqrt(-2*b*log(u) + c^2 + 2*a*c)/b - (a+c)/b
            else
                return -log(u)/c
            end
        else # a >0
            return sqrt(-log(u)*2.0*b + (a+c)^2)/b - (a+c)/b
        end
    elseif b == 0
        if a > 0
            return -log(u)/(a + c)
        else # a <= 0
            return -log(u)/c
        end
    else # b < 0
        if a <= 0.0 
            return -log(u)/c
        elseif  - c*a/b - a^2/(2*b)  + log(u) > 0.0
            return +sqrt((a+c)^2 - 2.0*log(u)*b)/b - (a+c)/b
        else
            return (-log(u)+ a^2/(2*b))/c
        end
    end
end

"""
    poisson_time(a[, u])

Obtaining waiting time for homogeneous Poisson Process
with rate of the form λ(t) = a, `a` ≥ 0, `u` uniform random variable
"""
function poisson_time(a::Number, u::Number)
    -log(u)/a
end

function poisson_time(rng, a)
    randexp(rng)/a
end
function poisson_time(a)
    randexp()/a
end

poisson_time((a, b)::Tuple, u=randn()) = poisson_time(a, b, u)
