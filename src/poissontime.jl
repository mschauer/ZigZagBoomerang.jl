
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


function poisson_time((a, b, c)::NTuple{3}, u)
    if a <= 0 && (a*c + -log(u)*b <= 0)
        return -log(u)/c, 0
    elseif a > 0 && (a + ((-log(u) +  a^2/(2*b))/c)*b < 0)
        return (-log(u) +  a^2/(2*b))/c, 0
    else
        π = -(a + c + sqrt((a + c)^2 - 2*b*log(u)))/b
        if a + π*b >= 0 && a > 0
            return π, 0 #1/2*π*(2*a + π*b + 2*c) == -log(u) 
        else
           #? 
        end
    end
    if b > 0
        if a < 0
            if c*a/b >= -log(u)
                return -log(u)/c, 1
            else
                return sqrt((-log(u) + a^2/(2*b))*2.0/b) - (a+c)/b, 2
            end
        else # a[i]>0
            return sqrt(-log(u)*2.0*b + (a+c)^2)/b - (a+c)/b, 3
        end
    elseif b == 0
        if a > 0
            return -log(u)/(a + c), 4
        else # a[i] <= 0
            return -log(u)/c, 5
        end
    else # b[i] < 0
        if a <= 0 # ok
            return -log(u)/c, 6
        elseif -log(u) <= -a^2/b + a^2/(2*b) - c*a/b # a > =0 NOT OK
            return -sqrt((a+c)^2 - 2.0*log(u)*b)/b^2 - (a+c)/b, 8
        else  # OK
            return -log(u)/c, 9
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
