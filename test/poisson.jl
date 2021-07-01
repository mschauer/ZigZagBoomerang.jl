using ZigZagBoomerang
using Test
using ZigZagBoomerang: poisson_time
"""
int (max(a + b*t, 0) + c) dt from 0 to pi 
"""
function F((a,b,c), π)
    if a <= 0 && (a + π*b <= 0)
        return π*c
    elseif a > 0 && (a + π*b < 0)
        return π*c - a^2/(2*b)
    elseif a > 0 && (a + π*b >= 0)
        return 1/2*π*(2*a + π*b + 2*c) 
    else # a <= 0 && (a + π*b > 0)
        return a^2/(2*b) + π*a + (π^2*b)/2 + π*c 
        #return (π*b*(π*b + 2*c) - 2*(-a)*π*b + a^2)/(2b)
    end
end
@testset "poisson" begin
    for k in 1:100
        a, b = 2rand(2) .- 1
        u = rand()
        s = poisson_time((a,b), u)
        if s == Inf
            t = F((a,b,false), s)
            @test t < -log(u)
        else
            t = F((a,b,false), s)
            @test t ≈ -log(u)
        end
    end
end

@testset "poisson" begin
    for k in 1:100
        a, b = 2rand(2) .- 1
        c = rand()
        u = rand()
        s = poisson_time((a,b,c), u)
        if s == Inf
            t = F((a,b,c), s)
            @test t < -log(u)
        else

            t = F((a,b,c), s)
            @test t ≈ -log(u)
        end
    end
end
