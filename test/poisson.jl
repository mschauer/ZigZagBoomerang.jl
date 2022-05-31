using ZigZagBoomerang
using Test
using ZigZagBoomerang: poisson_time
@testset "Poisson" begin
    Random.seed!(1)
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
        end
    end
    @testset "Poisson 1" begin
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

    @testset "Poisson 2" begin
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


    # testing poisson time sampler
    @testset "Poisson 3" begin
        Random.seed!(1)
        a, b = 1.1, 0.0
        n = 5000
        Λ0(a, b, T) = a*T + b*(T)^2/2
        P(a, b, T) = 1 - exp(-Λ0(a, b, T))
        T = 0.7

        for (a, b, pt) in ((1.1, 0.0, NaN), (1.1, 0.3, NaN), (0.0, 0.3, NaN), (1.1, -0.5, NaN),
            (-0.5, 1, P(0, 1, T-0.5)), (-1, -2, 0.0))
            p = mean(poisson_time(a, b, rand()) < T for i in 1:n)
            if isnan(pt)
                pt = P(a, b, T)
            end
            @test abs(p - pt) < 2/sqrt(n)
        end
    end
end