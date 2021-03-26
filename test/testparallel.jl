#using Revise
#using ProfileView
#using Profile
#Profile.init()
using Random
Random.seed!(1)
#Profile.clear()
using Statistics, ZigZagBoomerang, LinearAlgebra, Test, SparseArrays
using ZigZagBoomerang: Partition
@testset "Partition" begin
    d = 6
    partition = ZigZagBoomerang.Partition(2, d)
    for i in 1:d
        a,b = partition(i)
        @test i == partition(a,b)
        @test a == (i > 3) + 1
        @test b == mod1(i, 3)
    end

end
@testset "Parallel ZigZag" begin
    if (@isdefined CI) && CI
        d = 20
        K = 2
        T = 1000.0
        Δ = 0.05
    else
        d = 50000
        K = 16
        T = 100.0
        Δ = 0.2
    end

    d2 = d÷K
   
    Γ = sparse(SymTridiagonal(1.0ones(d), -0.4ones(d-1)))
    Γ2 = copy(Γ)
    for k1 in 0:K-1
        for k2 in 0:K-1
            k2 == k1 && continue
            Γ2[(k1*d2 + 1):(k1+1)*d2,(k2*d2 + 1):(k2+1)*d2] .= 0
        end
    end

    dropzeros!(Γ2)

    partition = ZigZagBoomerang.Partition(K, d)

    ∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation

    t0 = 0.0
    x0 = 0.1randn(d)
    θ0 = rand([-1.0,1.0], d)

    G = [i => rowvals(Γ)[nzrange(Γ, i)] for i in eachindex(θ0)]

    c = 5*[norm(Γ[:, i], 2) for i in 1:d]

    dt = 0.5
 
    println("Single thread:")
    Z = ZigZag(Γ, x0*0)
    tr, _ = @time ZigZagBoomerang.spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ; structured=true)
    @test 0.1/sqrt(T) < mean(abs.(mean(tr))) < 4/sqrt(T)    
    if d <= 100
        ts, xs = ZigZagBoomerang.sep(collect(discretize(tr, dt)))
        @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) <4/sqrt(T)
    end
    Z = ZigZag(Γ2, x0*0)
    println("Multithreaded: (",Threads.nthreads()," cores)")
    tr, (t, x, θ), (acc, num) = @time ZigZagBoomerang.parallel_spdmp(partition, ∇ϕ, t0, x0, θ0, T, c, G, Z, Γ; Δ=Δ)
    @test 0.1/sqrt(T) < mean(abs.(mean(tr))) < 4/sqrt(T)

    if d < 100 
         ts, xs = ZigZagBoomerang.sep(collect(discretize(tr, dt)))
         @test_broken mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 4/sqrt(T)
    end
end
