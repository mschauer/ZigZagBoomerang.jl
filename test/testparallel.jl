using Revise
using ProfileView
using Profile
Profile.init()
using Random
Random.seed!(1)
Profile.clear()
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
    d = 2000
    d2 = d÷2
   
    Γ = sparse(SymTridiagonal(1.0ones(d), -0.3ones(d-1)))
    Γ2 = copy(Γ)
    Γ2[1:d2,d2+1:d] .= 0
    Γ2[d2+1:d,1:d2] .= 0
    dropzeros!(Γ2)

    partition = ZigZagBoomerang.Partition(2, d)

    ∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0,1.0], d)

    G = [i => rowvals(Γ)[nzrange(Γ, i)] for i in eachindex(θ0)]

    c = 10*[norm(Γ[:, i], 2) for i in 1:d]

    T = 100.0
    # Z = ZigZag(Γ, x0*0)
    # tr, _ = @time ZigZagBoomerang.spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
    # dt = 0.5
    # @test 0.1/sqrt(T) < mean(abs.(mean(tr))) < 4/sqrt(T)    
    # ts, xs = ZigZagBoomerang.sep(collect(discretize(tr, dt)))
    # @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) <4/sqrt(T)

    Z = ZigZag(Γ2, x0*0)
    tr, (t, x, θ), (acc, num) = @time ZigZagBoomerang.parallel_spdmp(partition, ∇ϕ, t0, x0, θ0, T, c, G, Z, Γ)
 
  
    @test 0.1/sqrt(T) < mean(abs.(mean(tr))) < 4/sqrt(T)
    dt = 0.5  
   # ts, xs = ZigZagBoomerang.sep(collect(discretize(tr, dt)))
   # @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 5/sqrt(T)
end
