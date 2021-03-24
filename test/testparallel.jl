using SparseArrays
using LinearAlgebra

@testset "Parallel ZigZag" begin
    d = 12
    d2 = d÷2
    S = 2.0I + 0.5sprandn(d, d, 0.1)
    Γ = S*S'
    Γ2 = copy(Γ)
    Γ2[1:d2,d2+1:d] .= 0
    Γ2[d2+1:d,1:d2] .= 0
    dropzeros!(Γ2)

    partition = ZigZagBoomerang.Partition(2, d)

    ∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0,-0.5,0.5,1.0], d)

    G = [i => rowvals(Γ)[nzrange(Γ, i)] for i in eachindex(θ0)]

    c = 100*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(Γ2, x0*0)
    T = 100.0
    tr, (t, x, θ), (acc, num) = ZigZagBoomerang.parallel_spdmp(partition, ∇ϕ, t0, x0, θ0, T, c, G, Z, Γ)
 
  
    dt = 0.5
    @test 0.1/sqrt(T) < mean(abs.(mean(tr))) < 2/sqrt(T)
    
    ts, xs = ZigZagBoomerang.sep(collect(discretize(tr, dt)))

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end
