using SparseArrays
using LinearAlgebra

@testset "SZigZag (Iter" begin
    d = 12
    d2 = d÷2
    S = 1.3I + 0.5sprandn(d, d, 0.1)
    Γ = S*S'
    Γ2 = copy(Γ)
    Γ2[1:d2,d2+1:d] .= 0
    Γ2[d2+1:d,1:d2] .= 0
    dropzeros!(Γ2)

    ∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0,-0.5,0.5,1.0], d)


    c = .7*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(Γ2, x0*0)
    T = 1000.0

  
    dt = 0.5
    @test 0.1/sqrt(T) < mean(abs.(mean(tr))) < 2/sqrt(T)
    
    ts, xs = sep(collect(discretize(tr, dt)))

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end
