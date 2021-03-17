using SparseArrays
 

@testset "SZigZag (Iter" begin
    d = 8
    S = 1.3I + 0.5sprandn(d, d, 0.1)
    Γ = S*S'

    ∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation
    ∇ϕmoving(t, x, θ, i, t′, F, Γ) = ZigZagBoomerang.idot_moving!(Γ, i, t, x, θ, t′, F) # sparse computation

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0,-0.5,0.5,1.0], d)


    c = .7*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(0.9Γ, x0*0)
    T = 1000.0

    sampler = FactSampler(∇ϕ, t0 =>(x0, θ0), c, Z, Γ)

    tr = trace(sampler, T)
    dt = 0.5
    ts, xs = sep(collect(discretize(tr, dt)))

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end
