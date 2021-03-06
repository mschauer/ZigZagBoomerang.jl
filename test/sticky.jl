@testset "Sticky ZigZag 1d" begin
    Random.seed!(1)
    t0 = 0.0
    x0 = [1.0]
    θ0 = [0.8]
    σ = sqrt(0.5)
    μ = 0.9
    ∇ϕ(x, i) = (x[] .- μ)/σ^2


    Γ = sparse(fill(1.0, 1, 1))
    c = 20ones(1)

    Z = ZigZag(Γ, x0*0)
    T = 1000.0
    
    #sspdmp(∇ϕ, t0, x0, θ0, T, c, F::ZigZag, κ, args...; strong_upperbounds = false,  factor=1.5, adapt=false)
    κ = [0.7]
    trace, _, acc = sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ)
    @show acc[1]/acc[2]
    dt = 0.2
    global ts, xs = sep(collect(discretize(trace, dt)))

    w = sqrt(2π)*σ/(sqrt(2π)*σ + exp(-0.5*μ^2/σ^2)/κ[]) # compute weight

    @test abs(mean(getindex.(xs) .!= 0) - w) < 2.5/sqrt(T) # P(X = 0) = w
    @test abs(mean(getindex.(xs)) - w*μ) < 2/sqrt(T) # E X = wμ
    @test abs(mean(getindex.(xs).^2) - w*(σ^2 + μ^2)) < 2.5/sqrt(T) # E X^2 = w(σ^2 + μ^2)
end
