Random.seed!(2)
using SparseArrays
d = 8
S = 1.3I + 0.5sprandn(d, d, 0.1)
const Γ = S*S'

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
    T = 2000.0
    
    #sspdmp(∇ϕ, t0, x0, θ0, T, c, F::ZigZag, κ, args...; strong_upperbounds = false,  factor=1.5, adapt=false)
    κ = [1.5]
    trace, _, acc = sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ)
    @show acc[1]/acc[2]
    dt = 0.2
    global ts, xs = sep(collect(discretize(trace, dt)))

    w = sqrt(2π)*σ/(sqrt(2π)*σ + exp(-0.5*μ^2/σ^2)/κ[]) # compute weight

    @test abs(mean(getindex.(xs) .!= 0) - w) < 2.5/sqrt(T) # P(X = 0) = w
    @test abs(mean(getindex.(xs)) - w*μ) < 5.0/sqrt(T) # E X = wμ
    @test abs(mean(getindex.(xs).^2) - w*(σ^2 + μ^2)) < 5.0/sqrt(T) # E X^2 = w(σ^2 + μ^2)

end


@testset "Sticky SZigZag" begin
    Random.seed!(1)
    global Γ
    d = size(Γ, 1)
    
    
    ∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation
    ∇ϕmoving(t, x, θ, i, t′, F, Γ) = ZigZagBoomerang.idot_moving!(Γ, i, t, x, θ, t′, F) # sparse computation
    
    t0 = 0.0
    x0 = rand(d)
    κ = 1000.0*ones(d) # dont stop, actually
    θ0 = rand([-1.0,-0.5,0.5,1.0], d)


    c = .7*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(0.9Γ, x0*0)
    T = 1000.0

    trace, _, acc = @time sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ, Γ)
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end

@testset "Sticky Boomerang" begin
    Random.seed!(1)
    global Γ
    d = size(Γ, 1)
    
    μ = rand(d)
    ∇ϕ!(y, x, Γ, μ) = mul!(y, Γ, x-μ)

    t0 = 0.0
    x0 = rand(d)
    κ = 1000.0*ones(d) # dont stop, actually
    θ0 = rand([-1.0,-0.5,0.5,1.0], d)


    c = 10.0

    B = Boomerang(sparse(I(d)), μ, 0.5, 0.95, sparse(I(d)))
    T = 1000.0

    trace, _, acc = @time sspdmp(∇ϕ!, t0, x0, θ0, T, c, B, κ, Γ, μ)
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))

    @test mean(abs.(mean(xs) - μ)) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end