using ZigZagBoomerang
using StaticArrays
using LinearAlgebra
using SparseArrays
using Random


@testset "Vector of SVector" begin
    Random.seed!(1)

    d = 2
    𝕏 = SArray{Tuple{d},Float64,1,d}

    n = 5

    Γ = sprand(n, n, 0.1).*[0.25*SMatrix{d,d}(randn(4)) for i in 1:n, j in 1:n]
    Γ = Γ + Γ' + Diagonal(fill(SMatrix{d,d}(1.0I), n))

    ∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x)

    B
    t0 = 0.0
    x0 = randn(𝕏, n)
    θ0 = [randn(𝕏) for i in 1:n]

    μ = 0*x0
    c = [50.0 for i in 1:n]
    σ = [SMatrix{d,d}(1.0I) for i in 1:n]
    Z = ZigZag(Γ, μ, σ; λref=0.05, ρ=0.8)
    T = 200.0

    @time trace, (tT, xT, θT), (acc, num) = spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
    xs = last.(collect(discretize(trace, 0.01)))
end


@testset "SVector" begin
    Random.seed!(1)

    d = 5
    Γ = sparse(SymTridiagonal(1.0ones(d), -0.4ones(d-1)))
    ∇ϕ!(y, x::T,  Γ) where {T} = T(Γ*x)::SVector
    
    t0 = 0.0
    x0 = @SVector randn(5)
    θ0 = @SVector ones(Float64, 5)

    μ = 0*x0
    c = 50.0
    σ = [SMatrix{d,d}(1.0I) for i in 1:n]
    BP = BouncyParticle(Γ, x0*0, 0.5)
    T = 200.0

    @time trace, (tT, xT, θT), (acc, num) = pdmp(∇ϕ!, t0, x0, θ0, T, c, BP, Γ)
    xs = last.(collect(discretize(trace, 0.01)))
end
