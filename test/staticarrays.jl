using ZigZagBoomerang
using StaticArrays
using LinearAlgebra
using SparseArrays
using Random
using Test
using Statistics

include("blockchol.jl")

function tofull(Γ::AbstractMatrix)
    n = size(Γ, 1)
    d = size(Γ[1,1],1) 
    Γfull = Matrix(Γ)
    [Γfull[i[2],j[2]][i[1],j[1]] for i in vec(CartesianIndices((d,n))), j in vec(CartesianIndices((d,n)))]
    #[Γfull[i[1],j[1]][i[2],j[2]] for i in vec(CartesianIndices((n,d))), j in vec(CartesianIndices((n,d)))]
end
function tofull(x::Vector)
    n = size(x, 1)
    d = size(x[1],1) 
    [x[i[2]][i[1]] for i in vec(CartesianIndices((d,n)))]
end
@testset "Vector of SVector" begin
    Random.seed!(2)

    d = 2
    𝕏 = SArray{Tuple{d},Float64,1,d}

    n = 5

    I_nd = Diagonal(fill(SMatrix{d,d}(1.0I), n))

    Γ0 = sparse(Tridiagonal([SMatrix{d,d}([0.0 -0.4;-0.0 -0.0]) for i in 1:n-1], fill(SMatrix{d,d}([1.0 -0.4;-0.4 1.0]), n), [SMatrix{d,d}([0.0 -0.0;-0.4 -0.0]) for i in 1:n-1]))
    Γ = Γ0*Γ0

#    ∇ϕ(x, i, Γ) = sum(Γ[i,j]*x[j] for j in 1:n)
    ∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x)
    t0 = 0.0
    x0 = randn(𝕏, n)
    θ0 = [randn(𝕏) for i in 1:n]

    μ = 0*x0
    c = [50.0 for i in 1:n]
    σ = [SMatrix{d,d}(1.0I) for i in 1:n]
    Z = ZigZag(Γ, μ, σ; λref=0.02, ρ=0.0) # need refreshments!
    T = 1000.0

    @time trace, (tT, xT, θT), (acc, num) = spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)

    xs = last.(collect(discretize(trace, 2.0)))
    L = lchol(Matrix(Γ))
    Σ = cholinverse!(L, Matrix(I_nd))
    @test_broken mean(norm.(cov(xs) - Σ)) < 3/sqrt(T)
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
    c = 100.0
    σ = [SMatrix{d,d}(1.0I) for i in 1:n]
    BP = BouncyParticle(Γ, x0*0, 0.5, 0., nothing, MMatrix{d,d}(sparse(cholesky(Symmetric(Γ)).L)))
    T = 800.0

    @time trace, (tT, xT, θT), (acc, num) = pdmp(∇ϕ!, t0, x0, θ0, T, c, BP, Γ)
    xs = last.(collect(discretize(trace, 0.01)))
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 3/sqrt(T)
end
