using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
using LinearAlgebra
using Test
using ZigZagBoomerang
using SparseArrays

d = 40
const σa = 6.0
const σb = 0.6

R = [i == j ? 1 : (j - i == -1 ? -1 : 0) for i in 2:d, j in 1:d]
c1 = [(i == 1 || i == d-1) ? -1 : -1 for i in 1:d-1]
Γ = R'*R/σb^2 + I/σa^2
ci = 2*[0.5+cos(i) for i in range(0,2pi,length=d-1)]
μ = -inv(Γ)*R'/σb^2*(ones(d-1).*ci)


# μpost = [zeros(d); μ; zeros(d)]
function postmean(μ, ℓ)
    d = length(μ)
    μpost = zeros(d*ℓ)
    jj = (ℓ + 1) ÷ 2
    [μpost[(j-1)*d + i] = (j == jj ? μ[i] : 0.0) for i in 1:d, j in 1:ℓ] 
    μpost
end

# Γpost =  [Γ 0I 0I; 0I Γ 0I; 0I 0I Γ]
function postprecision(Γ, ℓ)
    d = size(Γ,1)
    Γpost = zeros(d*ℓ, d*ℓ)
    [Γpost[(i-1)*d + 1 : (i)*d , (j-1)*d + 1 : (j)*d ] = (i == j ? Γ : 0I(d)) for i in 1:ℓ, j in 1:ℓ] 
    Γpost
end

include("./reversiblejump.jl")
function benchmark(μ, Γ, ℓℓ)
    println("Make sure that, at every iteration, @elapsed for the Gibbs ≈ @lapsed for sticky ZZ")
    T = fill(5000.0, length(ℓℓ))
    N = fill(100, length(ℓℓ))
        for (ii,ℓ) in enumerate(ℓℓ)
        println("Iteration $(ii), ℓ = $(ℓ)")
        μpost = postmean(μ, ℓ)
        Γpost = postprecision(Γ, ℓ)
        d = length(μpost)
        sΓpost = sparse(Γpost)
        Z = ZigZag(sΓpost, μpost) 
        ∇ϕ(x, i, Γ, μ) = ZigZagBoomerang.idot(Γ, i, x) -  ZigZagBoomerang.idot(Γ, i, μ)
        # prior w = 0.5
        wi = 0.35
        ki = 1/(sqrt(2*π*σa^2))/(1/wi - 1)
        x0 = 0rand(d)
        κ = ki*ones(length(x0))
        c = fill(0.001, d)
        θ0 = rand([-0.1,0.1], d)
        t0 = 0.0
        if ii == 1
            println("Precompile gibbs and zz")
            ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T[ii], c, nothing, Z, κ, sΓpost, μpost)
            z = [abs(d÷2 - i) > 2 for i in eachindex(x0)]  
            reversible_jump(sΓpost, μpost, wi, N[ii], x0, z, σa,  N[ii]÷10)
            println("End precompilation")
            println("")
        end
        println("sticky Zig-Zag:")
        trace, acc = ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T[ii], c, nothing, Z, κ, sΓpost, μpost)
        println("Gibbs")
        x0 = 0rand(d)
        z = [abs(d÷2 - i) > 2 for i in eachindex(x0)]  
        @time ββ, ZZ = reversible_jump(sΓpost, μpost, wi, N[ii], x0, z, σa,  N[ii]÷10)
        #trace2 = [ββ[i].*ZZ[i] for i in 1:length(ZZ)]
        # trace2b = [ββ[i][j].*ZZ[i][j] for i in 1:length(ZZ), j in 1:length(ZZ[1])] 
        println("")
    end
end

ℓℓ = [3,5]
benchmark(μ, Γ, ℓℓ)