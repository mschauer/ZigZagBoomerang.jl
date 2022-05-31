using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
using LinearAlgebra
using Test, Revise
using ZigZagBoomerang
using SparseArrays
using CSV, DataFrames, Tables
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

include("./benchmark_tools.jl")
include("./reversiblejump.jl")
function logrun!(df, μ, Γ, ℓ, wi, T0 = 200_000_000.0)
    μpost = postmean(μ, ℓ)
    Γpost = postprecision(Γ, ℓ)
    d = length(μpost)
    T = T0/sqrt(sqrt(d)) # assune sub-linear dependence relative to d 
    sΓpost = sparse(Γpost)
    Z = ZigZag(sΓpost, μpost) 
    ∇ϕ(x, i, Γ, μ) = ZigZagBoomerang.idot(Γ, i, x) -  ZigZagBoomerang.idot(Γ, i, μ)
    ki = 1/(sqrt(2*π*σa^2))/(1/wi - 1) 
    x0 = 0rand(d)
    κ = ki*ones(length(x0))
    c = fill(0.001, d)
    θ0 = rand([-0.1,0.1], d)
    t0 = 0.0
    N = 1_000_000
    trace, acc = ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, sΓpost, μpost)
    tsh, xsh = ZigZagBoomerang.sep(collect(discretize(trace, T/N)))
    traceh = [xsh[i][j] for i in 1:length(xsh), j in 1:d]
    pℓ = [marginal_sticky(i, traceh[10:end,:]) for i in 1:d]
    mℓ = sum(traceh[10:end,:], dims = 1)/size(traceh[10:end,:], 1)
    mℓ = Vector(mℓ[:])
    for i in 1:d
        push!(df, (exp  = 0, index = i, ell = ℓ, stat = 1, val = mℓ[i], sampler = "SZZ"))
        push!(df, (exp  = 0, index = i, ell = ℓ, stat = 2, val = pℓ[i], sampler = "SZZ"))
    end
    return df
end

const ℓℓ = [3, 5, 7]
const wi = [0.33, 0.20, 0.15] #WARNING

FileName ="benchmark.csv" 
if !isfile(FileName)
    df = DataFrame(exp = Int[], index = Int[], ell = Int[], stat = Int[], val = Float64[], sampler = String[])
    for (i,ℓ) in enumerate(ℓℓ)
        global df
        df = logrun!(df, μ, Γ, ℓ, wi[i]) 
    end
    CSV.write(FileName, df)
end


# error("")
function benchmark!(df, μ, Γ, ℓℓ, wwi, num_exp = 10)
    println("Make sure that, at every iteration, @elapsed for the Gibbs ≈ @lapsed for sticky ZZ")
    T0 = 1_000_000.0
    N0 = 13_500_000
        for (ii,ℓ) in enumerate(ℓℓ)
        println("Iteration $(ii), ℓ = $(ℓ)")
        μpost = postmean(μ, ℓ)
        Γpost = postprecision(Γ, ℓ)
        d = length(μpost)
        T = T0/(sqrt(sqrt(d)))
        N = N0 ÷ d^2
        sΓpost = sparse(Γpost)
        Z = ZigZag(sΓpost, μpost) 
        ∇ϕ(x, i, Γ, μ) = ZigZagBoomerang.idot(Γ, i, x) -  ZigZagBoomerang.idot(Γ, i, μ)
        # prior w = 0.5
        wi = wwi[ii]
        ki = 1/(sqrt(2*π*σa^2))/(1/wi - 1)
        x0 = 0rand(d)
        κ = ki*ones(length(x0))
        c = fill(0.001, d)
        θ0 = rand([-0.1,0.1], d)
        t0 = 0.0
        if ii == 1
            println("Precompile gibbs and zz")
            ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, sΓpost, μpost)
            z = [abs(d÷2 - i) > 2 for i in eachindex(x0)]  
            reversible_jump(sΓpost, μpost, wi, N, x0, z, σa,  N÷10)
            println("End precompilation")
            println("")
        end
        for jj in 1:num_exp
            println("sticky Zig-Zag:")
            trace, acc = ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ, sΓpost, μpost)
            tsh, xsh = ZigZagBoomerang.sep(collect(discretize(trace, 1.0)))
            traceh = [xsh[i][j] for i in 1:length(xsh), j in 1:d]
            for i in 1:d
                pℓ = marginal_sticky(i, traceh[10:end,:])
                mℓ = sum(traceh[10:end,i])/size(traceh[10:end,:], 1)
                push!(df, (exp  = jj, index = i, ell = ℓ, stat = 1, val = mℓ, sampler = "SZZ"))
                push!(df, (exp  = jj, index = i, ell = ℓ, stat = 2, val = pℓ, sampler = "SZZ"))
            end

            println("Gibbs")
            x0 = 0rand(d)
            z = [abs(d÷2 - i) > 2 for i in eachindex(x0)]  
            @time ββ, ZZ = reversible_jump(sΓpost, μpost, wi, N, x0, z, σa,  10)
            trace2b = [ββ[i][j].*ZZ[i][j] for i in 1:length(ZZ), j in 1:length(ZZ[1])] 
            for i in 1:d
                pℓ = marginal_sticky(i, trace2b[10:end,:])
                mℓ = sum(trace2b[10:end,i])/size(trace2b[10:end,:], 1)
                push!(df, (exp  = jj, index = i, ell = ℓ, stat = 1, val = mℓ, sampler = "GIBBS"))
                push!(df, (exp  = jj, index = i, ell = ℓ, stat = 2, val = pℓ, sampler = "GIBBS"))
            end
            println("")
        end
    end
end

FileName ="benchmark1.csv" 
if !isfile(FileName)
    df1 = DataFrame(exp = Int[], index = Int[], ell = Int[], stat = Int[], val = Float64[], sampler = String[])
    benchmark!(df1, μ, Γ, ℓℓ, wi)
    CSV.write(FileName, df1)
end
