using Revise
using Makie, ZigZagBoomerang, SparseArrays, LinearAlgebra

function ϕ(x, i, μ)
    x[i] - μ[i]
end
κ = 1.0
n = 100
μ = zeros(n)
x0 = randn(n)
θ0 = rand((-1.0,1.0), n)
T = 1000.0
c = 10ones(n)

#@time trace0, _ = ZigZagBoomerang.spdmp(ϕ, 0.0, x0, θ0, T, c, ZigZag(sparse(1.0I,n,n), zeros(n)), μ)

@time trace0, _ = ZigZagBoomerang.sspdmp(ϕ, 0.0, x0, θ0, T, c, κ, ZigZag(sparse(1.0I,n,n), zeros(n)), μ)
ts0, xs0 = splitpairs(trace0)

#lines(ts0, getindex.(xs0,1))
#ts0, xs0 = splitpairs(discretize(trace0, 0.01))
#p1 = scatter(getindex.(xs0,1), getindex.(xs0,2), color=(:black, 0.4), markersize=0.05)
#save(joinpath("figures", "spikeandslab.png"), title(p1, "Spike and Slab"))
function prob0(ts0, xs0)
    p = 0.0
    t⁻ = 0.0
    first = true
    for i in eachindex(ts0)
        if xs0[i] == 0.0
            if first
                t⁻ = ts0[i]
                first = false
                continue
            else
                continue
            end
        elseif first == false && xs0[i] != 0.0
            p += ts0[i-1] - t⁻
            first = true
        else
            continue
        end
    end
    p/ ts0[end]
end
prob0(ts0, getindex.(xs0,1))


tp(κ, μ) = 1/(1 + κ*sqrt(2pi)*exp(0.5*(-μ)^2))

using Random
Random.seed!(0)
μ1 = 0.0:0.5:2.0
κ = 0.1:0.1:1.0
A = zeros(length(μ1), length(κ))
B =  zeros(length(μ1), length(κ))
for i in eachindex(μ1)
    for j in eachindex(κ)
        global κ
        global μ1
        μ = fill(μ1[i], n)
        trace0, _ = ZigZagBoomerang.sspdmp(ϕ, 0.0, x0, θ0, T, c, κ[j], ZigZag(sparse(1.0I,n,n), zeros(n)), μ)
        ts0, xs0 = splitpairs(trace0)
        A[i,j] = sum([prob0(ts0, getindex.(xs0, k)) for k in 1:100])/100
    end
end


using CairoMakie, ColorSchemes
mygradcol = ColorSchemes.deep
length(mygradcol)
p1 = Scene()
for i in eachindex(μ1)
    global μ1, κ
    p1 = plot!(κ, A[i,:], color = mygradcol[Int(floor(i*256/5))], label = "mu = 0.0")
    Makie.lines!(κ, tp.(κ, μ1[i]), color = mygradcol[Int(floor(i*256/5))])
end
save("figures/zz_test.png", title(p1, "sticky ZZ"))
