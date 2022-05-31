using Pkg
Pkg.activate(@__DIR__)
#cd(@__DIR__)
using Revise
using ZigZagBoomerang
const ZZB = ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using FileIO
using Statistics

using Revise
function gridlaplacian(::Type{T}, m, n) where T
    linear = LinearIndices((1:m, 1:n))
    Is = Int[]
    Js = Int[]
    Vs = T[]
    for i in 1:m
        for j in 1:n
            for (i2, j2) in ((i + 1, j), (i, j + 1))
                if i2 <= m && j2 <= n
                    push!(Is, linear[i, j])
                    push!(Js, linear[i2, j2])
                    push!(Vs, -1)
                    push!(Is, linear[i2, j2])
                    push!(Js, linear[i, j])
                    push!(Vs, -1)
                    push!(Is, linear[i, j]) 
                    push!(Js, linear[i, j]) 
                    push!(Vs, 1)
                    push!(Js, linear[i2, j2])
                    push!(Is, linear[i2, j2])
                    push!(Vs, 1)
                end
            end
        end
    end
    sparse(Is, Js, Vs)
end

Random.seed!(1)


# Define precision operator of a Gaussian random field (sparse matrix operating on `vec`s of `n*n` matrices)

n = 100
ℂ = CartesianIndices((n,n))
𝕃 = LinearIndices(ℂ)
const σ2  = 0.5
Γ0 = gridlaplacian(Float64, n, n)
c₁ = 2.0
c₂ = 0.1
Γ = c₁*Γ0 + c₂*I 
mat(x) = reshape(x, (n, n)) # vector to matrix

# Γ is very sparse
nnz_(x) = sum(x .!= 0)
nnz_(x::SparseVector) = nnz(x)
nnz_(x::SparseMatrixCSC) = nnz(x)

sparsity(Γ) = nnz_(Γ)/length(Γ)
@show sparsity(Γ) # 0.000496

# Corresponding Gaussian potential
# ϕ(x, Γ, y) = 0.5*x'*Γ*x  + dot(x - y, x - y)/(2*σ2) # not used by the program
I
# Define ∇ϕ(x, i, Γ) giving the partial derivative of ϕ(x) with respect to x[i]
∇ϕ(x, i, Γ, y) = ZigZagBoomerang.idot(Γ, i, x)  + (x[i] - y[i])/σ2 # idot(Γ, i, x) more efficient that dot(Γ[:, i], x)

# Random initial values
t0 = 0.0
h(x, y) = x^2+(5y/4-sqrt(abs(x)))^2
sz = 3.0
heart = [ max.(1 - h(x, y), 0) for x in range(-1.5-sz,1.5+sz, length=n), y   in range(-1.1-sz,1.9+sz, length=n)]

μ0 = 5.0*vec(heart) 
y = μ = μ0 + sqrt(σ2)*randn(n*n)
μpost = yhat1 = (Γ + I/σ2)\y/σ2 # ok

Γpost = (Γ + I/σ2)

droptol!_(x::Vector, tol) = map!(x -> abs(x) > tol ? x : 0.0, x, x)
x0 = zero(μpost)
θ0 = rand([-1.0,1.0], n*n)
@show sparsity(x0)

# Rejection bounds
c = fill(0.1, n*n)
c2 = fill(5.0, n*n)

# Define ZigZag
Z = ZigZag(Γpost, μpost)

κ1 = 1*ones(length(x0))
# Run sparse ZigZag for T time units and collect trajectory
T = 300.0
using Graphs
G = SimpleGraph(Γ)

#G2 = ZZB.saturate(G)
C = Graphs.Coloring(9, Vector(vec([1 + i%3 + 3*(j%3) for i in 1:n, j in 1:n])))

su = false
adapt = false
if (!@isdefined lastn) || lastn != n
    lastn = n
    println("New implementation")                                                
    @time trace1, acc1 = ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ1, Γ, μ;
                                                    strong_upperbounds = su ,
                                                    adapt = adapt, progress=true)
end
println("Async implementation")  
@time trace2, acc2 = ZigZagBoomerang.sspdmp4(C, ∇ϕ, t0, x0, θ0, T, c2, nothing, Z, κ1, Γ, μ;
                                                adapt = adapt, progress=true)
#

using Colors
cols = [rand(RGB) for i in 1:9]
div1(a,b) = (a-1)÷b + 1
region(i) = div1(4*i, n*n) 
error("hier")

using GLMakie
#image(map(i->cols[region(i[1], i[2])], CartesianIndices((n,n))), interpolate=false)
#image(map(i->cols[i], mat(C.colors)), interpolate=false)
#image(mat(map(i->cols[region(i)], 1:n^2)), interpolate=false)

error("hier")
ŷ1 = mat(mean(trace1))
ŷ2 = mat(mean(trace2))

using GLMakie
fig1 = fig = Figure()

ax = [Axis(fig[1,i], title = "posterior mean $i") for i in 1:2]
heatmap!(ax[1], ŷ1, colorrange=(-1,6))
heatmap!(ax[2], ŷ2, colorrange=(-1,6))

ax = [Axis(fig[2,i], title = "error $i") for i in 1:2]
heatmap!(ax[1], abs.(mat(μ0) - ŷ1), colorrange=(-2,2))
heatmap!(ax[2], abs.(mat(μ0) - ŷ2), colorrange=(-2,2))
ℂ = CartesianIndices((n,n))
r = LinearIndices(ℂ)[CartesianIndex(n÷2, n÷2)]
st1, sx1 = ZigZagBoomerang.sep(collect(ZigZagBoomerang.subtrace(trace1, r-5:r+5)))
st2, sx2 = ZigZagBoomerang.sep(collect(ZigZagBoomerang.subtrace(trace2, r-5:r+5)))

ax = [Axis(fig[3,i], title = "trace $i") for i in 1:2]
lines!(ax[1], st1, getindex.(sx1,1))
lines!(ax[2], st2, getindex.(sx2,1))

#@time traj1 = collect(discretize(trace1, 0.2))
#@time traj2 = collect(discretize(trace2, 0.2))

#display(scene)

#scene1, _ = heatmap(abs.([mat(mean(last.(traj0[end÷2:end]))) mat(mean(last.(traj[end÷2:end]))) ]))


error("stop")
if false

# Prepare surface plot
    M = Node(mat0(traj[end].second))
    scene = Scene()
    surface!(scene, M; shading=false, show_axis=false, colormap = :oleron, colorrange = (-3,3))
    Makie.scale!(scene, 1.0, 1.0, 2.0)
    zlims!(scene, -5.0, 5.0)
    # Movie frames
    mkpath(joinpath(@__DIR__, "output"))
    for i in eachindex(traj)
        M[] = mat0(traj[i].second)
        FileIO.save(joinpath(@__DIR__, "output", "surfs$i.png"), scene)
    end

    # Make video
    dir = joinpath(@__DIR__, "output")
    run(`ffmpeg -y -r 40 -f image2 -i $dir/surfs%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $(typeof(Z).name)field.mp4`)
end
error()

# Save hearts
save("hearttrue.png", image(mat(μ0)))
save("heart.png", image(mat(y)))
save("hearthat.png", image(mat(yhat)))
save("heartpostmeana.png", image(mat(mean(last.(traj0[end÷2:end])))))
save("heartpostmeanb.png", image(mat(mean(last.(traj[end÷2:end])))))
save("hearterrora.png", scene2a)
save("hearterrorb.png", scene2b)

mean((mat(μ0 - mean(last.(traj0[end÷2:end])))).^2)
mean((mat(μ0 - mean(last.(traj[end÷2:end])))).^2)

@show mean(abs.(mat(μ0 - mean(last.(traj0[end÷2:end])))))
@show mean(abs.(mat(μ0 - mean(last.(traj[end÷2:end])))))
@show extrema(μ0)


@show  extrema(y)


@show extrema(yhat)


@show extrema((mat(mean(last.(traj0[end÷2:end])))))
@show extrema((mat(mean(last.(traj[end÷2:end])))))


@show extrema(abs.(mat(μ0 - mean(last.(traj0[end÷2:end])))))

@show extrema(abs.(mat(μ0 - mean(last.(traj[end÷2:end])))))
