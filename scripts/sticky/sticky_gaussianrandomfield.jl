using Revise
using ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using FileIO
using CairoMakie, AbstractPlotting

include("gridlaplace.jl")

# Define precision operator of a Gaussian random field (sparse matrix operating on `vec`s of `n*n` matrices)
n = 100
Γ = 0.01I + gridlaplacian(Float64, n, n)
mat(x) = reshape(x, (n, n)) # vector to matrix

# Γ is very sparse
@show nnz(Γ)/length(Γ) # 0.000496

# Corresponding Gaussian potential
# ϕ(x', Γ) = 0.5*x'*Γ*x  # not needed

# Define ∇ϕ(x, i, Γ) giving the partial derivative of ϕ(x) with respect to x[i]
∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # more efficient that dot(Γ[:, i], x)

# Random initial values
t0 = 0.0
x0 = randn(n*n)
θ0 = rand([-1.0,1.0], n*n)

# Rejection bounds
c = [norm(Γ[:, i], 2) for i in 1:n*n]

# Define ZigZag
Z = ZigZag(Γ, x0*0)
# or try the FactBoomerang
#Z = FactBoomerang(Γ, x0*0, 0.1)

κ = 1.0
# Run sparse ZigZag for T time units and collect trajectory
T = 20.0
@time trace, (tT, xT, θT), (acc, num) = spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
su = false
adapt = true
trace, (t, x, θ), (acc, num), c = @time sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ, Γ;
                                                strong_upperbounds = su ,
                                                adapt = adapt)

@time traj = collect(discretize(trace, 0.1))

# Prepare surface plot
M = Node(mat(traj[1].second))
p1 = surface(M; shading=false, show_axis=false, colormap = :deep)
scale!(p1, 1.0, 1.0, 7.5)
zlims!(p1, -2.0, 2.0)

# Movie frames
mkpath(joinpath(@__DIR__, "output"))
for i in Iterators.take(eachindex(traj), 100)
    M[] = mat(traj[i].second)
    FileIO.save(joinpath(@__DIR__, "output", "surf$i.png"), p1)
end

# Make video
dir = joinpath(@__DIR__, "output")
run(`ffmpeg -y -r 40 -f image2 -i $dir/surf%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $(typeof(Z).name)field.mp4`)
