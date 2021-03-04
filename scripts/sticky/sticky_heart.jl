using Revise
using ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using FileIO
using Makie, AbstractPlotting
cd(@__DIR__)
function gridlaplacian(T, m, n)
    S = sparse(T(0.0)I, n*m, n*m)
    linear = LinearIndices((1:m, 1:n))
    for i in 1:m
        for j in 1:n
            for (i2, j2) in ((i + 1, j), (i, j + 1))
                if i2 <= m && j2 <= n
                    S[linear[i, j], linear[i2, j2]] -= 1.
                    S[linear[i2, j2], linear[i, j]] -= 1.

                    S[linear[i, j], linear[i, j]] += 1.
                    S[linear[i2, j2], linear[i2, j2]] += 1.
                end
            end
        end
    end
    S
end


# Define precision operator of a Gaussian random field (sparse matrix operating on `vec`s of `n*n` matrices)
#n = 100
n = 60
const σ2 = 0.5
Γ0 = 2gridlaplacian(Float64, n, n)
Γ = 0.1I + Γ0
mat(x) = reshape(x, (n, n)) # vector to matrix
function mat0(y)
 #   mat(y + 0.1*sign.(y) .- 0.0)
    mat(y  .- 0.1)
end
# Γ is very sparse
@show nnz(Γ)/length(Γ) # 0.000496

# Corresponding Gaussian potential
# ϕ(x', Γ) = 0.5*x'*Γ*x  # not needed

# Define ∇ϕ(x, i, Γ) giving the partial derivative of ϕ(x) with respect to x[i]
#∇ϕ(x, i, Γ, y) = ZigZagBoomerang.idot(Γ, i, x) - ZigZagBoomerang.idot(Γ, i, y)    # more efficient that dot(Γ[:, i], x)
∇ϕ(x, i, Γ, y) = ZigZagBoomerang.idot(Γ, i, x)  + (x[i]-y[i])/σ2 # more efficient that dot(Γ[:, i], x)


# Random initial values
t0 = 0.0
h(x, y) = x^2+(5y/4-sqrt(abs(x)))^2
heart = [ max.(1 - h(x, y), 0).^0.3 for x in range(-1.5,1.5, length=n), y   in range(-1.5,1.5, length=n)]

μ0 = 7.0*vec(heart)
y = μ = μ0 + randn(n*n)
μpost = yhat = (I + Γ)\y
Γpost = (Γ + I)/σ2


#x0 = 0.2μ + 0*randn(n*n)
θ0 = rand([-1.0,1.0], n*n)


# Rejection bounds
c = [norm(Γpost[:, i], 2) for i in 1:n*n]

# Define ZigZag
Z = ZigZag(Γpost, μpost)
# or try the FactBoomerang
#Z = FactBoomerang(Γ, x0*0, 0.1)

κ = 0.01
# Run sparse ZigZag for T time units and collect trajectory
T = 200.0
@time trace, (tT, xT, θT), (acc, num) = spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ, μ; adapt = false)

@time traj0 = collect(discretize(trace, 0.2))

su = false
adapt = false
trace, (t, x, θ), (acc, num), c = @time sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ, Γ, μ;
                                                strong_upperbounds = su ,
                                                adapt = adapt)

@time traj = collect(discretize(trace, 0.2))

# Prepare surface plot
M = Node(mat0(traj[end].second))
scene = Scene()
surface!(scene, M; shading=false, show_axis=false, colormap = :oleron, colorrange = (-3,3))
scale!(scene, 1.0, 1.0, 2.0)
zlims!(scene, -5.0, 5.0)
#display(scene)

scene1, _ = heatmap(abs.([mat(mean(last.(traj0[end÷2:end]))) mat(mean(last.(traj[end÷2:end]))) ]))
scene2, _ = heatmap(abs.([mat(μ0 - mean(last.(traj0[end÷2:end]))) mat(μ0 - mean(last.(traj[end÷2:end]))) ]))

error("stop")
# Movie frames
mkpath(joinpath(@__DIR__, "output"))
for i in eachindex(traj)
    M[] = mat0(traj[i].second)
    FileIO.save(joinpath(@__DIR__, "output", "surfs$i.png"), scene)
end

# Make video
dir = joinpath(@__DIR__, "output")
run(`ffmpeg -y -r 40 -f image2 -i $dir/surfs%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p $(typeof(Z).name)field.mp4`)
