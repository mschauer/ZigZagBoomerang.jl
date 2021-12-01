using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
#using Revise
using ZigZagBoomerang
using DataStructures
using LinearAlgebra
using Random
using SparseArrays
using Test
using FileIO
using Statistics
using GLMakie
using Revise

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

Random.seed!(1)


# Define precision operator of a Gaussian random field (sparse matrix operating on `vec`s of `n*n` matrices)
#n = 100
n = 100
const σ2  = 0.5
Γ0 = gridlaplacian(Float64, n, n)
c₁ = 2.0
c₂ = 0.1
Γ = c₁*Γ0^2 +  c₁*I # Γ = c1*2Γ + c2*I 
mat(x) = reshape(x, (n, n)) # vector to matrix
function mat0(y)
    mat(y  .- 0.1)
end
# Γ is very sparse
@show nnz(Γ)/length(Γ) # 0.000496

# Corresponding Gaussian potential
# ϕ(x, Γ, y) = 0.5*x'*Γ*x  + dot(x - y, x - y)/(2*σ2) # not used by the program
I
# Define ∇ϕ(x, i, Γ) giving the partial derivative of ϕ(x) with respect to x[i]
∇ϕ(x, i, Γ, y) = ZigZagBoomerang.idot(Γ, i, x)  + (x[i]-y[i])/σ2 # more efficient that dot(Γ[:, i], x)


# Random initial values
t0 = 0.0
h(x, y) = x^2+(5y/4-sqrt(abs(x)))^2
sz = 3.0
heart = [ max.(1 - h(x, y), 0) for x in range(-1.5-sz,1.5+sz, length=n), y   in range(-1.1-sz,1.9+sz, length=n)]
# image(heart)
μ0 = 5.0*vec(heart) 
y = μ = μ0 + randn(n*n)
μpost = yhat1 = (Γ + I/σ2)\y/σ2 # ok

Γpost = (Γ + I/σ2)
x0 = μpost
θ0 = rand([-1.0,1.0], n*n)


# Rejection bounds
c = [norm(Γpost[:, i], 2) for i in 1:n*n]

# Define ZigZag
Z = ZigZag(Γpost, μpost)
# or try the FactBoomerang
#Z = FactBoomerang(Γ, x0*0, 0.1)

κ1 = 0.4*ones(length(x0))
# Run sparse ZigZag for T time units and collect trajectory
T = 100.0
# @time trace0, (tT, xT, θT), (acc, num) = spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ, μ; adapt = false)
# @time traj0 = collect(discretize(trace0, 0.2))

su = false
adapt = false
# trace, (t, x, θ), (acc, num), c = @time sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ1, Γ, μ;
#                                                 strong_upperbounds = su ,
#                                                 adapt = adapt)
sspdmp(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ1, Γ, μ; strong_upperbounds = su ,
    adapt = adapt) 
println("Old implementation")
trace, (t, x, θ), (acc, num), c = @time sspdmp(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ1, Γ, μ;
                                                strong_upperbounds = su ,
                                                adapt = adapt)

println("New implementation")                                                
ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ1, Γ, μ;
                                                strong_upperbounds = su ,
                                                adapt = adapt)
ZigZagBoomerang.sspdmp2(∇ϕ, t0, x0, θ0, T, c, nothing, Z, κ1, Γ, μ;
                                                strong_upperbounds = su ,
                                                adapt = adapt)

error("")
# trace, (t, x, θ), (acc, num), c=  @time sspdmp(∇ϕ, t0, x0, θ0, T, c, nothing, z, κ2, args...;  kwargs...)
@time traj = collect(discretize(trace, 0.2))

#display(scene)

#scene1, _ = heatmap(abs.([mat(mean(last.(traj0[end÷2:end]))) mat(mean(last.(traj[end÷2:end]))) ]))
scene2a, _ = heatmap(abs.(mat(μ0 - mean(last.(traj[end÷2:end])))))
scene2b, _ = heatmap(abs.(mat(μ0 - mean(last.(traj[end÷2:end])))))

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
