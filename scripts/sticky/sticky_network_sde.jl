using Pkg
using StaticArrays
using LinearAlgebra
using SparseArrays
using Random
cd(@__DIR__)
Random.seed!(1)
const d = 2
const Point = SArray{Tuple{d},Float32,1,d}
using Makie
T = (0:0.1f0:200)[2:end]
n = 50
σ = 0.1f0
μ = 0.2f0
a = 0.8f0
b = 0.15f0
B = sparse((-μ - a + b)*I, n, n)
B1 = sparse(-1f0*I, n, n)
B2 = sparse(1f0*I, n, n)

for i in 1:n
    k1 = rand(1:n-1)
    k1 += (i <= k1)
    k2 = rand(1:n-2)
    k2 += (min(i, k1) <= k2)
    k2 += (max(i, k1) <= k2)
    @assert i ≠ k1
    @assert i ≠ k2
    @assert k1 ≠ k2

    B[i, k1] = a
    B[i, k2] = -b

    B1[i,k1] = 1
    B2[i, k2] = -1

end

x0 = randn(Point, n)
x = copy(x0)
xs = [x]
ts = [0.0]
t = 0.0
# xp = Node(x)
r = 1.
# R = Rect2D(-r,-r,2r,2r)
# p1 = scatter(xp, markersize=3, limits=R)
# display(p1)
nlz(x) = x/norm(x)
for i in eachindex(T)
    global t, x, xs
    local dt = T[i] - t
    @assert dt > 0
#    x .= x + 0.1*nlz.((B*x))*dt + σ*sqrt(dt)*randn(Point, n)
    x = x + (B*x)*dt + σ*sqrt(dt)*randn(Point, n)
    # xp[] = x
    t = T[i]
    # yield()
    # sleep(0.0001)
    push!(ts, t)
    push!(xs, copy(x))
end




flatten = Base.Iterators.flatten
#flat(x) = collect(Base.Iterators.flatten(x))
function posterior(ts, xs, n, μ, σ)
    v = [c for c in CartesianIndices((n,n)) if c[1] != c[2]]
    a1 = 2
    a2 = 1
    i = 1
    x = xs[i]
    s = (ts[i+1] - ts[i])*σ^(-2.0) # dt/σ^2
    dx = xs[i+1] - (xs[i] + xs[i]*(ts[i+1] - ts[i])*(-μ)) # dxt + xt*dt*-μ

    z = [(x[c[a2]] - x[c[a1]])'*(σ^(-2.0)*dx[c[a1]]) for c in v]
    Γ = [(c1[a1]==c2[a1])*s*(x[c1[a2]] - x[c1[a1]])'*(x[c2[a2]] - x[c2[a1]]) for c1 in v, c2 in v]

    for i in 2:length(ts)-1

        x = xs[i]
        s = (ts[i+1] - ts[i])*σ^(-2.0)

        for i1 in eachindex(v) # same as z += [(x[c[a2]] - x[c[a1]])'*(σ^(-2.0)*dx[c[a1]]) for c in v]
            c = v[i1]
            dx = xs[i+1][c[a1]] - (xs[i][c[a1]] + xs[i][c[a1]]*(ts[i+1] - ts[i])*(-μ))
            z[i1] += (x[c[a2]] - x[c[a1]])'*(σ^(-2.0)*dx)
        end
        for i1 in eachindex(v) # same as Γ += [(c1[a1]==c2[a1])*s*(x[c1[a2]] - x[c1[a1]])'*(x[c2[a2]] - x[c2[a1]]) for c1 in v, c2 in v]
            c1 = v[i1]
            i2 = i1
            while i2 <= length(v)
                c2 = v[i2]
                c1[a1]!=c2[a1] && break
                uu = s*(x[c1[a2]] - x[c1[a1]])'*(x[c2[a2]] - x[c2[a1]])
                Γ[i1, i2] += uu
                if i1 != i2
                    Γ[i2, i1] += uu
                end
                i2 += 1
            end
        end
    end
    Γ, z
end


Γ, z = posterior(ts, xs, n, μ, σ)
size(Γ)
50*49
γ0 = 0.02
post = cholesky(Hermitian(Γ + γ0*I))\z
(Γ + γ0*I)*post - z

function img(x, n)
    B2 = zeros(n, n)
    v = [c for c in CartesianIndices((n,n)) if c[1] != c[2]]
    for (p, c) in zip(x, v)
        B2[c] = p
    end
    B2
end
B2a = img(post, n)
p1 = heatmap([B2a' Matrix(B - Diagonal(B))], colormap=:berlin, colorrange = (-1, 1))
using FileIO
#save("figures/sparseinteraction.png", p1)


[B2a' Matrix(B - Diagonal(B))]


using StatsBase, Statistics

Γ0 = sparse(Γ)
μ = cholesky(Hermitian(Γ + γ0*I))\z


using ZigZagBoomerang
# ϕ(x', Γ) = 0.5*x'*Γ*x - z'*x # not needed

# Define ∇ϕ(x, i, Γ) giving the partial derivative of ϕ(x) with respect to x[i]
∇ϕ(x, i, Γ, z) = ZigZagBoomerang.idot(Γ, i, x) - z[i] # more efficient that dot(Γ[:, i], x)
# Random initial values
t0 = 0.0
m = length(z)
x0 = μ
θ0 = rand([-1.0,1.0], m)

# Rejection bounds
c = [1e-4 for i in 1:m]

# Define ZigZag
Z = ZigZag(Γ0, μ)
# or try the FactBoomerang
#Z = FactBoomerang(Γ, x0*0, 0.1)

# Run sparse ZigZag for T time units and collect trajectory
T0 = 20.
@time trace, (tT, xT, θT), (acc, num) = spdmp(∇ϕ, t0, x0, θ0, T0, c, Z, Γ0, z; structured = true, adapt = true)
@time traj0 = collect(discretize(trace, 0.1))

κ = 0.05*ones(length(x0))
c = [1e-4 for i in 1:m]
#@time trace, (tT, xT, θT), (acc, num) = ZigZagBoomerang.sspdmp(∇ϕ, t0, x0, θ0, T0, c, Z, κ, Γ0, z; adapt = true)
@time trace, (tT, xT, θT), (acc, num) = ZigZagBoomerang.sspdmp(∇ϕ, t0, x0, θ0, T0, c, Z, κ, Γ0, z; structured = true, strong_upperbounds = false, adapt = true)


@time traj = collect(discretize(trace, 0.1))


using ColorSchemes

A = Node(img(traj[1].second, n))

p2 = heatmap(A, colormap=:berlin, colorrange = (-1, 1))
display(p2)
for (t, x) in traj
    A[] = img(x, n)
    yield()
    sleep(0.01)
end

using Statistics

xhat = [median(getindex.(last.(traj), i)) for i in eachindex(μ)]
xhat0 = [median(getindex.(last.(traj0), i)) for i in eachindex(μ)]

p3 = heatmap([img(xhat, n)'  Matrix(B - Diagonal(B))], colormap=:berlin, colorrange = (-1, 1))

norm(img(xhat, n)' - Matrix(B - Diagonal(B)))
norm(img(μ, n)' - Matrix(B - Diagonal(B)))

#=
# animation settings
n_frames = 30
framerate = 30

p2 = heatmap(A, colormap=:berlin, colorrange = (-1, 1))
record(figure, "boids_animation.mp4", traj; framerate = framerate) do (t, x)
    A[] = img(x, n)
end
=#

using AbstractPlotting, CairoMakie, GeometryBasics
fi0 = linesegments(repeat(1:n*n, inner = 2), vec([(vec(Matrix(B)-Diagonal(B))) vec(img(post,n)')]'), linewidth = 0.3, resolution = (1200,900))
scatter!(1:n*n, vec(img(post,n)'), color=(sqrt∘abs).(vec(Matrix(B)-Diagonal(B))), colormap=:berlin, markersize=7,
            strokewidth = 0.5,marker = map(x -> x != 0 ? GeometryBasics.HyperSphere{2} : :x, vec(Matrix(B)-Diagonal(B))))
fi0

# fi0 = scatter(1:n*n, vec(img(xhat0,n)'), color=(sqrt∘abs).(vec(Matrix(B)-Diagonal(B))), colormap=:berlin, markersize=5, alpha=0.3)
fi = linesegments(repeat(1:n*n, inner = 2), vec([(vec(Matrix(B)-Diagonal(B))) vec(img(xhat,n)')]'), linewidth = 0.3, resolution = (1200,900))
scatter!(1:n*n, vec(img(xhat,n)'), color=(sqrt∘abs).(vec(Matrix(B)-Diagonal(B))), colormap=:berlin, markersize=7,
            strokewidth = 0.5,marker = map(x -> x != 0 ? GeometryBasics.HyperSphere{2} : :x, vec(Matrix(B)-Diagonal(B))))
fi
save("figures/sparseinteraction.png", fi0)
save("figures/sparseinteractionsticky.png", fi)


function confusion(xhat)
    Dict(
    "false negative" => sum((vec(img(xhat,n)') .== 0) .& (vec(Matrix(B)-Diagonal(B)) .!= 0)),
    "true positive" => sum((vec(img(xhat,n)') .!= 0) .& (vec(Matrix(B)-Diagonal(B)) .!= 0)),
    "true negative" => sum((vec(img(xhat,n)') .== 0) .& (vec(Matrix(B)-Diagonal(B)) .== 0)),
    "false positive" => sum((vec(img(xhat,n)') .!= 0) .& (vec(Matrix(B)-Diagonal(B)) .== 0)),
    )
end
confusion(xhat)
confusion(abs.(xhat0) .> 0.1265)
