### compute the empirical measure
using Random
using ZigZagBoomerang
N = 10000
σ = 1.0
x_true = randn()
a = randn(N)
y = a*x_true + randn(N)*σ
# ϵ ∼ N(y - a*x, 1)
function tilde∇ϕ(x)
    i = rand(1:N)
    (y[i] - a[i]*x)/σ^2
end

x0, θ0 = randn(), rand([-1,+1])
T = 100.0
c = 50.0
out1, acc = ZigZagBoomerang.pdmp(tilde∇ϕ, x0, θ0, T, 50.0, ZigZag1d())


# make sure the last reflection has opposite sign than the initial reflection
if out1[end][3] == θ0
    out1 = out1[1:end-1]
end

function empirical_measure(out, x_min, x_max)
    x = getindex.(out, 2)
    @assert x_min < minimum(x) && x_max > maximum(x) 
    v = getindex.(out, 3)
    p = sortperm(x)
    xx = [x_min, x[p]..., x_max]
    pxx = Vector{Int}(undef, length(xx))
    pxx[1] = 1
    for i in 2:length(x)+1
        pxx[i] = pxx[i-1] + v[p[i-1]] 
    end
    pxx[end] =pxx[end-1] - 1
    return xx, pxx
end

xx, pxx = empirical_measure(out1, -10.0, +10.0)

