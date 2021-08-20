### compute the empirical measure
using Random
using ZigZagBoomerang
N = 10000
x_true = randn()
a = randn(N)
y = a*x_true + randn(N)*0.1
# ϵ ∼ N(y - a*x, 1)
function ∇ϕ(x)
    i = rand(1:N)
    (y[i] - a[i]*x)*0.01 
end
x0, θ0 = randn(), 1.0
T = 100.0
c = 50.0
out1, acc = ZigZagBoomerang.pdmp(∇ϕ, x0, θ0, T, 50.0, ZigZag1d())


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

