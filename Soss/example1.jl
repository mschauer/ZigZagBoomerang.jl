using Pkg
Pkg.activate(@__DIR__)
using Soss
using ZigZagBoomerang




include("zigzag.jl")


m = @model x begin
    α ~ Normal()
    β ~ Normal()
    yhat = α .+ β .* x
    y ~ For(eachindex(x)) do j
        Normal(yhat[j], 2.0)
    end
end



x = randn(20);
obs = -0.1 .+ 2x + 1randn(20); 
posterior = m(x=x) | (y=obs,)


T = 100.0
trace, final, (num, acc) = @time zigzag(posterior, T)

# trace is a continous object, discretize to obtain samples
ts, xs = ZigZagBoomerang.sep(discretize(trace, 0.1))

xs = xform(posterior).(xs)

using Plots
p = plot(ts, getindex.(xs, 1))
plot!(ts, getindex.(xs, 2), color=:red)
p2 = plot(first.(xs), last.(xs))
p
