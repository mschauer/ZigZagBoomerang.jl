using Pkg
Pkg.activate(@__DIR__)
using Soss
using ZigZagBoomerang

m = @model x begin
    α ~ SpikeMixture(Normal(), 0.3) # 0.2*Normal() + 0.8*Dirac(0)
    β ~ SpikeMixture(Normal(), 0.3)
    yhat = α .+ β .* x
    y ~ For(eachindex(x)) do j
        Normal(yhat[j], 2.0)
    end
end

vars = (α=1, β=2)

x = randn(20);
obs = -0.1 .+ 2x + 1randn(20); 
T = 200.0
posterior = m(x=x) | (y=obs,)

bm_ = Soss.basemeasure(posterior, (β=0.0, α=0.0));
bm = getindex.(Ref(bm_.data), (keys(vars)));
@show kappa.(bm)



include("sparsezigzag.jl")
trace, final, (num, acc) = @time sparse_zigzag(posterior, T, c=50)

ts, xs = ZigZagBoomerang.sep(discretize(trace, 0.1))

xs = xform(posterior).(xs)

using Plots
p = plot(ts, getindex.(xs, 1))
plot!(ts, getindex.(xs, 2), color=:red)
savefig("tracesticky.png")
p2 = plot(first.(xs), last.(xs))
savefig("phasesticky.png")
p
