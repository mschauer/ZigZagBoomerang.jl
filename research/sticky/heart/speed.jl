if !@isdefined timeels
  errs = Any[]
  timeels = Float64[]
  nevents = Int[]
  nsfull = 50:50:600
  ns = Int[]
end
for i in nsfull[length(timeels)+1:end]
    global n = i
    @show n

    include("heart_sparse2.jl")
    push!(ns, n)
    push!(errs, err)
    push!(timeels, timeel)
    push!(nevents, nevent)
end

#t = [8.6, 29.2, 71.0, 151.5]

if true
    ns = nsfull[1:length(nevents)]
    using GLMakie
    fig = Figure()
    ax = Axis(fig[1,1], yscale=log2, xscale=log2)
    l1 = lines!(ax, ns.^2, timeels)
    scatter!(ax, ns.^2, timeels)
    
    ops = @. ns^2*log2(ns^2)
    l2 = lines!(ax, ns.^2, ops*timeels[end÷2]/ops[end÷2], linestyle=:dot) 
    Legend(fig[1,2], [l1, l2], ["empirical", "theoretical"])
    fig
end

#=
100
  8.629198 seconds (41.57 M allocations: 2.031 GiB, 5.37% gc time, 18.82% compilation time)
acc 0.3556902142990697
length(trace2.events) = 2173074

200
 29.176336 seconds (155.35 M allocations: 7.416 GiB, 11.00% gc time, 2.13% compilation time)
acc 0.35670809301812756
length(trace2.events) = 8660599

300

 71.003698 seconds (348.01 M allocations: 16.590 GiB, 6.27% gc time, 0.82% compilation time)
acc 0.35691171087612455
length(trace2.events) = 19461510
sparsity(uT) = 0.13973333333333332

400
151.548660 seconds (618.12 M allocations: 29.479 GiB, 6.35% gc time, 0.38% compilation time)
acc 0.3572412791894073
length(trace2.events) = 34594761
sparsity(uT) = 0.137475
=#
#= fig = Figure()
ax = Axis(fig[1,1], yscale=log2, xscale=log2)
lines!(ax, 2:1000, map(x->x*log(x), 2:1000))
scatter!(ax, 2:1000, map(x->x*log(x), 2:1000))
lines!(ax, 2:1000,  2:1000)
scatter!(ax, 2:1000, 2:1000)
lines!(ax, 2:1000,  10*(2:1000))

fig=#
