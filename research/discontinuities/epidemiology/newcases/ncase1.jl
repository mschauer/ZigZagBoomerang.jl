# Case 1: A1 -> A2 -> A3 -> A4
# when γ = 1.0, always cross notification time
# when γ ≈ 0.0 almost never cross notification time
# when β large stay close to notification time
# when β is small get far away from notification time
using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
using Random, DataStructures, Revise
using CSV, Tables
# number of individuals
const N = 1000
# reduction after notification 
const γ = 0.1
# rate of 1 -> 2
const β = 0.5 # β
# rate of 2 -> 3 with exponential rate equal to δ2
const δ2 = 0.1
# observation time
const T = 50.0
# infectivity metric 
d(i,j) = (abs(j-i) <= 5 && j != i) ? 0.2 : 0.0
# d(i,j) = 0.5

# baseline susceptibility 
const ξ = zeros(N) .+ 1.0
# baseline infectivity
const ϑ = zeros(N) .+ 1.0
### Define infectivity pressure of `i` to `j`

### simulate forward and plot
include("./../virus_forward.jl")
# include("./../final_zz_epi.jl")
include("./../plotting.jl")
include("./../opt_zz_epi3.jl")

Random.seed!(2)
# Initialize population 
case1 = N ÷ 2
xint = (1:N .== case1)*1
S, it, nobs, robs, u0 = forward_simulation(xint, ξ, ϑ, γ, β, δ2, T)
x0, v0, s0, tag0, ind0 = u0
println(" (Unobserved) number of infected: at time $T:   $(sum(it .< T - eps()))")
println(" (Observed) number of notified at time $T:   $(sum(nobs .< T - eps()))")
println(" (Observed) number of removed at time $T:   $(sum(robs .< T - eps()))")
###############################
plotting = false
if plotting
    fig = Figure(resolution = (1800, 1200))
    fig = plot_forard_model!(fig, 1, it, nobs, robs, T, N)
    display(fig)
end
# the first agent is fixed
v0[case1] = 0.0
clock = 200.0
# ℬ = Baseline(ξ, ϑ)
# pressure(ℬ::Baseline, i,j) = ℬ.ξ[i]*ℬ.ϑ[j]*d(i, j)
EDG = zeros(N,N)
[EDG[i,j] = ξ[i]*ϑ[j]*d(i,j) for i in 1:N for j in 1:N]
p0 = Pressure(EDG, γ)
## new input
s1 = State1(x0[1:N], v0[1:N], s0[1:N])
s2 = State2(x0[N+1:end], tag0[N+1:end], ind0[N+1:end])
f0 = sortperm([s1.x; s2.y])
b0 = sortperm(f0)
function mapobstacles(s1::State1, s2::State2, N)
    x,v,s = s1.x, s1.v, s1.s
    id = s2.id
    G = Dict([i => findall(j -> j == i, id) .+ N for i in eachindex(x)])
end
G0 = mapobstacles(s1, s2, N)
u0 = State(s1,s2, f0, b0, G0)


Ξ, u0, u, p, count = zz_epi(u0, p0, clock)
# @assert sum(diff(ordered_state(u)) .<0.0) == 0
[@assert abs(u.S1.λ1[i] - Lambda(i,u, p)) <= 1.0e-4 for i in 1:N]
[@assert abs(u.S1.λ2[i] - Gamma(i,u, p)) <= 1.0e-4 for i in 1:N]
error("")
if plotting
    fig = plot_trace_zz!(fig, 2, u0, Ξ, N, T)
    display(fig)
end
xx = getindex.(Ξ,3)
xtrace = reshape(xx)
tt = getindex.(Ξ,1)
function meantrace(tt, xtrace)
    T = tt[end]
    n1,n2 = size(xtrace)
    res = zeros(n2)
    for i in 2:n1
        res .+= (tt[i]-tt[i-1])*(xtrace[i-1, :] .+ xtrace[i, :])./2
    end
    res./T
end

inf1 = meantrace(tt, xtrace)
# 
A = [ind0[1:N] inf1[1:N] x0[1:N]]
@show [ind0[1:N] inf1[1:N] x0[1:N]] 
CSV.write("./csv/case_1.csv",  Tables.table(A), writeheader=false)

if plotting
    using LinearAlgebra, SparseArrays, ZigZagBoomerang
    FB = BouncyParticle(sparse(I(length(x0))), zero(u0.S1.x), 0.0)
    trace2 = ZigZagBoomerang.PDMPTrace(FB, 0.0, u0.S1.x, u0.S1.v, ones(Bool, length(x0)), [(t, x, θ, nothing) for (t, i, x, θ) in Ξ])
    trc2 = collect(discretize(trace2, 0.01))
    display(fig)
    X = reshape(getindex.(trc2, 2))
    fig3 = fig[1,3]
    ax1, h1 = hist(fig3[1,2], X[:,2], color = (:blue, 0.1))
    vlines!(ax1, [it[2]], color = :red)
    vlines!(ax1, [0.0, nobs[2]], color = :red, linestyle = :dash)  
    ax1.title ="A2"

    ax2, h1 = hist(fig3[2,1], X[:,3], color = (:blue, 0.1))
    vlines!(ax2, [it[3]], color = :red) 
    vlines!(ax2, [0.0, nobs[3]], color = :red, linestyle = :dash)  
    ax2.title ="A3"

    ax3, h1 = hist(fig3[2,2], X[:,4], color = (:blue, 0.1))
    vlines!(ax3, [it[4]], color = :red) 
    vlines!(ax3, [0.0, nobs[4]], color = :red, linestyle = :dash)  
    ax3.title ="A4"
    Label(fig[0, :], text = "Case 1", textsize = 30)
    save("./plots/case_1.png", fig )
    display(fig)
end