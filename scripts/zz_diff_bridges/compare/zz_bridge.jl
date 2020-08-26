#################################################################################
# Comparison of Zig-Zag for diffusion bridges with tailored Poisson rates       #
# and with adapted Poisson rate. Reference: https://arxiv.org/abs/2001.05889.   #
#################################################################################

using ZigZagBoomerang, SparseArrays, LinearAlgebra
#using CairoMakie
include("../../faberschauder.jl")
const ZZB = ZigZagBoomerang
using CSV
using DataFrames
using Statistics
# Drift
b(x) = α * sin(x)
# First derivative
b′(x) = α * cos(x)
# Second derivative
b″(x) = -α * sin(x)


####################################################################
# Overloading Poisson times in order to have tighter upperbounds   #
####################################################################
struct MyBound
    c::Float64
end
function ZZB.adapt!(b::Vector{MyBound}, i, x)
    b[i] = MyBound(b[i].c * x)
end

"""
    poisson_time(a, b, c, u)
Obtaining waiting time for inhomogeneous Poisson Process
with rate of the form λ(t) = a + (b + c*t)^+, where `c`,`a`> 0 ,`b` ∈ R, `u` uniform random variable
"""
function ZZB.poisson_time((a, b, c)::NTuple{3}, u = rand()) # formula (22)
    if b > 0
        return (-(b + a) + sqrt((b + a)^2 - 2.0 * c * log(u))) / c # positive solution of quadratic equation c*0.5 x^2 + (b + a) x + log(u) = 0
    elseif a * b / c <= log(u)
        return -log(u) / a
    else
        return (-(a + b) + sqrt((a + b)^2 - 2.0 * c * (b * b * 0.5 / c + log(u)))) / c    # positive solution of quadratic equation c*0.5 x^2 + (b + a) x + log(u) + b*b*0.5/c = 0
    end
end

ZZB.sλ̄((a, b, c)::NTuple{3}, Δt) = a + ZZB.pos(b + c * Δt)
"""
    abc(G, i, x, θ, c, Flow)
Returns the constant term `a` and linear term `b` when computing the Poisson times
from the upper upper bounding rates λᵢ(t) = max(a + b*t)^2. The factors `a` and `b`
can be function of the current position `x`, velocity `θ`, tuning parameter `c` and
the Graph `G`
"""
function ZZB.ab(G, i, x, θ, c::Vector{MyBound}, F::ZigZag)
    if i == 1
        a = c[i].c + T^(1.5)*0.5*(α^2 + α) * abs(θ[i])  # initial point
        b1 = θ[i]*(x[i] - x[end])
        b2 = θ[i]*(θ[i] - θ[end])
    elseif i == (2 << L) + 1
        a = c[i].c + T^(1.5)*0.5*(α^2 + α) * abs(θ[i])  # final point
        b1 = θ[i]*(x[i] - x[1])
        b2 = θ[i]*(θ[i] - θ[1])
    else
        l = lvl(i, L)
        a = c[i].c + T^(1.5) / 2^((L - l) * 1.5 + 2) * (α^2 + α) * abs(θ[i]) # formula (22)
        b1 = x[i] * θ[i]
        b2 = θ[i] * θ[i]
    end
    return a, b1, b2
end

# c = [MyBound(0.0) for i in 1:n]
# trace, (t, ξ, θ), (acc, num), c = @time spdmp(∇ϕmoving, 0.0, ξ0, θ0, T′, c, ZigZag(Γ, ξ0 * 0),
#                         SelfMoving(), L, T, adapt = false);


function run_zz(df, T′)
    # Zig-Zag impmentation
    n = (2 << L) + 1
    ξ0 = 0randn(n)
    u, v = 0.0, 0.0  # initial and fianl point
    ξ0[1] = u / sqrt(T)
    ξ0[end] = v / sqrt(T)
    θ0 = rand((-1.0, 1.0), n)
    θ0[end] = θ0[1] = 0.0 # fix final point
    Γ = sparse(1.0I, n, n)
    c = [MyBound(0.0) for i in 1:n]
    zz_time = @elapsed((trace, (t, ξ, θ), (acc, num), c) = spdmp(∇ϕmoving, 0.0, ξ0, θ0, T′, c, ZigZag(Γ, ξ0 * 0),
    SelfMoving(), L, T, adapt = false))
    t_trace = getindex.(trace.events, 1)
    t_trace = [0.0, t_trace...]
    x_trace, v_trace = resize(trace)
    ess = ess_pdmp_components(t_trace, x_trace, v_trace, n_batches = 50)
    ess_xt2 = ess[Int((length(ξ0)+1)/2)]
    push!(df, Dict(:sampler => "ZZ", :alpha => α, :ess_XT2 => ess_xt2/zz_time, :ess_mean => sum(ess[2:end-1])/(length(ess)-2)/zz_time,
            :ess_median => median(ess[2:end-1])/zz_time, :ess_min => minimum(ess[2:end-1])/zz_time, :runtime => zz_time ), )
    return df
end
function resize(zz_trace)
    x0 = zz_trace.x0
    v0 = zz_trace.θ0
    x_trace = [deepcopy(x0)]
    v_trace = [deepcopy(v0)]
    for i in eachindex(zz_trace.events)
        if i == 1
            x0 = x0 + v0*zz_trace.events[i][1]
        else
            x0 = x0 + v0*(zz_trace.events[i][1] - zz_trace.events[i-1][1])
        end
        v0[zz_trace.events[i][2]] = zz_trace.events[i][4]
        x0[zz_trace.events[i][2]] = zz_trace.events[i][3]
        # if !(v0[zz_trace.events[i][2]] == zz_trace.events[i][4])
        #     println("vel: $(v0[zz_trace.events[i][2]]) vs $(zz_trace.events[i][4])")
        #     error("")
        # end
        # if !(|x0[zz_trace.events[i][2]]| zz_trace.events[i][4])
        #     println("pos: $(x0[zz_trace.events[i][2]]) vs $(zz_trace.events[i][3])")
        #     error("")
        # end
        push!(x_trace, deepcopy(x0))
        push!(v_trace, deepcopy(v0))
    end
    x_trace, v_trace
end



using DataFrames
df = DataFrame(sampler = String[], alpha = Float64[], ess_XT2 = Float64[],  ess_mean = Float64[],
    ess_median = Float64[], ess_min = Float64[], runtime = Float64[])
function data_collection(df)
    for α′ in [0.1, 0.3, 0.7]
        global α = α′
        global L = 5
        global T = 100.0
        T′ = 10000
        run_zz(df, T′)
    end
    return df
end
using CSV
data_collection(df)
CSV.write("./scripts/zz_diff_bridges/compare/benchamrk_zz.csv", df)



#
#
#
#
# using Makie
# using CairoMakie
# p1 = lines(S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
# for ξ in ξs[1:5:end]
#     lines!(p1, S, [dotψ(ξ, s, L, T) for s in S], linewidth=0.3)
# end
# display(p1)
