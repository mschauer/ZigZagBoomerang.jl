# using Revise
# using CairoMakie, AbstractPlotting, ZigZagBoomerang, SparseArrays, LinearAlgebra

# Essence of the model section procedure.
# a = Dict((1,0)=>1, (0,0)=>2)
# push!(a, (2,0) => 3)
# (1,0) ∈ keys(a)
# key1 = (1,0)
# a[key1] = a[key1] + 1

# function ϕ(x, i, μ)
#     x[i] - μ[i]
# end
# κ = 1.0
# n = 100
# μ = zeros(n)
# x0 = randn(n)
# θ0 = rand((-1.0,1.0), n)
# T = 1000.0
# c = 10ones(n)
#@time trace0, _ = ZigZagBoomerang.spdmp(ϕ, 0.0, x0, θ0, T, c, ZigZag(sparse(1.0I,n,n), zeros(n)), μ)
# @time trace0, _ = ZigZagBoomerang.sspdmp(ϕ, 0.0, x0, θ0, T, c, κ, ZigZag(sparse(1.0I,n,n), zeros(n)), μ)

is_stuck(xi,vi) = xi == 0 && vi == 0 ? 1 : 0

"""
    model_trace(x0, v0, events)
given an initial position x0, v0 and a trace of events, assign to each
event, a vector k corresponding to the open sets as in Appendix 1.
`m0` is the initial model, `mm` is the trace of models at each event
"""
function model_trace(x0, v0, events)
    m0 = is_stuck.(x0, v0)
    m = copy(m0)
    mm = Vector{Int64}[]
    k = 1
    for event in events
        t, i, x, v = event
        m[i] = is_stuck(x, v)
        push!(mm, copy(m))
        k += 1
    end
    m0, mm
end

"""
    model_probs(m0, mm, events)
Given the initial model and the trace of models and trace of events,
computes the time spent by the sampler in each model explored. returns
a Ditcionary `probs` containing as `key` the model and as value its total time
"""
function model_probs(m0, mm, events)
    probs = Dict(m0=> events[1][1],)
    t = 0.0
    for i in eachindex(events)[2:end]
        tnew, i, x, v = events[i]
        δt = tnew - t
        t = tnew
        key = mm[i]
        if key ∈ keys(probs)
            probs[key] += δt
        else
            push!(probs, key => δt)
        end
    end
    probs
end




# m0, mm = model_trace(x0, θ0, trace0.events)
# model_probs(m0, mm, trace0.events)
