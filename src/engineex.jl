using ZigZagBoomerang
using StructArrays
using StructArrays: components
using Random
using LinearAlgebra

function lastiterate(itr) 
    ϕ  = iterate(itr)
    if ϕ === nothing
        error("empty")
    end
    x, state = ϕ
    while true
        ϕ = iterate(itr, state)
        if ϕ === nothing 
            return x
        end
        x, state = ϕ
    end
end
using Random
#include("engine.jl")
T = 100.0
d = 10000
function reset!(i, t′, u, args...)
    false, i
end

function next_reset(j, i, t′, u, args...)
    0, Inf
end

function action1!(i, t′, u, args...)
    t, x, θ, m = components(u)
    x[i] += θ[i]*(t′ - t[i]) 
  #  if rand() < 0.1
        θ[i] = -abs(θ[i])
   # end
    t[i] = t′
    x[i] = 1.0
    true, i
    
end

function action2!(i, t′, u, args...)
    t, x, θ, m = components(u)
    x[i] += θ[i]*(t′ - t[i]) 
    @assert norm(x[i]) < 1e-7
    θ[i] = abs(θ[i])
    x[i] = 0.0
    t[i] = t′
    true, i
end

function next_action1(j, i, t′, u, args...)
    t, x, θ, m = components(u)
  #  t[j] + randexp()
    0, θ[j]*(x[j]-1) >= 0 ? Inf : t[j] - (x[j]-1)/θ[j]
end 
function next_action2(j, i, t′, u, args...) 
    t, x, θ, m = components(u)
    0, θ[j]*x[j] >= 0 ? Inf : t[j] - x[j]/θ[j]
end



action! = (reset!, action1!, action2!)
#action! = FunctionWrangler(action!)
next_action = FunctionWrangler((next_reset, next_action1, next_action2))

u0 = StructArray(t=zeros(d), x=zeros(d), θ=ones(d), m=zeros(Int,d))
u0.θ[1] = 1/Base.MathConstants.golden
Random.seed!(1)
h = Schedule(action!, next_action, u0, T, ())
l_ = lastiterate(h) 
l_ = @time lastiterate(h) 

Random.seed!(1)
l = simulate(h)
@assert l[3].x == l_[3].x
l = @time simulate(h)
trc_ = @time collect(h)
#@code_warntype handler(zeros(d), T, (f1!, f2!));

#using ProfileView

#ProfileView.@profview handler(zeros(d), 10T);

subtrace = [t for t in trc_ if t[2] == 1]
lines(getindex.(subtrace, 1), getfield.(getindex.(subtrace, 3), :x))

trc = Zig.FactTrace(ZigZag(sparse(1.0I(d)), zeros(d)), t0, x, θ, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_])
ts, xs = Zig.sep(discretize(Zig.subtrace(trc, [1,4]), 0.01))

cummean(x) = cumsum(x) ./ eachindex(x)
fig = lines(ts, getindex.(xs, 2))
lines!(ts, cummean(getindex.(xs, 2)), color=:green, linewidth=2.0)
fig = lines(getindex.(xs, 1), getindex.(xs, 2), color=:red)