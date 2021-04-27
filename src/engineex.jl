using ZigZagBoomerang
using Random
include("engine.jl")
T = 100.0
d = 10000
function reset!(i, t′, u, args...)
    false, i
end

function next_reset(j, i, t′, u, args...)
    Inf
end

function action1!(i, t′, u, _)
    t, x, θ, m = components(u)
    x[i] += θ[i]*(t′ - t[i]) 
    if rand() < 0.1
        θ[i] = -θ[i]
    end
    t[i] = t′
    true, i
    
end

function action2!(i, t′, u, _)
    t, x, θ, m = components(u)
    x[i] += θ[i]*(t′ - t[i]) 
    @assert norm(x[i]) < 1e-7
    θ[i] = -θ[i]
    t[i] = t′
    true, i
end

function next_action1(j, i, t′, u, _)
    t, x, θ, m = components(u)
    t[j] + randexp()
end 
function next_action2(j, i, t′, u, _) 
    t, x, θ, m = components(u)
    θ[j]*x[j] >= 0 ? Inf : t[j] - x[j]/θ[j]
end

Random.seed!(1)

action! = (reset!, action1!, action2!)
#action! = FunctionWrangler(action!)
next_action = FunctionWrangler((next_reset, next_action1, next_action2))

u0 = StructArray(t=zeros(d), x=zeros(d), θ=ones(d), m=zeros(Int,d))
h = Handler(action!, next_action, u0, T, ())
l_ = lastiterate(h) 
@assert l_[3].x == 44.18692841846383
l_ = @time lastiterate(h) 
handle(h)
l = @time handle(h)
trc = @time collect(h)
#@code_warntype handler(zeros(d), T, (f1!, f2!));

#using ProfileView

#ProfileView.@profview handler(zeros(d), 10T);

subtrace = [t for t in trc if t[2] == 1]
lines(getindex.(subtrace, 1), getfield.(getindex.(subtrace, 3), :x))