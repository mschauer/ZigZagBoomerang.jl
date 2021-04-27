using ZigZagBoomerang
using ZigZagBoomerang: sλ, sλ̄, reflect!, Rng, ab, smove_forward!, neighbours
using Random
include("engine.jl")
T = 500.0
d = 80
seed = (UInt(1),UInt(1))
using ConcreteStructs
@concrete struct SPDMP
    G
    G1
    G2
    ∇ϕ
    F
    rng
    adapt
    factor
end

function rand_reflect!(i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, m, c, t_old, b = components(u)
    smove_forward!(G, i, t, x, θ, t′, F)
    ∇ϕi = P.∇ϕ(x, i, args...)
    l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i], t[i] - t_old[i])
   # num += 1
    if rand(P.rng)*lb < l
  #      acc += 1
        if l >= lb
            !P.adapt && error("Tuning parameter `c` too small.")
      #      acc = num = 0
            adapt!(c, i, P.factor)
        end
        smove_forward!(G2, i, t, x, θ, t′, F)
        ZigZagBoomerang.reflect!(i, ∇ϕi, x, θ, F)
        return u, neighbours(G1, i)
    else
        return u, [i]
    end
    
end


function reflect0!(i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, m, c, t_old, b = components(u)
     
    @assert norm(x[i] + θ[i]*(t′ - t[i])) < 1e-7
    x[i] = 0.0
    θ[i] = abs(θ[i])
    t[i] = t′
    return u, neighbours(G1, i)
end

function reflect1!(i, t′, u, P::SPDMP, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, m, c, t_old, b = components(u)
     
    @assert norm(x[i] + θ[i]*(t′ - t[i]) - 1) < 1e-7
    x[i] = 1.0
    θ[i] = -abs(θ[i])
    t[i] = t′
    return u, neighbours(G1, i)
end

function next_rand_reflect(j, i, t′, u, P, args...)
    G, G1, G2 = P.G, P.G1, P.G2
    F = P.F
    t, x, θ, m, c, t_old, b = components(u)
    b[j] = ab(G1, j, x, θ, c, F)
    t_old[j] = t[j]
    t[j] + poisson_time(b[j], rand(P.rng))
end

function next_reflect0(j, i, t′, u, args...) 
    t, x, θ, m = components(u)
    θ[j]*x[j] >= 0 ? Inf : t[j] - x[j]/θ[j]
end

function next_reflect1(j, i, t′, u, args...) 
    t, x, θ, m = components(u)
    θ[j]*(x[j]-1) >= 0 ? Inf : t[j] - (x[j]-1)/θ[j]
end

Random.seed!(1)

using SparseArrays

S = 1.3I + 0.5sprandn(d, d, 0.1)
Γ = S*S'

∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation

# t, x, θ, m, c, t_old, b 
t0 = 0.0
t = fill(t0, d)
x = rand(d) 
θ = θ0 = rand([-1.0, 1.0], d)
F = ZigZag(0.9Γ, x*0)


c = .6*[norm(Γ[:, i], 2) for i in 1:d]
G = G1 = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]

b = [ab(G1, i, x, θ, c, F) for i in eachindex(θ)]
  
u0 = StructArray(t=t, x=x, θ=θ, m=zeros(Int,d), c=c, t_old=copy(t), b=b)


rng = Rng(seed)
t_old = copy(t)
adapt = false
factor = 1.7
P = SPDMP(G, G1, G2, ∇ϕ, F, rng, adapt, factor)

#action! = FunctionWrangler((reset!, rand_reflect!, reflect0!, reflect1!))
action! = (reset!, rand_reflect!, reflect0!, reflect1!)

next_action = FunctionWrangler((next_reset, next_rand_reflect, next_reflect0, next_reflect1))
#action! = FunctionWrangler((reset!, rand_reflect!))
#next_action = FunctionWrangler((next_reset, next_rand_reflect))


h = Handler(action!, next_action, u0, T, (P, Γ))

l_ = lastiterate(h) 

l_ = @time lastiterate(h) 

l = handle(h)
l = @time handle(h)
trc_ = @time collect(h);
trc = ZigZagBoomerang.FactTrace(F, t0, x, θ, [(ev[1], ev[2], ev[3].x, ev[3].θ) for ev in trc_])


trace, _, acc = @time spdmp(∇ϕ, t0, x, θ, T, c, G, F, Γ);
#@code_warntype handler(zeros(d), T, (f1!, f2!));

#using ProfileView

#ProfileView.@profview handler(zeros(d), 10T);

subtrace1 = [t for t in trc_ if t[2] == 1]
lines(getindex.(subtrace1, 1), getfield.(getindex.(subtrace1, 3), :x))

ts, xs = ZigZagBoomerang.sep(ZigZagBoomerang.subtrace(trc, [1,4]))

lines(ts, getindex.(xs, 2))
fig = lines(getindex.(xs, 1), getindex.(xs, 2))