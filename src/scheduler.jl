using ZigZagBoomerang: SPriorityQueue, enqueue!
using FunctionWranglers
using StructArrays
using StructArrays: components


using Random, LinearAlgebra

@inline @generated function _sfindmin(wrangler::FunctionWrangler{TOp, TNext}, tmin, j, n, args...) where {TNext, TOp}
    TOp === Nothing && return :(tmin, j)
    argnames = [:(args[$i]) for i = 1:length(args)]
    return quote
        t = wrangler.op($(argnames...))
        tmin, j = t ≤ tmin ? (t, n) : (tmin, j)
        return _sfindmin(wrangler.next, tmin, j, n + 1, $(argnames...))            
    end
end

"""
    sfindmin(wrangler::FunctionWrangler, args...)
    
Look for the function which returns smallest value for the given arguments, and returns its index.
"""
sfindmin(wrangler::FunctionWrangler, args...) = _sfindmin(wrangler, Inf, 0, 1, args...)



T = 100.0
d = 10000

function f1!(i, t′, u)
    t, x, θ, m = components(u)
    x[i] += θ[i]*(t′ - t[i]) 
    if rand() < 0.1
        θ[i] = -θ[i]
    end
    t[i] = t′
end

function f2!(i, t′, u)
    t, x, θ, m = components(u)
    x[i] += θ[i]*(t′ - t[i]) 
    @assert norm(x[i]) < 1e-7
    θ[i] = -θ[i]
    t[i] = t′
end

function next1(j, e, i, t′, u)
    t, x, θ, m = components(u)
    t[j] + randexp()
end 
function next2(j, e, i, t′, u) 
    t, x, θ, m = components(u)
    θ[j]*x[j] >= 0 ? Inf : t[j] - x[j]/θ[j]
end

struct Handler{T1,T2,T3,T4,T5,T6}
    G::T1
    f::T2
    next::T3
    state::T4
    T::T5
    args::T6
end
Base.eltype(::Type{Handler{T1,T2,T3,T4,T5,T6}}) where {T1,T2,T3,T4,T5,T6} = Tuple{Float64, Int, eltype(T4), Int}
function Base.iterate(handler::Handler)
    d = length(handler.state)
    action = zeros(Int, d) # null events to trigger next event computations
    Q = SPriorityQueue(zeros(d))
    u = deepcopy(handler.state)
    iterate(handler, (u, action, Q))
end
Base.IteratorSize(::Type{<:Handler}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:Handler}) = Base.HasEltype()

function Base.iterate(handler, (u, action, Q))
    f! = handler.f
    next = handler.next
    G = handler.G
    ev = handle!(u, G, f!, next, action, Q, handler.args...)
    ev[1] > handler.T && return nothing
    ev, (u, action, Q)
end

function handle(handler, args...)
     u = deepcopy(handler.state)
     G = handler.G
     f! = handler.f
     next = handler.next
     d = length(handler.state)
     action = zeros(Int, d)
     Q = SPriorityQueue(zeros(d))
     while true
        ev = handle!(u, G, f!, next, action, Q, args...)
        ev[1] > handler.T && return ev
     end
end

function handle!(u, G, f!, next, action, Q, args...)
    # Who is (i) next, when (t′) and what (j) happens?
    i, t′ = peek(Q)
    e = action[i] 
    # Trigger state change
    if e > 0 # something? 
        sindex(f!, e, i, t′, u, args...)
    end
    # What happens next now has changed for each j, trigger updates
    for j in affected(G, e, i, t′, u, args...)
        τ, e = sfindmin(next, j, e, i, t′, u, args...)
        Q[j] = τ
        action[j] = e
    end

    (t′, i, u[i], e)
end
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
Random.seed!(1)

fs = FunctionWrangler((f1!, f2!))
next = FunctionWrangler((next1, next2))
G = nothing
affected(::Nothing, e, i, t′, u) = i 
u0 = StructArray(t=zeros(d), x=zeros(d), θ=ones(d), m=zeros(Int,d))
h = Handler(G, fs, next, u0, T, ())
l_ = @time lastiterate(h)
@assert l_[3].x == 44.18692841846383
l = @time handle(h)
trc = @time collect(h)
#@code_warntype handler(zeros(d), T, (f1!, f2!));

#using ProfileView

#ProfileView.@profview handler(zeros(d), 10T);

subtrace = [t for t in trc if t[2] == 1]
lines(getindex.(subtrace, 1), getfield.(getindex.(subtrace, 3), :x))