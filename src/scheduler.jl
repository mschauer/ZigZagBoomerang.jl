using ZigZagBoomerang: SPriorityQueue, enqueue!
using FunctionWranglers
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

function f1!(i, t′, u, event_type, Q)
    t, x, θ, m = u
    x[i] += θ[i]*(t′ - t[i]) 
    if rand() < 0.1
        θ[i] = -θ[i]
    end
    t[i] = t′
end

function f2!(i, t′, u, event_type, Q)
    t, x, θ, m = u
    x[i] += θ[i]*(t′ - t[i]) 
    @assert norm(x[i]) < 1e-7
    θ[i] = -θ[i]
    t[i] = t′
end

function next1(i, t′, u)
    t, x, θ, m = u
    t[i] + randexp()
end 
function next2(i, t′, u) 
    t, x, θ, m = u
    θ[i]*x[i] >= 0 ? Inf : t[i] - x[i]/θ[i]
end

struct Handler{T1,T2,T3,T4,T5}
    G::T1
    f::T2
    next::T3
    state::T4
    T::T5
end
function Base.iterate(handler::Handler)
    d = length(handler.state[2])
    event_type = zeros(Int, d)
    Q = queue(zeros(d))
    u = handler.state
    iterate(handler, (u, event_type, Q))
end
Base.IteratorSize(::Handler) = Base.SizeUnknown()

function Base.iterate(handler, (u, event_type, Q))
    f! = handler.f
    next = handler.next
    G = handler.G
    ev = handle!(u, G, f!, next, event_type, Q)
    ev[1] > handler.T && return nothing
    ev, (u, event_type, Q)
end

function handle(handler)
     u = handler.state
     G = handler.G
     f! = handler.f
     next = handler.next
     d = length(handler.state[2])
     event_type = zeros(Int, d)
     Q = queue(zeros(d))
     while true
        ev = handle!(u, G, f!, next, event_type, Q)
        ev[1] > handler.T && return ev
     end
end

function handle!(u, G, f!, next, event_type, Q)
    # Who is (i) next, when (t′) and what (j) happens?
    i, t′ = peek(Q)
    e = event_type[i] 
    # Trigger state change
    if e > 0 # something? 
        sindex(f!, e, i, t′, u, event_type, Q)
    end
    # What happens next now has changed for each j
    for j in neighbours(G, i)
        τ, e = sfindmin(next, j, t′, u)
        Q[j] = τ
        event_type[j] = e
    end

    (t′, i, u[1][i], e)
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
neighbours(::Nothing, i) = i 
h = Handler(G, fs, next, (zeros(d), zeros(d), ones(d), zeros(Int,d)), T)
trc = @time lastiterate(h)
h = Handler(G, fs, next, (zeros(d), zeros(d), ones(d), zeros(Int,d)), T)
l = @time handle(h)
#@code_warntype handler(zeros(d), T, (f1!, f2!));

#using ProfileView

#ProfileView.@profview handler(zeros(d), 10T);
