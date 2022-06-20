# Implementation of d-dimensional Boomerang and Bouncy particle samplers (the two
# most known not-factorised PDMC)
using LinearAlgebra

grad_correct!(y, x, F::Union{BouncyParticle, ZigZag}) = y
function grad_correct!(y, x, F::FactBoomerang)
    @. y -= x - F.μ
    y
end
function grad_correct!(y, x, F::Boomerang)
    y .-= (F.L'\(F.L\(x - F.μ)))
    y
end
λ(∇ϕx::AbstractVector, θ, F::Union{BouncyParticle, Boomerang}) = pos(dot(∇ϕx, θ))
λ(θdϕ::Number, F::Union{BouncyParticle, Boomerang}) = pos(θdϕ)
#=
function refresh!(rng, θ, F::BouncyParticle)
    ρ̄ = sqrt(1-F.ρ^2)
    U = F.U    
    θ .= F.ρ*θ + ρ̄*(U\randn(rng, length(θ)))
    θ
end
=#

# Here use sparsity as the factorised samplers
function ab(x, θ, C::GlobalBound, ∇ϕx, v, B::BouncyParticle)
    (C.c + θ'*(B.Γ*(x-B.μ)), θ'*(B.Γ*θ), Inf)
end
function ab(x, θ, C::LocalBound, ∇ϕx::AbstractVector, v, B::BouncyParticle)
    (C.c + dot(θ, ∇ϕx), v, 2sqrt(length(θ))/C.c/norm(θ, 2))
end


function ab(x, θ, C::GlobalBound, ∇ϕx, v, B::Boomerang)
    (sqrt(normsq(θ) + normsq((x - B.μ)))*C.c, 0.0, Inf)
end
ab(x, θ, c, flow) =  ab(x, θ, GlobalBound(c), nothing, nothing, flow)

function event(t, x, θ, Z::Union{BouncyParticle,Boomerang})
    t, copy(x), copy(θ), nothing
end

function next_time(t, abc, z = rand(rng))
    Δt = poisson_time(abc[1], abc[2], z)
    if Δt > abc[3]
        return t + abc[3], true
    else
        return t + Δt, false
    end
end

function pdmp_inner!(rng, ∇ϕ!, ∇ϕx, t, x, θ, c::Bound, abc, (t′, renew), τref, v, (acc, num),
     Flow::Union{BouncyParticle, Boomerang}, args...; subsample=false, factor=1.5, adapt=false)
    while true
        if τref < t′
            t, x, θ = move_forward!(τref - t, t, x, θ, Flow)
            θ = refresh!(rng, θ, Flow)
            ∇ϕx, v = ∇ϕ!(∇ϕx, t, x, θ, args...)
            ∇ϕx = grad_correct!(∇ϕx, x, Flow)
            l = λ(∇ϕx, θ, Flow) 
            τref = t + waiting_time_ref(rng, Flow)
            abc = ab(x, θ, c, ∇ϕx, v, Flow)
            t′, renew = next_time(t, abc, rand(rng))
            return t, x, θ, (acc, num), c, abc, (t′, renew), τref, v
        elseif renew
            τ = t′ - t
            t, x, θ = move_forward!(τ, t, x, θ, Flow) 
            ∇ϕx, v = ∇ϕ!(∇ϕx, t, x, θ, args...)
            ∇ϕx = grad_correct!(∇ϕx, x, Flow)
            abc = ab(x, θ, c, ∇ϕx, v, Flow)
            t′, renew = next_time(t, abc, rand(rng))
        else
            τ = t′ - t
            t, x, θ = move_forward!(τ, t, x, θ, Flow)
            ∇ϕx, v = ∇ϕ!(∇ϕx, t, x, θ, args...)
            ∇ϕx = grad_correct!(∇ϕx, x, Flow)
            l, lb = λ(∇ϕx, θ, Flow), pos(abc[1] + abc[2]*τ)
            num += 1
            if rand(rng)*lb <= l
                acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ = reflect!(∇ϕx, x, θ, Flow)
                ∇ϕx, v = ∇ϕ!(∇ϕx, t, x, θ, args...)
                ∇ϕx = grad_correct!(∇ϕx, x, Flow)
                abc = ab(x, θ, c, ∇ϕx, v, Flow)
                t′, renew = next_time(t, abc, rand(rng))
                !subsample && return t, x, θ, (acc, num), c, abc, (t′, renew), τref, v
            else
                abc = ab(x, θ, c, ∇ϕx, v, Flow)
                t′, renew = next_time(t, abc, rand(rng))
            end
        end
    end
end

addnothing(u::Tuple) = u
addnothing(u) = u, nothing

"""
    pdmp(∇ϕ!, t0, x0, θ0, T, c::Bound, Flow::Union{BouncyParticle, Boomerang}; adapt=false, factor=2.0)

Run a Bouncy particle sampler (`BouncyParticle`) or `Boomerang` sampler from time,
location and velocity `t0, x0, θ0` until time `T`. `∇ϕ!(y, x)` writes the gradient
of the potential (neg. log density) into y.
`c` is a tuning parameter for the upper bound of the Poisson rate.
If `adapt = false`, `c = c*factor` is tried, otherwise an error is thrown.

It returns a `PDMPTrace` (see [`Trace`](@ref)) object `Ξ`, which can be collected
into pairs `t => x` of times and locations and discretized with `discretize`.
Also returns the `num`ber of total and `acc`epted Poisson events and updated bounds
`c` (in case of `adapt==true` the bounds are multiplied by `factor` if they turn
out to be too small.)
"""
function pdmp(∇ϕ!, t0, x0, θ0, T, c::Bound, Flow::Union{BouncyParticle, Boomerang}, args...; adapt=false, subsample=false, progress=false, progress_stops = 20, islocal = false, seed=Seed(), factor=2.0)
    t, x, θ, ∇ϕx = t0, copy(x0), copy(θ0), copy(θ0)
    rng = Rng(seed)
    Ξ = Trace(t0, x0, θ0, Flow)
    τref = waiting_time_ref(rng, Flow)
    ∇ϕx, v = ∇ϕ!(∇ϕx, t, x, θ, args...)
    ∇ϕx = grad_correct!(∇ϕx, x, Flow)
    num = acc = 0
    #l = 0.0
    abc = ab(x, θ, c, ∇ϕx, v, Flow)
    if progress
        prg = Progress(progress_stops, 1)
    else
        prg = missing
    end
    stops = ismissing(prg) ? 0 : max(prg.n - 1, 0) # allow one stop for cleanup
    tstop = T/stops

    t′, renew = next_time(t, abc, rand(rng))
    while t < T
        t, x, θ, (acc, num), c, abc, (t′, renew), τref, v = pdmp_inner!(rng, ∇ϕ!, ∇ϕx, t, x, θ, c, abc, (t′, renew), τref, v, (acc, num), Flow, args...; subsample=subsample, factor=factor, adapt=adapt)
        push!(Ξ, event(t, x, θ, Flow))

        if t > tstop
            tstop += T/stops
            next!(prg) 
        end 
    end
    ismissing(prg) || ProgressMeter.finish!(prg)
    return Ξ, (t, x, θ), (acc, num), c
end


##################################    ##################################
function ab(t, x, θ, C::LocalBound, vdϕ::Number, v, B::BouncyParticle)
    @assert vdϕ isa Number
    (C.c + vdϕ, v, t + 2sqrt(length(θ))/C.c/norm(θ, 2))
end

function next_event1(rng, u::Tuple, abc, flow)
    t, x, v = u
    a, b, Δ = abc
    τ = t + poisson_time(a, b, rand(rng))
    τrefresh = t + waiting_time_ref(rng, flow)
    when, what = findmin((τ, Δ, τrefresh))
    return when, (:bounce, :expire, :refresh)[what]
end

function pdmp_inner!(rng, dϕ::F1, ∇ϕ!::F2, ∇ϕx, t, x, θ, c::Bound, abc, (t′, action), τref, (acc, num),
    flow::BouncyParticle, args...; subsample=false, oscn=false, factor=1.5, adapt=false) where {F1, F2}
    while true
        if action == :refresh
            t, _ = move_forward!(t′ - t, t, x, θ, flow)
            refresh!(rng, θ, flow)
            θdϕ, v = dϕ(t, x, θ, args...) 
            #∇ϕx = grad_correct!(∇ϕx, x, flow)
            l = λ(θdϕ, flow) 
            abc = ab(t, x, θ, c, θdϕ, v, flow)
            t′, action = next_event1(rng, (t, x, θ), abc, flow)
            return t, (acc, num), c, abc, (t′, action), τref
        elseif action == :expire
            τ = t′ - t
            t, _ = move_forward!(τ, t, x, θ, flow) 
            θdϕ, v = dϕ(t, x, θ, args...) 
            #∇ϕx = grad_correct!(∇ϕx, x, flow)
            abc = ab(t, x, θ, c, θdϕ, v, flow)
            t′, action = next_event1(rng, (t, x, θ), abc, flow)
        else # action == :reflect
            τ = t′ - t
            t, _ = move_forward!(τ, t, x, θ, flow)
            θdϕ, v = dϕ(t, x, θ, args...) 
            #∇ϕx = grad_correct!(∇ϕx, x, flow)
            l, lb = λ(θdϕ, flow), pos(abc[1] + abc[2]*τ)
            num += 1
            if rand(rng)*lb <= l
                acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                ∇ϕ!(∇ϕx, t, x, args...)
                if !(dot(θ, ∇ϕx) ≈ θdϕ)
                    #@show  dot(θ, ∇ϕx) θdϕ
                    #error("subsampling needs to be seeded by time")
                end
                if oscn
                    @assert flow.L == I
                    oscn!(rng, θ, ∇ϕx, flow.ρ; normalize=false)
                else
                    reflect!(∇ϕx, x, θ, flow)
                end
                θdϕ, v = dϕ(t, x, θ, args...) 
                #∇ϕx = grad_correct!(∇ϕx, x, flow)
                abc = ab(t, x, θ, c, θdϕ, v, flow)
                t′, action = next_event1(rng, (t, x, θ), abc, flow)
                !subsample && return t, (acc, num), c, abc, (t′, action), τref
            else
                abc = ab(t, x, θ, c, θdϕ, v, flow)
                t′, action = next_event1(rng, (t, x, θ), abc, flow)
            end
        end
    end
end
"""
    pdmp(dϕ, ∇ϕ!, t0, x0, θ0, T, c::Bound, flow::BouncyParticle, args...; oscn=false, adapt=false, subsample=false, progress=false, progress_stops = 20, islocal = false, seed=Seed(), factor=2.0)

The first directional derivative `dϕ[1]` tells me if I move up or down the potential. The second directional derivative `dϕ[2]` tells me how fast the first changes. The gradient `∇ϕ!` tells me the surface I want to reflect on.

     dϕ = function (t, x, v, args...) # two directional derivatives
         u = ForwardDiff.derivative(t -> -ℓ(x + t*v), Dual{:hSrkahPmmC}(0.0, 1.0))
         u.value, u.partials[]
     end
     ∇ϕ! = function (y, t, x, args...)
         ForwardDiff.gradient!(y, ℓ, x)
         y .= -y
         y
     end

The remaining arguments:
     
    d = 25 # number of parameters 
    t0 = 0.0
    x0 = zeros(d) # starting point sampler
    θ0 = randn(d) # starting direction sampler
    T = 200. # end time (similar to number of samples in MCMC)
    c = 50.0 # initial guess for the bound

    # define BouncyParticle sampler (has two relevant parameters) 
    Z = BouncyParticle(∅, ∅, # ignored
        10.0, # momentum refreshment rate 
        0.95, # momentum correlation / only gradually change momentum in refreshment/momentum update
        0.0, # ignored
        I # left cholesky factor of momentum precision
    ) 

    trace, final, (acc, num), cs = @time pdmp(
            dneglogp, # return first two directional derivatives of negative target log-likelihood in direction v
            ∇neglogp!, # return gradient of negative target log-likelihood
            t0, x0, θ0, T, # initial state and duration
            ZZB.LocalBound(c), # use Hessian information 
            Z; # sampler
            oscn=false, # no orthogonal subspace pCR
            adapt=true, # adapt bound c
            progress=true, # show progress bar
            subsample=true # keep only samples at refreshment times
    )

    # to obtain direction change times and points of piecewise linear trace
    t, x = ZigZagBoomerang.sep(trace)

"""
function pdmp(dϕ, ∇ϕ!, t0, x0, θ0, T, c::Bound, flow::BouncyParticle, args...; oscn=false, adapt=false, subsample=false, progress=false, progress_stops = 20, islocal = false, seed=Seed(), factor=2.0)
    t, x, θ, ∇ϕx = t0, copy(x0), copy(θ0), copy(θ0)
    rng = Rng(seed)
    Ξ = Trace(t0, x0, θ0, flow)
    θdϕ, v = dϕ(t, x, θ, args...) 
    #@assert v2 ≈ v
    #@assert θdϕ ≈ dot(∇ϕx, θ)
    
    #∇ϕx = grad_correct!(∇ϕx, x, flow)
    num = acc = 0
    #l = 0.0
    abc = ab(t, x, θ, c, θdϕ, v, flow)
    if progress
        prg = Progress(progress_stops, 1)
    else
        prg = missing
    end
    stops = ismissing(prg) ? 0 : max(prg.n - 1, 0) # allow one stop for cleanup
    tstop = T/stops
    tref = nothing
    t′, action = next_event1(rng, (t, x, θ), abc, flow)
    while t < T
        t, (acc, num), c, abc, (t′, action), τref = pdmp_inner!(rng, dϕ, ∇ϕ!, ∇ϕx, t, x, θ, c, abc, (t′, action), τref, (acc, num), flow, args...; oscn=oscn, subsample=subsample, factor=factor, adapt=adapt)
        push!(Ξ, event(t, x, θ, flow))

        if t > tstop
            tstop += T/stops
            next!(prg) 
        end 
    end
    ismissing(prg) || ProgressMeter.finish!(prg)
    return Ξ, (t, x, θ), (acc, num), c
end

wrap(f) = wrap_(f,  methods(f)...)
wrap_(f, args...) = f
@inline wrap_(f, m) = m.nargs <= 4 ? Wrapper(f) : f
struct Wrapper{F}
    f::F
end
(F::Wrapper)(y, t, x, θ, args...) = F.f(y, x, args...), nothing

pdmp(∇ϕ!, t0, x0, θ0, T, c, flow::Union{BouncyParticle, Boomerang}, args...; nargs...) = 
pdmp(Wrapper(∇ϕ!), t0, x0, θ0, T, GlobalBound(c), flow, args...; nargs...)
