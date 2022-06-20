import Base.iterate
export NotFactSampler, trace


# pdmp_inner!(rng, ∇ϕ!, ∇ϕx, t, x, θ, c::Bound, abc, (t′, renew), τref, v, (acc, num),
# Flow::Union{BouncyParticle, Boomerang}, args...; subsample=false, factor=1.5, adapt=false)

struct NotFactSampler{TF,T∇ϕ!,Tc,Tu0,Trng,Targs,Tkargs} <: PDMPSampler
    F::TF
    ∇ϕ!::T∇ϕ!
    c::Tc
    u0::Tu0 # t0 => (x0, θ0)
    rng::Trng

    args::Targs

    kargs::Tkargs
end

function NotFactSampler(∇ϕ!, u0, c, F::Union{BouncyParticle,Boomerang}, args...;
    factor=1.8, subsample=subsample, adapt=false, seed=Seed())
    kargs = (factor=factor, adapt=adapt, subsample=subsample)
    return NotFactSampler(F, ∇ϕ!, c, u0, Rng(seed), args, kargs)
end

function iterate(FS::NotFactSampler)
    t0, (x0, θ0) = FS.u0
    Flow = FS.F
    n = length(x0)
    t, x, θ, ∇ϕx = t0, copy(x0), copy(θ0), copy(θ0)
    c = FS.c
    rng = FS.rng
    τref = waiting_time_ref(rng, Flow)
    ∇ϕx, v = FS.∇ϕ!(∇ϕx, t, x, θ, FS.args...)
    ∇ϕx = grad_correct!(∇ϕx, x, Flow)
    num = acc = 0
    abc = ab(x, θ, c, ∇ϕx, v, Flow)

    t′, renew = next_time(t, abc, rand(rng))

    iterate(FS, ((t => (x, θ)), ∇ϕx, (acc, num), c, abc, (t′, renew), τref, v))
end


function iterate(FS::NotFactSampler,  (u, ∇ϕx, (acc, num), c, abc, (t′, renew), τref, v))
    t, (x, θ) = u
    t, x, θ, (acc, num), c, abc, (t′, renew), τref, v = pdmp_inner!(FS.rng, FS.∇ϕ!, ∇ϕx, t, x, θ, c, abc, (t′, renew), τref, v, (acc, num), FS.F, FS.args...; FS.kargs...)
    ev = event(t, x, θ, FS.F)
    u = t => (x, θ)
    return ev, (u, ∇ϕx, (acc, num), c, abc, (t′, renew), τref, v)
end

function rawevent(t, x, θ, Z::Union{BouncyParticle,Boomerang})
    t, x, θ, nothing
end

######

function iterate(FS::NotFactSampler{<:Any, <:Tuple})
    t0, (x0, θ0) = FS.u0
    flow = FS.F
    n = length(x0)
    t, x, θ, ∇ϕx = t0, copy(x0), copy(θ0), copy(θ0)
    c = FS.c
    rng = FS.rng
    τref = NaN


    dϕ, ∇ϕ! = FS.∇ϕ![1], FS.∇ϕ![2]  
    θdϕ, v = dϕ(t, x, θ, FS.args...) 
    num = acc = 0
    abc = ab(t, x, θ, c, θdϕ, v, flow)
    t′, action = next_event1(rng, (t, x, θ), abc, flow)
    iterate(FS, ((t => (x, θ)), ∇ϕx, (acc, num), c, abc, (t′, action), τref))
end
using Test


function iterate(FS::NotFactSampler{<:Any, <:Tuple},  (u, ∇ϕx, (acc, num), c, abc, (t′, action), τref))
    t, (x, θ) = u
    dϕ, ∇ϕ! = FS.∇ϕ![1], FS.∇ϕ![2]  
    t, (acc, num), c, abc, (t′, action), τref = pdmp_inner!(FS.rng, dϕ, ∇ϕ!, ∇ϕx, t, x, θ, c, abc, (t′, action), τref, (acc, num), FS.F, FS.args...; FS.kargs...)
   
    ev = rawevent(t, x, θ, FS.F)
    u = t => (x, θ)
    return ev, (u, ∇ϕx, (acc, num), c, abc, (t′, action), τref)
end
