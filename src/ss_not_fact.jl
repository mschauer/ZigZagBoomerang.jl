using LinearAlgebra

# For now just centered at 0
freezing_time(x, θ, μ, F) = freezing_time(x, θ, F)
function freezing_time(x, θ, μ, F::Union{Boomerang, Boomerang1d})
    if μ == 0
        if θ*x >= 0.0
            return π - atan(x/θ) 
        else
            return atan(-x/θ)
        end
    else
        u = x^2 - 2μ*x + θ^2
        u < 0 && return Inf
        t1 = mod(2atan((sqrt(u) - θ)/(2μ - x)), 2pi)
        t2 = mod(-2atan((sqrt(u) + θ)/(2μ - x)), 2pi)
        x == 0 && return max(t1, t2)
        return min(t1, t2)
    end
end

function freezing_time!(tfrez, t, x, θ, f, Z::Union{BouncyParticle, Boomerang})
    for i in eachindex(x)
        if f[i]
            tfrez[i] = t + freezing_time(x[i], θ[i], Z.μ[i], Z)
        end
    end
    tfrez
end

function refresh_sticky_vel!(θ, θf, f, F::Union{BouncyParticle, Boomerang})
    ρ̄ = sqrt(1-F.ρ^2)
    for i in eachindex(θ)
        if f[i]
            θ[i] = F.ρ*θ[i] +  ρ̄*randn()
        else
            θf[i] = abs( F.ρ*θf[i] +  ρ̄*randn())*sign(θf[i])
        end
    end
    θ, θf
end

function subnormsq(∇ϕx, θ)
    res = 0.0
    for i in eachindex(θ)
        if θ[i] == 0
            continue
        else
            res += ∇ϕx[i]^2
        end
    end
    res
end

function sdot(x,y,θ)
    res = 0.0
    @inbounds for i in eachindex(θ)
        if θ[i] == 0.0
            continue
        else
            res += x[i]*y[i]
        end
    end
    res
end


function reflect_sticky!(∇ϕx, x, θ, f, Flow::Union{BouncyParticle, Boomerang})
    c = (2*sdot(∇ϕx, θ, θ)/subnormsq(∇ϕx, θ))
    for i in eachindex(θ)
        if f[i]
            θ[i] -= c*∇ϕx[i]
        end
    end
    θ
end

function smove_forward!(τ, t, x, θ, f, Z::Union{BouncyParticle, ZigZag})
    t += τ
    for i in eachindex(x)
        if f[i] 
            x[i] += θ[i]*τ
        end
    end
    t, x, θ
end

function smove_forward!(τ, t, x, θ, f, B::Union{Boomerang, FactBoomerang})
    s, c = sincos(τ)
    for i in eachindex(x)
        if f[i]
            x[i], θ[i] = (x[i] - B.μ[i])*c + θ[i]*s + B.μ[i],
                        -(x[i] - B.μ[i])*s + θ[i]*c
        end
    end
    t + τ, x, θ
end


function sevent(t, x, θ, f, Z::Union{BouncyParticle,Boomerang})
    t, copy(x), copy(θ), copy(f)
end

function sticky_pdmp_inner!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
        Flow::Union{BouncyParticle, Boomerang}, κ, args...; strong_upperbounds = false, factor=1.5, adapt=false)
    
    # f[i] is true if i is free
    # frez[i] is the time to freeze, if free, the time to unfreeze, if frozen
    while true
        tᶠ, i = findmin(tfrez) # could be implemented with a queue
        tt, j = findmin([tref, tᶠ, t′])
        τ = tt - t
        t, x, θ = smove_forward!(τ, t, x, θ, f, Flow)
        # move forward
        if j == 1 # refreshments of velocities
            θ, θf = refresh_sticky_vel!(θ, θf, f, Flow)
            tref = t + waiting_time_ref(Flow) # regenerate refreshment time
            b = ab(x, θ, c, Flow) # regenerate reflection time
            told = t
            t′, _ = next_time(t, b, rand())
            tfrez = freezing_time!(tfrez, t, x, θ, f, Flow)
            for i in eachindex(f) # make function later...
                if !f[i]
                    tfrez[i] = t - log(rand())/(κ[i]*abs(θf[i]))
                end
            end
        elseif j == 2 # get frozen or unfrozen in i
            if f[i] # if free
                if abs(x[i]) > 1e-8
                    tfrez[i] = t + freezing_time(x[i], θ[i], Flow.μ[i], Flow) # wrong zero of curve 
                    error("x[i] = $(x[i]) !≈ 0 at $(tfrez[i])")
                end
                x[i] = -0*θ[i]
                θf[i], θ[i] = θ[i], 0.0 # stop and save speed
                f[i] = false # change tag
                tfrez[i] = t - log(rand())/(κ[i]*abs(θf[i])) # sticky time
                # tfrez[i] = t - log(rand()) # option 2
                if !(strong_upperbounds) #not strong upperbounds, draw new waiting time
                    b = ab(x, θ, c, Flow) # regenerate reflection time
                    told = t
                    t′, _ = next_time(t, b, rand())
                end
            else # is frozen ->  unfreeze
                @assert x[i] == 0 && θ[i] == 0
                θ[i], θf[i] = θf[i], 0.0 # restore speed
                f[i] = true # change tag
                tfrez[i] = t + freezing_time(x[i], θ[i], Flow.μ[i], Flow)
                b = ab(x, θ, c, Flow) # regenerate reflection time
                told = t
                t′, _ = next_time(t, b, rand())
            end
        else #   t′ usual bouncy particle / boomerang step
            ∇ϕx = ∇ϕ!(∇ϕx, x, args...)
            ∇ϕx = grad_correct!(∇ϕx, x, Flow)
            l, lb = λ(∇ϕx, θ, Flow), sλ̄(b, t - told) # CHECK if depends on f
            num += 1
            if rand()*lb <= l # reflect!
                acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ = reflect_sticky!(∇ϕx, x, θ, f, Flow)
                b = ab(x, θ, c, Flow) # regenerate reflection time
                told = t
                t′, _ = next_time(t, b, rand())
                tfrez = freezing_time!(tfrez, t, x, θ, f, Flow)
            else # nothing happened
                b = ab(x, θ, c, Flow)
                told = t
                t′, _ = next_time(t, b, rand())
                continue
            end
        end
        push!(Ξ, sevent(t, x, θ, f, Flow))
        return t, x, θ, t′, tref, tfrez, told, f, θf, (acc, num), c, b
    end

end


function sspdmp(∇ϕ!, t0, x0, θ0, T, c, Flow::Union{BouncyParticle, Boomerang},
        κ, args...;  strong_upperbounds = false, adapt=false, factor=2.0)
    t, x, θ, ∇ϕx = t0, deepcopy(x0), deepcopy(θ0), deepcopy(θ0)
    told = t0
    θf = 0*θ # tags
    f = [true for _ in eachindex(x)]
    Ξ = Trace(t0, x0, θ0, f, Flow)
    push!(Ξ, sevent(t, x0, θ0, f, Flow))
    tref = waiting_time_ref(Flow) #refreshment times
    tfrez = zero(x)
    tfrez = freezing_time!(tfrez, t0, x0, θ0, f, Flow) #freexing times
    num = acc = 0
    b = ab(x, θ, c, Flow)
    t′, _ = next_time(t, b, rand()) # reflection time
    while t < T
        t, x, θ, t′, tref, tfrez, told, f, θf, (acc, num), c, b = sticky_pdmp_inner!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
                Flow, κ, args...; strong_upperbounds = strong_upperbounds, factor = factor, adapt = adapt)
    end
    return Ξ, (t, x, θ), (acc, num), c
end
