using LinearAlgebra

function freezing_time!(tfrez, t, x, θ, Z::BouncyParticle)
    for i in eachindex(x)
        if θ[i] != 0
            tfrez[i] = t + freezing_time(x[i], θ[i])
        end
    end
    tfrez
end

function refresh_sticky_vel!(θ,  θf, F::BouncyParticle)
    for i in eachindex(θ)
        if θ[i] == 0.0
            θf[i] = abs(randn())*sign(θf[i])
        else
            θ[i] = randn()
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


function reflect_sticky!(∇ϕx, x, θ, Flow)
    c = (2*dot(∇ϕx, θ)/subnormsq(∇ϕx, θ))
    for i in eachindex(θ)
        if θ[i] != 0.0
            θ[i] -= c*∇ϕx[i]
        end
    end
    θ
end

function sticky_pdmp_inner!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
        Flow::BouncyParticle, κ, args...; strong_upperbounds = false, factor=1.5, adapt=false)
    while true
        tᶠ, i = findmin(tfrez) # could be implemented with a queue
        tt, j = findmin([tref, tᶠ, t′])
        τ = tt - t
        t, x, θ = move_forward!(τ, t, x, θ, Flow)
        # move forward
        if j == 1 # refreshments of velocities
            θ, θf = refresh_sticky_vel!(θ, θf, Flow)
            tref = t + waiting_time_ref(Flow) # regenerate refreshment time
            b = ab(x, θ, c, Flow) # regenerate reflection time
            told = t
            t′ = t + poisson_time(b, rand())
            tfrez = freezing_time!(tfrez, t, x, θ, Flow)
        elseif j == 2
            if f[i] # hit 0 -> freeze
                if abs(x[i]) > 1e-8
                    error("x[i] = $(x[i]) !≈ 0")
                end
                x[i] = -0*θ[i]
                θf[i], θ[i] = θ[i], 0.0 # stop and save speed
                f[i] = false # change tag
                # tfrez[i] = t - log(rand())/(κ*abs(θf[i])) #option 1
                tfrez[i] = t - log(rand())/κ # option 2
                if !(strong_upperbounds) #not strong upperbounds, draw new waiting time
                    b = ab(x, θ, c, Flow) # regenerate reflection time
                    told = t
                    t′ = t + poisson_time(b, rand())
                end
            else # was frozen ->  unfreeze
                @assert x[i] == 0 && θ[i] == 0
                θ[i], θf[i] = θf[i], 0.0 # restore speed
                f[i] = true # change tag
                tfrez[i] = Inf
                b = ab(x, θ, c, Flow) # regenerate reflection time
                told = t
                t′ = t + poisson_time(b, rand())
            end
        else #   t′ usual boucy particle step
            ∇ϕx = ∇ϕ!(∇ϕx, x, args...)
            ∇ϕx = grad_correct!(∇ϕx, x, Flow)
            l, lb = λ(∇ϕx, θ, Flow), sλ̄(b, t - told) ##
            num += 1
            if rand()*lb <= l # reflect!
                acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ = reflect_sticky!(∇ϕx, x, θ, Flow)
                b = ab(x, θ, c, Flow) # regenerate reflection time
                told = t
                t′ = t + poisson_time(b, rand())
                tfrez = freezing_time!(tfrez, t, x, θ, Flow)
            else # nothing happened
                b = ab(x, θ, c, Flow)
                told = t
                t′ = t + poisson_time(b, rand())
                continue
            end
        end
        push!(Ξ, event(t, x, θ, Flow))
        return t, x, θ, t′, tref, tfrez, told, f, θf, (acc, num), c, b
    end
end


function sticky_pdmp(∇ϕ!, t0, x0, θ0, T, c, Flow::Union{BouncyParticle, Boomerang},
        κ, args...;  strong_upperbounds = false, adapt=false, factor=2.0)
    t, x, θ, ∇ϕx = t0, deepcopy(x0), deepcopy(θ0), deepcopy(θ0)
    told = t0
    θf = deepcopy(θ) # tags
    f = [true for _ in eachindex(x)]
    Ξ = Trace(t0, x0, θ0, Flow)
    push!(Ξ, event(t, x0, θ0, Flow))
    tref = waiting_time_ref(Flow) #refreshment times
    tfrez = zero(x)
    tfrez = freezing_time!(tfrez, t, x, θ, Flow) #freexing times
    num = acc = 0
    b = ab(x, θ, c, Flow)
    t′ = t + poisson_time(b, rand()) # reflection time
    while t < T
        t, x, θ, t′, tref, tfrez, told, f, θf, (acc, num), c, b = sticky_pdmp_inner!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, b, t′, f, θf, tfrez, tref, told, (acc, num),
                Flow, κ, args...; strong_upperbounds = strong_upperbounds, factor = factor, adapt = adapt)
    end
    return Ξ, (t, x, θ), (acc, num), c
end
