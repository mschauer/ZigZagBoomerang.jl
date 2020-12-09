

λ(∇ϕx, θ, Flow::ZigZag) = pos(∇ϕx*θ)

function event(t, x, θ, Z::ZigZag)
    t, copy(x), copy(θ)
end



function pdmp_inner_sub!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, a, b, t′, τref, (acc, num),
     Flow::ZigZag, args...; factor=1.5, adapt=false)
    while true
        if τref < t′
            error("don't go here")
            t, x, θ = move_forward!(τref - t, t, x, θ, Flow)
            #θ = randn!(θ)
            θ = refresh!(θ, Flow)
            τref = t + waiting_time_ref(Flow)
            a, b = ab(x, θ, c, Flow, args...)
            t′ = t + poisson_time(a, b, rand())
            push!(Ξ, event(t, x, θ, Flow))
            return t, x, θ, (acc, num), c, a, b, t′, τref
        else
            τ = t′ - t
            t, x, θ = move_forward!(τ, t, x, θ, Flow)
            (E_tilde, i) = ∇ϕ!(x, args...)
            l, lb = λ(E_tilde, θ[i], Flow), pos(a + b*τ)
            num += 1
            if rand()*lb <= l
                acc += 1
                if l > lb
                    !adapt && error("Tuning parameter `c` too small.")
                    c *= factor
                end
                θ[i] = -θ[i]
                push!(Ξ, event(t, x, θ, Flow))
                a, b = ab(x, θ, c, Flow)
                t′ = t + poisson_time(a, b, rand())
                return t, x, θ, (acc, num), c, a, b, t′, τref
            end
            a, b = ab(x, θ, c, Flow, args...)
            t′ = t + poisson_time(a, b, rand())
        end
    end
end


function pdmp_sub(∇ϕ!, t0, x0, θ0, T, c, Flow::ZigZag, args...; adapt=false, factor=2.0)
    t, x, θ, ∇ϕx = t0, copy(x0), copy(θ0), copy(θ0)
    Ξ = Trace(t0, x0, θ0, Flow)
    τref = waiting_time_ref(Flow)
    num = acc = 0
    a, b = ab(x, θ, c, Flow, args...)
    t′ = t + poisson_time(a, b, rand())
    while t < T
        t, x, θ, (acc, num), c, a, b, t′, τref = pdmp_inner_sub!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, a, b, t′, τref, (acc, num), Flow, args...; factor=factor, adapt=adapt)
    end
    return Ξ, (t, x, θ), (acc, num), c
end
