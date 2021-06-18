struct JointFlow{F,R}
    f::F
    λref::R
end
hasrefresh(F::JointFlow) = true

waiting_time_ref(rng, F::JointFlow) = poisson_time(rng, F.λref)/length(F.f)
waiting_time_ref(F::JointFlow) = poisson_time(F.λref)/length(F.f)


function grad_correct!(y, x, F::JointFlow)
    for i in eachindex(y)
        y[i] += grad_correct(x[i], F[i])
    end
    y
end
Base.getindex(F::JointFlow, args...) = getindex(F.f, args...)

function smove_forward!(G, i, t, x, θ, t′, F::JointFlow)
    nhd = neighbours(G, i)
    for i in nhd
        x[i], θ[i] = move_forward(t[i], x[i], θ[i], t′, F[i])
        t[i] = t′
    end
    return t, x, θ
end
function smove_forward!(i::Int, t, x, θ, t′, F::JointFlow)
    x[i], θ[i] = move_forward(t[i], x[i], θ[i], t′, F[i])
    t[i] = t′
    return t, x, θ
end

function move_forward!(τ, t, x, θ, F::JointFlow)
    t′ = t + τ
    for i in eachindex(x)
        x[i], θ[i] = move_forward(t, x[i], θ[i], t′, F[i])
    end
    t += τ
    t, x, θ
end

