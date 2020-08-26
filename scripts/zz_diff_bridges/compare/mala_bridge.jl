
#########################################
# Mala for comparison with adaptation   #
#########################################

function get_info(ϕ, ∇ϕ, ξ, L, α, T)
    U = 0.5*dot(ξ, ξ) #gaussian part
    ∇U = deepcopy(ξ) #gaussian part
    dt = T/(2<<L)
    t = dt*0.5
    tt = 0:T/(2<<L):T
    N = Int(T/dt)
    for i in 1:N
        Xₜ = dotψ(ξ, t, L, T)
        U += ϕ(Xₜ, ξ, t, L, α, T)*dt
        for i in 2:(length(ξ)-1)
            l = lvl(i, L)
            k = (i - 1) ÷ (2 << l)
            δ = T / (1 << (L - l))
            if δ*k < t < δ*(k+1)
                ∇U[i] += ∇ϕ(Xₜ, l, t, ξ,  L, α, T)*dt
            end
        end
        t += dt
    end
    ∇U[1] = ∇U[end] = 0.0
    return ∇U, U
end

function mala_sampler(ϕ, ∇ϕ, ξ₀::Vector{Float64}, niter::Int64, args...)
        ξtrace = Array{Float64}(undef, length(ξ₀), niter)
        ξtrace[:,1] = ξ₀
        ∇U₀, U₀ = get_info(ϕ, ∇ϕ, ξ₀, args...)
        ξ₁ = deepcopy(ξ₀)
        τ = 0.015
        count = 0
        for i in 2:niter
                # initial and final point fixed
                for j in 2:length(ξ₀)-1
                    ξ₁[j]  = ξ₀[j] - τ*∇U₀[j] + sqrt(2*τ)*randn()
                end
                ∇U₁, U₁ = get_info(ϕ, ∇ϕ, ξ₁, args...)
                acc_rej =  U₀ - U₁ + (norm(ξ₁ - ξ₀ + τ*∇U₀)^2 - norm(ξ₀ - ξ₁ + τ*∇U₁)^2)/(4τ)
                # @show acc_rej
                # @assert acc_rej == 0.0
                if acc_rej > log(rand())
                        ∇U₀, U₀, ξ₀ = deepcopy(∇U₁), deepcopy(U₁), deepcopy(ξ₁)
                        count += 1
                end
                if i % 100 == 0
                    if count/100 <= 0.6
                        τ = max(0.0001, τ - 0.0001)
                    else
                        τ = min(1.0, τ + 0.0001)
                    end
                    #println("Adaptive step: ar: $(count/100), new tau: $τ")
                    count = 0
                end
                ξtrace[:,i] = ξ₀
        end
        ξtrace
end

# sin application
function ϕ(Xₜ, ξ, t, L, α, T)
    0.5*α*(α*sin(Xₜ)^2 + cos(Xₜ))
end

function ∇ϕ(Xₜ, l, t, ξ,  L, α, T)
    Λ(t, L - l, T)*0.5*(α^2*sin(2.0*Xₜ) - α*sin(Xₜ))
end

function run_mala(df, niter, α)
    L = 5
    T = 100.0
    n = (2 << L) + 1
    ξ₀ =  randn(n)
    u, v = 0.0, 0.0  # initial and fianl point
    ξ₀[1] = u / sqrt(T)
    ξ₀[end] = v / sqrt(T)
    dt = 1/(2 << L)
    mala_time = @elapsed (out = mala_sampler(ϕ, ∇ϕ, ξ₀, niter, L, α, T))
    ess = ESS(out, n_batches = 50)
    ess_xt2 = ess[Int((length(ξ₀)+1)/2)]
    push!(df, Dict(:sampler => "MALA", :alpha => α, :ess_XT2 => ess_xt2/mala_time, :ess_mean => sum(ess[2:end-1])/(length(ess)-2)/mala_time,
            :ess_median => median(ess[2:end-1])/mala_time, :ess_min => minimum(ess[2:end-1])/mala_time, :runtime => mala_time ), )
    # S = T*(0:n)/(n+1)
    # p1 = lines(S, [dotψ(xx[:,end], s, L, T) for s in S], linewidth=0.3)
    # for i in 1:100:size(xx)[2]
    #     lines!(p1, S, [dotψ(xx[:, i], s, L, T) for s in S], linewidth=0.1, alpha = 0.1)
    # end
    # display(p1)
    return df
end

function data_collection_mala(df)
    niter = 300000
    for α in [0.1, 0.3, 0.7]
        df = run_mala(df, niter, α)
    end
    return df
end
df = DataFrame(sampler = String[], alpha = Float64[], ess_XT2 = Float64[],  ess_mean = Float64[],
    ess_median = Float64[], ess_min = Float64[], runtime = Float64[])
df = data_collection_mala(df)
# With Automatic Integration (TOO SLOW)
using CSV
CSV.write("./scripts/zz_diff_bridges/compare/benchamrk_mala.csv", df)



error("Stop here")
using QuadGK

# sin application
function ϕ(ξ, t, L, α, T)
    Xₜ = dotψ(ξ, t, L, T)
    0.5*α*(α*sin(Xₜ)^2 + cos(Xₜ))
end
function ∇ϕ(l, t, ξ,  L, α, T)
    Xₜ = dotψ(ξ, t, L, T)
    #println(Λ(t, L - l, T)*0.5*(α^2*sin(2.0*Xₜ) - α*sin(Xₜ)))
    0.5*(α^2*sin(2.0*Xₜ) - α*sin(Xₜ))*Λ(t, L - l, T)
end

function get_info(ϕ, ∇ϕ, ξ, L, α, T)
    U = 0.5*dot(ξ, ξ) #gaussian part
    ∇U = deepcopy(ξ) #gaussian part
    U += quadgk(t ->  ϕ(ξ, t, L, α, T), 0.0, T, rtol = 1e-1)[1]
    for i in 2:(length(ξ)-1)
        l = lvl(i, L)
        k = (i - 1) ÷ (2 << l)
        δ = T / (1 << (L - l))
        ∇U[i] += quadgk(t -> ∇ϕ(l, t, ξ, L, α, T), δ*k, δ*(k+1), 1e-1)[1]
    end
    ∇U[1] = ∇U[end] = 0.0
    return ∇U, U
end
runall()
