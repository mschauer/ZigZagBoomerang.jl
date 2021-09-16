
# First Faber Schauder Basis evaluated at time `t`
Λ(t, T::Float64) = sqrt(T)*0.5 - abs((t % T)/sqrt(T) - sqrt(T)*0.5)

# Rescaled Faber Schauder Basis evaluated at time `t`
Λ(t, l⁻::Int64, T::Float64) = Λ(t*(1<<l⁻), T)/sqrt(1<<l⁻)


"""
    dotψ(ξ, s, L, T, u, v)

Given the truncated FS expansion with truncation level `L` and
coefficients `ξ`, output the value of the diffuion bridge at time `s` (`r`)
with initial value `u` at time 0 and final value `v` at `T`.
"""
function dotψ(ξ, s, L, T)
    0 <= s < T || error("out of bounds")
    r = s/sqrt(T)*ξ[end] + sqrt(T)*(1 - s/T)*ξ[1]
    for i in 0:L
        j = floor(Int, s/T * (1 << (L - i)))*(2 << i) + (1 << i) + 1
        r += ξ[j]*Λ(s, L-i, T)
    end
    r
end

"""
    dotψmoving(t, ξ, θ, t′, s, F, L, T, u, v)

Jointly updates the coefficeints (locally) and evaluates the diffuion bridge at time `s`.

Given the truncated FS expansion with truncation level `L` and
coefficients `ξ` and velocities `θ`, move first the coefficients required for
the evaluation of the diffuion bridge at time `s` up to time `t′` (according
to the dynamics of the sampler `F`) and output the
value of diffuion bridge at time `s` with initial value `u` at time 0 and final value `v` at `T`.
"""
function dotψmoving(t, ξ, θ, t′, s, F, L, T)
    0 <= s <= T || error("out of bounds")
    r = s/sqrt(T)*ξ[end] + sqrt(T)*(1 - s/T)*ξ[1]
    for i in 0:L
        j = floor(Int, s/T*(1 << (L - i)))*(2 << i) + (1 << i) + 1
        ZigZagBoomerang.smove_forward!(j, t, ξ, θ, t′, F)
        r += ξ[j]*Λ(s, L-i, T)
    end
    r
end

# find level of index i
"""
    lvl0(i)

Level of element i in the Faber-Schauder base for functions pinned
down `f(0) = f(T) = 0` with elements
ordered according to their midpoint,

```
lvl0.(1:17)
[0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0]
```
"""
function lvl0(i)
    l = 0
    while (i & 1) == 0
        l += 1
        i = i >> 1
    end
    l
end

"""
    lvl(i, L)

Level of element i in the Faber-Schauder base with elements
ordered according to their midpoint.

```
lvl.(1:17, 3)
[3, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 3]
```
"""
function lvl(i, L)
    if i == 1 || i == (2 << L) + 1
        return L
    else
        return lvl0(i-1)
    end
end

"""
    ∇ϕ(ξ, i, K, L, T)

Unbiased estimate for the `i`th partial derivative of the potential function.
The variance of the estimate can be reduced by averaging over `K` independent realization.
`ξ` is the current position of the coefficients, `L` the truncation level.
The bridge has initial value `u` at time 0 and final value `v` at `T`.
"""
function ∇ϕ(ξ, i, K, L, T) # formula (17)
    if i == (2 << L) + 1    # final point
        s = T*(rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L,  T)
        return -b(ξ[end]*sqrt(T))*sqrt(T) + 0.5*sqrt(T)*s*(2b(x)*b′(x) + b″(x)) + ξ[i] - ξ[1]
    elseif i == 1   # initial point
        s = T*(rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L,  T)
        return b(ξ[1]*sqrt(T))*sqrt(T) + 0.5*T^(1.5)*(1 - s/T)*(2b(x)*b′(x) + b″(x)) + ξ[i]
    else
        l = lvl(i, L)
        k = (i - 1) ÷ (2 << l)
        δ = T/(1 << (L-l)) # T/(2^(L-l))
        r = 0.0
        for _ in 1:K
            s = δ*(k + rand())
            x = dotψ(ξ, s, L,  T)
            r += 0.5*δ*Λ(s, L-l, T)*(2b(x)*b′(x) + b″(x)) + ξ[i]
        end
        return r/K
    end
end


"""
    ∇ϕmoving(t, ξ, θ, i, t′, F, L, T, u, v)

Jointly updates the coefficeints (locally) and estimates
the `i`th partial derivative of the potential function.
The bridge has initial value `u` at time 0 and final value `v` at `T`.
"""
function ∇ϕmoving(t, ξ, θ, i, t′, F, L, T) # formula (17)
    if i == (2 << L) + 1    # final point
        s = T * (rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L, T)
        return -b(ξ[end]*sqrt(T))*sqrt(T) + 0.5 * sqrt(T) * s * (2b(x) * b′(x) + b″(x)) + ξ[i] - ξ[1]
    elseif i == 1   # initial point
        s = T * (rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L, T)
        return 0.5 * T^(1.5) * (1 - s / T) * (2b(x) * b′(x) + b″(x)) + ξ[1]
    else
        l = lvl(i, L)
        k = (i - 1) ÷ (2 << l)
        δ = T / (1 << (L - l))
        s = δ * (k + rand())
        x = dotψmoving(t, ξ, θ, t′, s, F, L, T)
        return 0.5 * δ * Λ(s, L - l, T) * (2b(x) * b′(x) + b″(x)) + ξ[i]
    end
end

"""
    ∇ϕ!(y, ξ, k, L,  T, u, v)

In-place evaluation of the gradient of the potential function.
`ξ` is the current position, `k` is the number of MC realization,
`L` is the truncation level. The bridge has initial value `u` at time 0 and final value `v` at `T`.
"""
function ∇ϕ!(y, ξ, k, L, T)
    for i in eachindex(ξ)
        y[i] = ∇ϕ(ξ, i, k, L, T)
    end
    y
end
