
using Gen
using ZigZagBoomerang
using SparseArrays
using LinearAlgebra
using Distributions


"""
    genpdmp(Flow, θ,
        trace, selection::Selection;
        check=false, observations=EmptyChoiceMap(), c=1e-5)

c is a rejection sampler constant. Modifies θ.
Works best with PDMP samplers with high refreshment rate and
Crank-Nicholson parameter ρ close to one.

```julia
Γ = sparse(Diagonal(1.0, n, n)
μ = zeros(n)
θ = randn(n)
Flow = BouncyParticle(Γ, μ, 2.0, ρ = 0.995)

trace, _ = genpdmp(Flow, θ, trace, selection)
```

"""
function genpdmp(Flow, θ,
        trace, selection::Selection;
        check=false, observations=EmptyChoiceMap(), c=1e1, λ0=Flow.λref)

    args = get_args(trace)
    retval_grad = accepts_output_grad(get_gen_fn(trace)) ? zero(get_retval(trace)) : nothing
    argdiffs = map((_) -> NoChange(), args)


    (_, values_trie, gradient_trie) = choice_gradients(trace, selection, retval_grad)
    values = to_array(values_trie, Float64)
    gradient = to_array(gradient_trie, Float64)

    x = values

    ZigZagBoomerang.refresh!(θ, Flow)
    t = 0.0
    ∇ϕx = copy(θ)
    acc = num = 0
    function ∇ϕ!(y, x, values_trie, trace, args, argdiffs, selection, retval_grad)
        values = x


        values_trie[] = from_array(values_trie[], values)
        @assert keys(values_trie[].leaf_nodes)[1] == :slope
        trace[]  = update(trace[], args, argdiffs, values_trie[])[1]
        (_, _, gradient_trie) = choice_gradients(trace[], selection, retval_grad)
        gradient = to_array(gradient_trie, Float64)


        @. y = -gradient
        y
    end

    Ξ = ZigZagBoomerang.Trace(t, x, θ, Flow)
    τref = T = ZigZagBoomerang.waiting_time_ref(Flow)
    a, b = ZigZagBoomerang.ab(x, θ, c, Flow)
    t′ = t + poisson_time(a, b, rand())
    while t < T
        t, x, θ, (acc, num), c, a, b, t′, τref = ZigZagBoomerang.pdmp_inner!(Ξ, ∇ϕ!, ∇ϕx, t, x, θ, c, a, b, t′, τref, (acc, num), Flow,
            Ref(values_trie), Ref(trace), args, argdiffs, selection, retval_grad; adapt=false)
    end

    values = x
    trace = trace
    values_trie = from_array(values_trie, values)
    (trace, _, _) = update(trace, args, argdiffs, values_trie)

    check && check_observations(get_choices(trace), observations)

    return trace, true
end
