using Base.Threads
using Base.Threads: @spawn, fetch


struct Partition
    nt::Int
    n::Int
end
Base.length(pt::Partition) = pt.nt
(pt::Partition)(i) = div(pt.nt*(i-1), pt.n) + 1, i - div(pt.n,pt.nt)*div(pt.nt*(i-1), pt.n)
(pt::Partition)(q1, q2) = (q1-1)*pt.nt + q2 

each(pt::Partition) = 1:pt.nt

function parallel_innermost!(partition, G, G1, G2, ∇ϕ, i, t, x, θ, t′, Q, c, b, t_old,
    F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)
    t, x, θ = smove_forward!(G, i, t, x, θ, t′, F)
    ∇ϕi = ∇ϕ(x, i, args...)
    l, lb = sλ(∇ϕi, i, x, θ, F), sλ̄(b[i], t[i] - t_old[i])
    num = 1
    if rand()*lb < l
        if l >= lb
            !adapt && error("Tuning parameter `c` too small.")
            adapt!(c, i, factor)
        end
        t, x, θ = smove_forward!(G2, i, t, x, θ, t′, F)
        θ = reflect!(i, ∇ϕi, x, θ, F)
        for j in neighbours(G1, i)
            b[j] = ab(G1, j, x, θ, c, F)
            t_old[j] = t[j]
            q1, q2 = partition(j)
            Q[q1][q2] = t[j] + poisson_time(b[j], rand())
        end
        return true
    else
        b[i] = ab(G1, i, x, θ, c, F)
        t_old[i] = t[i]
        q1, q2 = partition(i)
        Q[q1][q2] = t[i] + poisson_time(b[i], rand())
        return false
    end
end

function parallel_spdmp_inner!(partition, ti, inner, G, G1, G2, ∇ϕ, t, x, θ, Q, c, b, t_old,
    F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false)
   n = length(x)
   num = 0
   while true
        num += 1
        ii, t′ = peek(Q[ti])
        i = partition(ti, ii)
        if !inner[i] # need neighbours at t′
            return false, event(i, t, x, θ, F), i, t′, num
        end

        success = parallel_innermost!(partition, G, G1, G2, ∇ϕ, i, t, x, θ, t′, Q, c, b, t_old, F, args...; factor=factor, adapt=adapt)

        success || continue
        return true, event(i, t, x, θ, F), i, t′, num
   end
end

function parallel_spdmp(partition, ∇ϕ, t0, x0, θ0, T, c, G, F::Union{ZigZag,FactBoomerang}, args...;
    factor=1.8, adapt=false)
    nthr = length(partition)
 #   @assert nthr == Threads.nthreads()
    n = length(x0)
    t′ = t0
    t = fill(t′, size(θ0)...)
    t_old = copy(t)
    if G === nothing
        G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    end
    inner = [all(j -> partition(j) == partition(i), js) for  (i,js) in G]
    G1 = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]

    @assert all(a.second ⊇ b.second for (a,b) in zip(G, G1))

    G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]
    x, θ = copy(x0), copy(θ0)
    num = acc = 0
    Q = [SPriorityQueue{Int,Float64}() for thr in 1:nthr]
    b = [ab(G1, i, x, θ, c, F) for i in eachindex(θ)]
    for i in eachindex(θ)
        q1, q2 = partition(i)
        enqueue!(Q[q1], q2 => poisson_time(b[i], rand()))
    end
    Ξ = Trace(t0, x0, θ0, F)
    task = Vector{Task}(undef, 5)
    evtime = zeros(nthr)
    perm = collect(1:nthr)
    while t′ < T
        for ti in each(partition)
            task[ti] = @spawn parallel_spdmp_inner!(partition, ti, inner, G, G1, G2, ∇ϕ, t, x, θ, Q,
                        c, b, t_old, F, args...; factor=factor, adapt=adapt)
        end    
        res = [fetch(task[ti]) for ti in each(partition)]
        for ti in each(partition)
            evtime[ti] = res[ti][4]
        end
        sortperm!(perm, evtime)
        for ti in perm
            done, ev, i, t′_, num_ = res[ti]
            num += num_
            t′ = max(t′, t′_)
            if !done  
                success = parallel_innermost!(partition, G, G1, G2, ∇ϕ, i, t, x, θ, t′, Q, c, b, t_old, F, args...; factor=factor, adapt=adapt)
            else
                success = true
            end
            if success
                acc += 1
                push!(Ξ, ev)
            end
        end

    end
    Ξ, (t, x, θ), (acc, num)
end

