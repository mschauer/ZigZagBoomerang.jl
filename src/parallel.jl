using Base.Threads
using Base.Threads: @spawn, fetch


struct Partition
    nt::Int
    n::Int
    k::Int
end
Partition(nt,n) = Partition(nt, n, div(n, nt))
Base.length(pt::Partition) = pt.nt
(pt::Partition)(i) = div((i-1), pt.k) + 1, i - pt.k*div((i-1), pt.k)
(pt::Partition)(q1, q2) = (q1-1)*pt.k + q2 

each(pt::Partition) = 1:pt.nt

function parallel_innermost!(partition, G, G1, G2, ∇ϕ, i, t, x, θ, t′, Q, c, b, t_old,
    F::Union{ZigZag,FactBoomerang}, (factor, adapt), args...)
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

function parallel_spdmp_inner!(events, partition, ti, inner, G, G1, G2, ∇ϕ, t, x, θ, Q, c, b, t_old,
    F::Union{ZigZag,FactBoomerang}, (factor, adapt), args...)
   n = length(x)
   acc = num = 0
   while true
        num += 1
        ii, t′ = peek(Q[ti])
        i = partition(ti, ii)
        if !inner[i] # need neighbours at t′
           return false, event(i, t, x, θ, F), i, t′, acc, num
        end

        success = parallel_innermost!(partition, G, G1, G2, ∇ϕ, i, t, x, θ, t′, Q, c, b, t_old, F, (factor, adapt), args...)

        success || continue
        acc += 1
        #return true, event(i, t, x, θ, F), i, t′, acc, num
        push!(events[ti], event(i, t, x, θ, F))
   end
end

function parallel_spdmp(partition, ∇ϕ, t0, x0, θ0, T, c, G, F::Union{ZigZag,FactBoomerang}, args...;
    factor=1.8, adapt=false)
    nthr = length(partition)
    n = length(x0)
    t′ = fill(t0, nthr) 
    t = fill(t0, size(θ0)...)
    t_old = copy(t)
    if G === nothing
        G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    end
    inner = [all(j -> partition(j)[1] == partition(i)[1], js) for  (i,js) in G]
 #   @show inner
    G1 = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]

    @assert all(a.second ⊇ b.second for (a,b) in zip(G, G1))

    G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]
    x, θ = copy(x0), copy(θ0)

    Q = [SPriorityQueue{Int,Float64}() for thr in 1:nthr]
    b = [ab(G1, i, x, θ, c, F) for i in eachindex(θ)]
    for i in eachindex(θ)
        q1, q2 = partition(i)
        enqueue!(Q[q1], q2 => poisson_time(b[i], rand()))
    end
    Ξ = Trace(t0, x0, θ0, F)
    task = Vector{Task}(undef, length(partition))
    evtime = zeros(nthr)
    perm = collect(1:nthr)
    waitfor = zeros(Int, nthr)
    events = [[event(1, 0., x, θ, F)] for ts in each(partition)]
    res = [(true, event(1, 0., x, θ, F), 1, 1.0, 1, 1) for ts in each(partition)]
    parallel_spdmp_loop(t′, T, task, waitfor, evtime, perm, res, Ξ, events, partition, inner, G, G1, G2, ∇ϕ, t, x, θ, Q,
    c, b, t_old, F, (factor, adapt), args...)
end
function parallel_spdmp_loop(t′, T, task, waitfor, evtime, perm, res, Ξ, events, partition, inner, G, G1, G2, ∇ϕ, t, x, θ, Q,
    c, b, t_old, F, (factor, adapt), args...)
    num = acc = 0
    run = runs = 0

    while minimum(t′) < T
        @threads for ti in each(partition) # can be parallel
            if waitfor[ti] == 0 
                resize!(events[ti], 0)
                res[ti] = parallel_spdmp_inner!(events, partition, ti, inner, G, G1, G2, ∇ϕ, t, x, θ, Q,
                    c, b, t_old, F, (factor, adapt), args...) 
            end
        end
        
        for ti in each(partition)
            if waitfor[ti] == 0 
                run += res[ti][end]
                runs += 1
                #println(res[ti][end])
                append!(Ξ.events, events[ti])
                evtime[ti] = res[ti][4]
            end
        end
        sortperm!(perm, evtime, alg=InsertionSort)
        for ti in perm
            done, ev, i, t′_, acc_, num_ = res[ti]
            if waitfor[ti] == 0
                num += num_
                acc += acc_
                t′[ti] = t′_
            end

    
            waitfor[ti] = 0
            for j in G[i][2]
                i == j && continue
                q1, q2 = partition(j)
                if t′[q1] < t′_
                    waitfor[ti] = i
                end
            end
      
            waitfor[ti] != 0 && continue

            if !done  
                success = parallel_innermost!(partition, G, G1, G2, ∇ϕ, i, t, x, θ, t′_, Q, c, b, t_old, F, (factor, adapt), args...)
            else
                error("why?")
                success = true
            end
            if success
                acc += 1
                push!(Ξ, ev)
            end
        end

    end
    @show run/runs, runs
    sort!(Ξ.events, by=ev->ev[1])
    Ξ, (t, x, θ), (acc, num)
end

