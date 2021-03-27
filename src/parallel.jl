using Base.Threads
using Base.Threads: @spawn, fetch
const VERBOSE = false

struct Partition{k}
    nt::Int
    n::Int
end
chunksize(pt::Partition{k}) where {k} = k
chunksize(pt::Type{Partition{k}}) where {k} = k
function index_slow(pt, i)
    j, i′ = divrem(i-1, chunksize(pt))
    j + 1, i′ + 1
end
@generated function index_fast(pt, i)
  pow = round(Int, log2(chunksize(pt)))
  :((i-1) >> $pow + 1, (i-1) & $(2^pow-1) + 1)
end

@generated function (pt::Partition{k})(i) where {k}
  ispow2(k) ?
    :(index_fast(pt, i)) :
    :(index_slow(pt, i))
end

Partition(nt,n) = Partition{div(n, nt)}(nt, n)
Base.length(pt::Partition) = pt.nt
#(pt::Partition)(i) = div((i-1), pt.k) + 1, i - pt.k*div((i-1), pt.k)
(pt::Partition{k})(q1, q2) where{k} = (q1-1)*k + q2 
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

function parallel_spdmp_inner!(latch, wakeup, ret, events, partition, ti, (t0, Δ), inner, G, G1, G2, ∇ϕ, t, x, θ, Q, c, b, t_old,
    F::Union{ZigZag,FactBoomerang}, (factor, adapt), args...)
   acc = num = 0
   VERBOSE && println("$ti starts.")
   tnext = t0 + Δ

   while true
        num += 1
        ii, t′ = peek(Q[ti])
        i = partition(ti, ii)
        if !inner[i] ||  t′ > tnext # need neighbours at t′, or just wait
            tnext = t′ + Δ
            ret[] = i, t′, acc, num
            u = UInt(1) << (ti - 1)
            ac = Threads.atomic_and!(latch.active, ~u)
            done = false
            lock(wakeup) do
                if ac == u # last one turns the light off
                    VERBOSE && println("Notify $ti")
                    lock(latch.condition) do
                        notify(latch.condition, nothing; all = true, error = false)
                    end
                end
            
                VERBOSE && println("Sleep: $ti at $(t′) ($ac, $u).")
                done = wait(wakeup)
            end
            if done
                println("Done: $ti.")     
                return
            end
            acc = num = 0
        else
            success = parallel_innermost!(partition, G, G1, G2, ∇ϕ, i, t, x, θ, t′, Q, c, b, t_old, F, (factor, adapt), args...)
            success || continue
            acc += 1
            push!(events, event(i, t, x, θ, F))
        end
   end
end

function parallel_spdmp(partition, ∇ϕ, t0, x0, θ0, T, c, G, F::Union{ZigZag,FactBoomerang}, args...;
    factor=1.8, adapt=false, Δ = 0.1, progress=()->return)
    nthr = length(partition)
    n = length(x0)
    t′ = fill(t0, nthr) 
    t = fill(t0, size(θ0)...)
    t_old = copy(t)
    if G === nothing
        G = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]
    end
    inner = [all(j -> partition(j)[1] == partition(i)[1], js) for  (i,js) in G]

    G1 = [i => rowvals(F.Γ)[nzrange(F.Γ, i)] for i in eachindex(θ0)]

    @assert all(a.second ⊇ b.second for (a,b) in zip(G, G1))

    G2 = [i => setdiff(union((G1[j].second for j in G1[i].second)...), G[i].second) for i in eachindex(G1)]

    if !all(all( first(partition(g[1])) .== first.(partition.(g[2]))) for g in G2)
        error("Upper bounds may not depend across chunks.")

    end    
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
    events = [resize!([event(1, 0., x, θ, F)], 0) for ts in each(partition)]
    res = [Ref((1, 1.0, 1, 1)) for ts in each(partition)]
    latch = (;active = Threads.Atomic{UInt}(1), condition=Threads.Condition())
    wakeup = [Threads.Condition() for _ in each(partition)]
    parallel_spdmp_loop(t′, T, task, waitfor, latch, wakeup, evtime, perm, res, Ξ, events, partition, inner, G, G1, G2, ∇ϕ, t, x, θ, Q,
    c, b, t_old, F, (Δ, factor, adapt), progress, args...)
end
function parallel_spdmp_loop(t′, T, task, waitfor, latch, wakeup, evtime, perm, res, Ξ, events, partition, inner, G, G1, G2, ∇ϕ, t, x, θ, Q,
    c, b, t_old, F, (Δ, factor, adapt), progress, args...)

    tmin = minimum(t′)
    for ti in each(partition)
        Threads.atomic_or!(latch.active, UInt(1) << (ti - 1))
        task[ti] = Threads.@spawn parallel_spdmp_inner!(latch, wakeup[ti], res[ti], events[ti], partition, ti, (tmin, Δ), inner, G, G1, G2, ∇ϕ, t, x, θ, Q,
            c, b, t_old, F, (factor, adapt), args...) 
    end
    task_outer = Threads.@spawn parallel_spdmp_outer!(tmin, t′, T, task, waitfor, latch, wakeup, evtime, perm, res, Ξ, events, partition, inner, G, G1, G2, ∇ϕ, t, x, θ, Q,
    c, b, t_old, F, (Δ, factor, adapt), progress, args...) 

    for ti in each(partition)
        wait(task[ti]) 
    end
 
    (acc, num), (run, runs), (run2, runs2) = fetch(task_outer)
    println("events per spawn: ", run/runs, " spawns: ", runs)
    println("wakeups per round: ", run2/runs2, " rounds: ", runs2)

    sort!(Ξ.events, by=ev->ev[1])
    Ξ, (t, x, θ), (acc, num)
end    
function parallel_spdmp_outer!(tmin, t′, T, task, waitfor, latch, wakeup, evtime, perm, res, Ξ, events, partition, inner, G, G1, G2, ∇ϕ, t, x, θ, Q,
    c, b, t_old, F, (Δ, factor, adapt), progress, args...) 
    acc = num = 0
    run = runs = 0
    run2 = runs2 = 0
    stops = 20
    tstop = T/stops
    while tmin < T
        if latch.active[] !== 0
            VERBOSE && println("Waiting $tmin")
            lock(latch.condition) do
                while latch.active[] != 0
                    wait(latch.condition)
                end
            end
        end        
        for ti in each(partition)
            if waitfor[ti] == 0 
                run += res[ti][][end]
                runs += 1
                #println(res[ti][end])
                append!(Ξ.events, events[ti])
                resize!(events[ti], 0)
                evtime[ti] = res[ti][][2]
            end
        end
        sortperm!(perm, evtime, alg=InsertionSort)
        alldone = true
        for ti in perm
            i, t′_, acc_, num_ = res[ti][]
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

            success = parallel_innermost!(partition, G, G1, G2, ∇ϕ, i, t, x, θ, t′_, Q, c, b, t_old, F, (factor, adapt), args...)
        
            if success
                acc += 1
                push!(Ξ, event(i, t, x, θ, F))
            end
        end

        tmin = minimum(t′)
        runs2 += 1
        if tmin > tstop
            tstop += T/stops
            progress()  
        end  
        for ti in each(partition)
            if waitfor[ti] .== 0 ||  tmin >= T
                run2 += 1
                Threads.atomic_or!(latch.active, UInt(1) << (ti - 1))
                lock(wakeup[ti]) do
                    notify(wakeup[ti], tmin >= T)
                end
            end
        end 
     

        tmin >= T && return (acc, num), (run, runs),  (run2, runs2)
 
    end

end

