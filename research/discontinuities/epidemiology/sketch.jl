using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
using DataStructures

x0 = collect(1.0:1.0:6.0)
v0 = [i%3==1 ? rand([1.0, -1.0]) : 0.0 for i in 1:6]

function hit_time(i, x, v, f, T)
    n = length(x)
    j = findfirst(k -> k==i, f)
    if v[i] == 0.0
        return error("")
    elseif v[i] == 1.0
        k = j + 1
        if j == n
            return (T-x[i])/v[i]
        elseif v[f[k]] == 1.0
            return Inf
        else
            return (x[f[k]] - x[i])/(v[i] - v[f[k]]) 
        end
    else # v[i] == -1.0
        k = j - 1
        if j == 1
            return -x[i]/v[i]
        elseif v[f[k]] == -1.0 || v[f[k]] == 1.0
            return Inf
        else
            return (x[f[k]] - x[i])/(v[i] - v[f[k]])  
        end
    end
end

function cont_time(i, x, v, f, T)
    return -log(rand())
end

function next_time(i, x, v, f, T)
    t1 = hit_time(i, x, v, f, T)
    t2 = cont_time(i, x, v, f, T)
    if t1 < t2
        return t1, 2
    else
        return t2, 1
    end
end

function update_nhb!(Q, i, x, v, f, t, T)
    j = findfirst(k -> k == i, f)
    if j == length(x) #last element
        nhb = f[j-1]
        if nhb % 3 == 1
            τ, tag = next_time(nhb, x, v, f, T)
            Q[nhb] = (t + τ, tag)
        end
    elseif j == 1 # first 
        nhb = f[j+1]
        if nhb % 3 == 1
            τ, tag = next_time(nhb, x, v, f, T)
            Q[nhb] = (t + τ, tag)
        end
    else
        for nhb in [f[j-1], f[j+1]] 
            if nhb % 3 == 1
                τ, tag = next_time(nhb, x, v, f, T)
                Q[nhb] = (t + τ, tag)
            end
        end
    end
    return Q
end

function zz_epi(x0, v0, T, zzclock)
    n = length(x0)
    t = 0.0
    f0 = sortperm(x0)
    Ξ = [(t, (x0, v0, f0))] 
    x = copy(x0)
    v = copy(v0)
    f = copy(f0)
    Q = PriorityQueue{Int64, Tuple{Float64, Int64}}()
    for i in 1:n
        if i%3 == 1
            τ, tag = next_time(i, x, v, f, T)
            enqueue!(Q, i => (t + τ, tag))
        end
    end
    while t < zzclock
        Q, Ξ, x , v, f, t = zz_inner!(Q, Ξ, x , v, f, t, T)
    end
    return Ξ
end

function zz_inner!(Q, Ξ, x , v, f, t, T)
    i, (t′, tag) = dequeue_pair!(Q) 
    @assert i % 3 == 1
    x += v.*(t′ - t)
    t = t′
    if tag == 1 #continuous time
        if rand()>0.5
            v[i] *= -1
            τ, tag = next_time(i, x, v, f, T)
            enqueue!(Q, i => (t + τ, tag))
            #neighbours
            Q = update_nhb!(Q, i, x, v, f, t, T)
        else
            τ, tag = next_time(i, x, v, f, T)
            enqueue!(Q, i => (t + τ, tag))
        end
    else # tag = 2 hitting time
        j = findfirst(k -> k == i, f)
        # f[j] = i 
        v[i] == 1 ? k = j+1 : k = j-1
        if k == 0
            x[i] = 0.0
            v[i] *= -1
            τ, tag = next_time(i, x, v, f, T)
            enqueue!(Q, i => (t + τ, tag))
            #neighbours
            Q = update_nhb!(Q, i, x, v, f, t, T)
        elseif k == length(x)+1
            x[i] = T
            v[i] *= -1
            τ, tag = next_time(i, x, v, f, T)
            enqueue!(Q, i => (t + τ, tag))
            #neighbours
            Q = update_nhb!(Q, i, x, v, f, t, T)
        else
            nhb = f[k]
            x[i] = x[nhb]
            if rand() > 0.5
                v[i] *= -1
                v[nhb] *= -1
                τ, tag = next_time(i, x, v, f, T)
                enqueue!(Q, i => (t + τ, tag))
                #neighbours
                Q = update_nhb!(Q, i, x, v, f, t, T)
            else
                #  k  = j ± 1
                # nhb = f[k] index of neighbours
                # f[j] = nhb 
                # f[k] = i
                # error("")
                c = f[j]
                f[j] = f[k]
                f[k] = c   
                τ, tag = next_time(i, x, v, f, T)
                enqueue!(Q, i => (t + τ, tag))
                if nhb % 3 == 1
                    τ, tag = next_time(nhb, x, v, f, T)
                    Q[nhb] = (t + τ, tag)
                end
                Q = update_nhb!(Q, i, x, v, f, t, T)
                Q = update_nhb!(Q, nhb, x, v, f, t, T)
            end
        end
    end
    push!(Ξ, (t, (x, v, f)))
    return Q, Ξ, x , v, f, t
end

x0[4] = 1.1
Ξ = zz_epi(x0, v0, 7.0, 20.0)
t = first.(Ξ)
x1 = getindex.(getindex.(last.(Ξ),1),1)
x2 = getindex.(getindex.(last.(Ξ),1),4)
using GLMakie
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, t, x1)
lines!(ax, t, x2)
vlines!(ax, first.(Ξ), color = (:blue, 0.1))
hlines!(ax, collect(1:7), color = (:red, 0.3))
fig