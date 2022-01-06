##### todo list
# introduce matrices ϑ, ξ
# introduce refuction after notification γ
# introduce delay parameter β  

# add: when i hits his boundary, always reflect
# add: when i is a time of individual (i+2)÷3 
# with tag i%3 = 1 if infected, 2 if notified, 0 if recovered
# add sticky time 

# strange case:  ☑
#   x1 > x2   
#   x1 sticks at T at time t1 	
#   x2 sticks at T at time t2>t1
#   x1 unstick from T at time t3>t2
#   we should change order of f, because now x1 < x2

# sort permutation when unstick   ☑
# insert while loop  ☑
# update swap of values  ☑

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
        return Inf, false
    elseif v[i] == 1.0
        k = j + 1
        if j == n
            return (T-x[i])/v[i], true
        elseif v[f[k]] == 1.0
            return Inf, false
        elseif x[f[k]] == T && v[f[k]] == 0.0  # sticky time
            (T-x[i])/v[i], true
        else
            return (x[f[k]] - x[i])/(v[i] - v[f[k]]), false 
        end
    else # v[i] == -1.0
        k = j - 1
        if j == 1
            return -x[i]/v[i], false
        elseif v[f[k]] == -1.0 || v[f[k]] == 1.0
            return Inf, false # no need to have the same hitting time twice
        else
            return (x[f[k]] - x[i])/(v[i] - v[f[k]]), false  
        end
    end
end

function cont_time(i, x, v, f, T)
    return -log(rand())
end

function next_time(i, x, v, f, T)
    t1, sticky = hit_time(i, x, v, f, T)
    t2 = cont_time(i, x, v, f, T)
    if t1 < t2
        if sticky == false
            return t1, 2 # simple discontinuity
        else
            return t1, 3 # sticky time
        end
    else
        return t2, 1 # continuous time
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
function update_permuation!(i, x, v, f, T)
    j = findfirst(k -> k == i, f) # i = f[j]
    while true
        k = j - 1
        nbh = f[k]
        if x[nbh] != T
            break
        else
            f[j] = nbh
            f[k] = i 
        end
    j -= 1    
    end
    return f
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
    while true
        i, (t′, tag) = dequeue_pair!(Q) 
        @assert i % 3 == 1
        x += v.*(t′ - t)
        t = t′
        if tag == 1 #continuous time
            if v[i] == 0 #usntick particle
                x[i] = T
                v[i] = -1
                f = update_permuation!(i, x, v, f, T)
                τ, tag = next_time(i, x, v, f, T)
                enqueue!(Q, i => (t + τ, tag))
                #neighbours
                Q = update_nhb!(Q, i, x, v, f, t, T)
            elseif rand()>0.5
                v[i] *= -1
                τ, tag = next_time(i, x, v, f, T)
                enqueue!(Q, i => (t + τ, tag))
                #neighbours
                Q = update_nhb!(Q, i, x, v, f, t, T)
            else
                τ, tag = next_time(i, x, v, f, T)
                enqueue!(Q, i => (t + τ, tag))
                continue
            end
        elseif tag == 2  # hitting time but not sticky time
            j = findfirst(k -> k == i, f)
            # f[j] = i
            v[i] == 1 ? k = j+1 : k = j-1 
            if j == 1 && v[i] == -1
                x[i] = 0.0
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
                    f[j] = nhb 
                    f[k] = i
                    # error("")
                    # c = f[j]
                    # f[j] = f[k]
                    # f[k] = c   
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
        else # tag == 3
            x[i] = T
            v[i] = 0.0
            τ, tag = next_time(i, x, v, f, T)
            enqueue!(Q, i => (t + τ, tag))
            #neighbours
            Q = update_nhb!(Q, i, x, v, f, t, T)
        end
        # println("at time $t′, particle $i with tag $tag 
            # and x[i],v[i] = $(x[i]), $(v[i])))  ")
        push!(Ξ, (t, (x, v, f)))
        return Q, Ξ, x , v, f, t
    end
end

x0[1] = 5.1
Ξ = zz_epi(x0, v0, 6.0, 20.0)
t = first.(Ξ)
x1 = getindex.(getindex.(last.(Ξ),1),1)
x2 = getindex.(getindex.(last.(Ξ),1),4)
plot_result = true
if plot_result
    using GLMakie
    fig = Figure()
    ax = Axis(fig[1,1])
    lines!(ax, t, x1)
    lines!(ax, t, x2)
    vlines!(ax, first.(Ξ), color = (:blue, 0.3))
    hlines!(ax, collect(2:3), color = (:red, 0.3))
    hlines!(ax, collect(4:6), color = (:red, 0.3))
    fig
end