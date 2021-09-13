

### statespace 
#  fake program to deal with discontinuities


# task
# n points (some of them are infection times, some of them are notifications and removing times (the two last times don't move))
# n + 1 hitting times ordered in a queue
# random events which flip the velocity of an active coordinate
# distinguish and print all the events hitting a notification/remove/permutation of infected/


T = 30.0
d = 5
N = d*3 # each individual has a notification time, removal time, 
it = rand(d)*T 
notif = it .-log.(rand(d))./1.0
remov = notif .+ 1.0 

x0 = [[it[i] min(notif[i], T) min(remov[i], T)] for i in 1:d]
x0 = collect(Iterators.flatten(x0))

for i in 1:3:d
    if !(x0[i] <= x0[i+1] <= x0[i+2])
        error("$(x0[i]) <= $(x0[i+1]) <= $(x0[i+2])")
    end 
end

θ0 = [(i)%3 == 1 ? rand([-1.0,+1.0]) : 0.0 for i in 1:3*d] # have non-zero velocities only the ifection times


function hitting_neighbors(x1, x2, v1, v2) 
    res = 0.0
    if v1 == v2 || (v1 < v2) || (x1 == x2)
        return Inf
    else
        res = (x1 - x2)/(v2 - v1)
        if res < 0.0
            error("x1 = $(round(x1, digits=2)), x2 = $(round(x2, digits=2)), v1 = $(round(v1, digits=2)), v2 = $(round(v2, digits=2))")
        end
    end
    return res
end

function tag(ii)
    i = (ii)%3 
    if (i == 1) 
        return "infected time" 
    elseif (i == 2) 
        return "notified time" 
    else 
        return "removing time" 
    end
end

function run_coordinates(x0, θ0, T, TT)
    x = deepcopy(x0)
    θ = deepcopy(θ0)
    f = sortperm(x)
    t = 0.0
    while t<TT
        [@assert xi <= T for xi in x]
        lb = θ[f[1]] != -1.0 ? Inf : -x[f[1]]/θ[f[1]]
        ub = (θ[f[end]] != 1.0 || x[f[end]] == T) ? Inf : (T-x[f[end]])/θ[f[end]]
        ht = [hitting_neighbors(x[f[i]], x[f[i+1]], θ[f[i]], θ[f[i+1]]) for i in 1:length(x)-1]
        τ, ii = findmin(ht)
        if lb < min(ub, τ)
            x .= x .+ θ.*lb
            t += lb
            if  x[f[1]] != 0.0
                error("x[f[1]] = $(x[f[1]])")
            end
            println("at time $(t) : $(tag(f[1])) at $(x[f[1]])" )
            θ[f[1]] *= -1
        elseif ub < τ
            x .= x .+ θ.*ub
            t += ub
            if  x[f[end]] != T
                error("x[f[end]] = $(x[f[end]])")
            end
            println("at time $(t) : $(tag(f[end])) at $(x[f[end]])" )
            x[f[end]] = T
            θ[f[end]] *= -1
        else # τ <  min(ub, lb)
            x .= x .+ θ.*τ
            t += τ
            println("at time $(t) : collision between a $(tag(f[ii])) and a $(tag(f[ii+1]))")
            # j1 = (f[ii]+2)÷3 # index
            # t1 = f[ii]%3 #notifi
            # j2 = (f[ii+1]+2)÷3 
            # t2 = f[ii+1]%
            if f[ii]%3 == 1 && f[ii]+1 == f[ii+1] # infected time hits its notification time
                # x[ii] = x[f[ii+1]]
                θ[f[ii]] *= -1
            else
                @assert abs(x[f[ii]] - x[f[ii+1]]) < eps()
                x[f[ii]] == x[f[ii+1]]
                if rand() < 0.5 #sometimes relfect without changing order
                    println("reflect")
                    # x[f[ii]] = x[f[ii+1]] # avoid numerical problems
                    θ[f[ii]] *= -1
                    θ[f[ii+1]] *= -1
                else 
                    println("change order")
                    # x[f[ii]] = x[f[ii+1]] # avoid numerical problems
                    c = f[ii]
                    f[ii] = f[ii+1]
                    f[ii + 1] = c
                end
            end
        end
    end
    return x, θ, f
end
TT = 100.0
x, θ, f = run_coordinates(x0, θ0, T, TT)
y = x[f]
[@assert y[i] <= y[i+1] for i in 1:length(y)-1]







# x = [0.24, 0.35, 0.31, 0.01, 0.49] # round.(rand(d), digits=2)
# f = sortperm(x)

# # bounds
# lb = x[f[1]]
# up = x[f[end]]

# # change the order
# c = f[5]
# f[5] = f[4] 
# f[4] = c
# x[f]



# b = sortperm(f)
# f[b[2]]
# x[f[1]]
# x[b]
# i = 4
# f1, f2 = f[b[i]-1], f[b[i]+1]
# println(i, "(",x[i], ")", " is between ", f1,"(",x[f1],") and ", f2, "(", x[f2],")")