using DataStructures
struct Pressure # pressure
    G0 # weighted directed graph
    γ # reduction after notification
end

struct State1    
    x # pos
    v # vel
    s # stuck
    λ1 # pressure 
    λ2 # slope 
end
State1(x, v, s) = State1(x, v, s, zeros(length(x)), zeros(length(x)))
struct State2
    y # position of obstacles
    t # tag of obstacles
    id  # id of obstacles
end

struct State
    S1::State1
    S2::State2
    f::Vector{Int64}
    b::Vector{Int64}
    M # map (Dictionary) from idexes of S1 to indeces of S2    
    # maybe not needed
end

state(S::State) = S.S1.x, S.S1.v, S.S1.s
ordering(S::State) = S.f, S.b 
lambdas(S::State) = S.S1.λ1, S.S1.λ2 

function lambdas!(i, S::State, c1, c2)
    S.S1.λ1[i] += c1
    if S.S1.λ1[i] < 0.0
        println("Warning: λ1 = $(S.S1.λ1[i]) out of its domain. Setting it to 0")
        S.S1.λ1[i] = 0.0
    end
    S.S1.λ2[i] += c2
    return S
end

obstacles(S::State) = S.S2.y, S.S2.t, S.S2.id
obstacle(i, S::State) = S.S2.y[i], S.S2.t[i], S.S2.id[i]
tag(i, S::State) = S.S2.tag[i]
ind(i, S::State) = S.S2.ind[i]
mapobstacles(i, S::State) = S.M[i]
pressure(P::Pressure, i, j) = P.G0[i,j]
ordered_state(u) = [u.S1.x; u.S2.y][u.f]



function stuckdependentparticles(i, u::State, p::Pressure)
    x,v,s = state(u)
    k = Vector{Int64}()
    for j in 1:N
        if p.G0[i,j] != 0.0 && s[j] == 1
            push!(k, j)
        end
    end
    return k
end

function zz_epi(u0::State, p::Pressure,  clock)
    u = deepcopy(u0)
    x0, v0, s0 = state(u0)
    f0, b0 = ordering(u0) 
    x, v, s = state(u)
    λ1, λ2 = lambdas(u)
    t = 0.0
    Ξ = [(0.0, 0, x0, v0)] 
    Q = PriorityQueue{Int64, Tuple{Float64, Bool}}()
    for i in 1:N
        u = lambdas!(i, u, Lambda(i, u, p), Gamma(i, u, p)) 
    end
    for i in 1:N
        if v[i] == 0.0 && s[i] == false
            continue
        else
            # equeue stickying event or random event or hitting time
            τ, h = next_event(i, t, u, p)
            enqueue!(Q, i => (τ, h))
        end
    end
    stop = false
    count = 0
    while !(stop)
        count += 1
        if count % 10000 == 0
            λ1, λ2 = lambdas(u)
            for i in 1:N
                if abs(λ1[i] - Lambda(i, u, p)) > 1.0e-4
                    println("warning: for particle $i, accumulated λ1 = $(λ1[i]), real value $(Lambda(i, u, p))")
                    λ1[i] = Lambda(i, u, p)
                end
                if abs(λ2[i] - Gamma(i, u, p)) > 1.0e-4
                    println("warning: for particle $i, accumulated λ2 = $(λ2[i]), real value $(Gamma(i, u, p))")
                    λ2[i] = Gamma(i, u, p)
                end
            end
        end
        Ξ, Q, t, u, p, stop = inner_loop!(Ξ, Q, t, u, p, clock, stop)
    end
    # error("")
    return Ξ, u0, u, p, count
end


function inner_loop!(Ξ, Q::PriorityQueue{Int64, Tuple{Float64, Bool}}, t::Float64, u::State, p::Pressure, clock::Float64, stop::Bool)
     # dequeue firsy time
    i0, (τ0, h0) = peek(Q)  
    # println("time $(τ0), hitting time $(h0)")
    x, v, s = state(u)
    if clock < τ0
        x .+= v.*(clock - t)
        t = clock
        push!(Ξ, event(i0, t, x, v))
        stop = true
    else
        x .+= v.*(τ0 - t)
        t = τ0
        if h0
            u, Q = devent!(Q, t, i0, u, p)
            # update priority queue
        else
            u, Q = revent!(Q, t, i0, u, p)
            # update priority queue
        end
        push!(Ξ, event(i0, t, x, v))
    end
    return Ξ, Q, t, u, p, stop
end

function event(i, t, x, θ)
    t, i, deepcopy(x), deepcopy(θ)
end
    
function next_event(i::Int64, t::Float64, u::State, p::Pressure)
    τ1 = reflect(i, u, p)
    τ2 = hit(i, u, p)
    if τ1 < τ2  
        return (t + τ1, false) 
    else  
        return (t + τ2, true) 
    end
end



function reflect(i::Int64, u::State, p::Pressure)
    x,v,s = state(u)
    λ1, λ2 = lambdas(u)
    if v[i] == 0.0 
        if s[i] == 0
            return Inf
        else # s[i] == 1
            # c = λratej(u, i, ℬ, G)
            # println("Agent $(ind[i]). Status: Stuck. Pressure: $c")
            # @assert c >= 0.0
            return -log(rand())/λ1[i] # unsticking event
        end
    else
        # TODO
        # c1 = slope_j(u, i, ℬ, G)
        # if ind[i] == 4
            # 
            # println("(x,v) = ($(x[i]), $(v[i])), λ_τ₄(τ) = $(max(0.0, v[i]*(c1 - β)))")
        # end
        c = max(0.0, v[i]*(λ2[i] - β))
        return -log(rand())/c # random reflection
    end
end


function Lambda(j::Int64, u::State, p::Pressure)
    res = 0.0
    for i in 1:N
        res += Lambda(i, j, u, p)
    end
    return res
end

function Lambda(i, j, u, p)
    x, v, s = state(u)
    f, b = ordering(u)
    if i == j
        return 0.0
    else
        ii = mapobstacles(i, u)
        if length(ii) == 0 # must be infection
            if b[j] < b[i] || s[i] == 1
                return 0.0
            else
                return pressure(p, i, j)
            end
        elseif length(ii) == 1 # either inf or not
            if b[j] < b[i] || s[i] == 1
                return 0.0
            elseif b[i] < b[j] < b[ii[1]]
                return pressure(p, i, j)
            else
                return pressure(p, i, j)*p.γ
            end
        else 
            # @assert length(ii) == 2
            if b[j] < b[i] || s[i] == 1
                return 0.0
            elseif b[i] < b[j] < b[ii[1]]
                return pressure(p, i, j)
            elseif b[ii[1]] < b[j] < b[ii[2]]
                return pressure(p, i, j)*p.γ
            else
                return 0.0
            end
        end
    end
    error("")
end

function Gamma(j::Int64, u::State, p::Pressure)
    res = 0.0
    for i in 1:N
        res += Gamma(i, j, u, p)
    end
    return res
end

function Gamma(i, j, u, p)
    x, v, s = state(u)
    f, b = ordering(u)
    @assert i <= N
    @assert j <= N
    if i == j
        return 0.0
    else
        ii = mapobstacles(i, u)
        if length(ii) == 0 # must be infection
            if b[j] < b[i] || s[i] == 1
                return -pressure(p, j, i)
            else
                return pressure(p, i, j)
            end
        elseif length(ii) == 1 # either inf or not
            if b[j] < b[i] || s[i] == 1
                return -pressure(p, j, i)
            elseif b[i] < b[j] < b[ii[1]]
                return pressure(p, i, j)
            else
                return pressure(p, i, j)*p.γ
            end
        else 
            @assert length(ii) == 2
            if b[j] < b[i] || s[i] == 1
                return -pressure(p, j, i)
            elseif b[i] < b[j] < b[ii[1]]
                return pressure(p, i, j)
            elseif b[ii[1]] < b[j] < b[ii[2]]
                return pressure(p, i, j)*p.γ
            else
                return 0.0
            end
        end
    end
    error("")
end


# move particles and sort them manually
function hit(i::Int64, u::State, p::Pressure)
    x ,v, s = state(u)
    y, tag, id = obstacles(u)
    f, b = ordering(u)
    if s[i] == 1.0 || v[i] == 0.0
        return Inf
    else
        if v[i] > 0.0 
            j = f[b[i] + 1] 
        else 
            j = f[b[i] - 1] 
        end
        # obstacles have no velocity
        if j > N
            j0 = j - N
            return abs((x[i] - y[j0])/v[i])
        end
        # make it doable for BPS 
        if v[i] == v[j]
            return Inf
        elseif v[i] < 0 && v[j] > 0
            return Inf
        else
            # error("if i and j are moving particles, then compute Inf for one of them")
            # println("vi = $(v[i]), xi = $(x[i]), vj = $(v[j]), xj = $(x[j])")
            return  (x[j] - x[i])/(v[i] - v[j])
        end
    end
end


function devent!(Q, t::Float64, i::Int64, u::State, p::Pressure) #swap and bounce
    x, v, s = state(u)
    f, b = ordering(u)
    λ1, λ2 = lambdas(u)
    if v[i] == 0.0
        error("error: a particle with velcoty 0 cannot hit a particle")
    elseif v[i] > 0.0
        bi, bj = b[i], b[i] + 1
    else
        bi, bj = b[i], b[i] - 1
    end
    j = f[bj] 
    if j > N # it is an obstacles
        j0 = j - N 
        y0, tag0, id0 = obstacle(j0, u)
        x[i] = y0
        if id0 == i
            # boundary
            v[i] *= -1
            # do local update
            k = i, f[b[i] - 1] 
        elseif tag0 == 0
            if y0 == 0.0
                v[i] *= -1
                k = i, f[b[i] + 1]
            else # boundary at T, Stick!
                @assert T == y0
                v[i] = 0.0 
                s[i] = 1
                # there are no stuck particles yet
                k = i, f[b[i] - 1] # local update
            end      
        else # τj is either notification or removal
            δ1, δ2 = deltalambdasstate2(u, i, j0, p) 
            prob = 1 + δ1/λ1[i]
            # if (prob <= 0.0)  println("probability equal to $(prob)") end
            if rand() > prob #true -> do not cross
                v[i] *= -1.0
                k = v[i] > 0.0 ? (i, f[b[i] + 1]) : (i, f[b[i] - 1])
            else # cross
                u = lambdas!(i, u, δ1, δ2)
                f[bi], f[bj] = f[bj], f[bi]
                b[i], b[j] = b[j], b[i]
                # if abs((u.S1.λ1[i] - Lambda(i,u,p)) > eps()) error("λ1 = $(u.S1.λ1[i]) != $(Lambda(i,u,p))") end
                # if abs((u.S1.λ2[i] - Gamma(i,u,p)) > eps()) error("λ2 = $(u.S1.λ2[i]) != $(Gamma(i,u,p))") end
                if b[i] < b[j]
                    k = f[b[i]-1], i, j, f[b[j] + 1] # local update
                else
                    k = f[b[j]-1], j, i, f[b[i] + 1] # local update
                end    
            end      
        end
    else # collision between two agents
        x[i] = x[j]
        if s[j] == 1 # STICK!
            v[i] = 0.0 
            s[i] = 1
            #### TODO LOCAL UPDATE OF STUCK particles
            ss = stuckdependentparticles(i, u, p)
            k = i, f[b[i] - 1], ss...
            for j1 in ss
                u.s1.λ1[j1] -= p.G0[i,j1] 
            end
        else # it is a active infection time
            # cb = log(λ1[i]) + log(λ1[j])
            δλ1i, δλ1j, δλ2i, δλ2j = deltalambdasstate1(u, i, j, p)
            # error("todo")
            # ca = log(Lambda(i, u, p)) + log(Lambda(j, u, p))
            # prob = exp((log(λ1[i] + δi1) + log(λ1[j] + δj1)) - cb)
            # prob = (λ1[i] + δλ1i)*(λ1[j] + δλ1j)/((λ1[i])*(λ1[j]))
            prob = 1 + δλ1i/λ1[i] + δλ1j/λ1[j] + δλ1i*δλ1j/(λ1[i]*λ1[j]) 
            # (λ1[i] + δλ1i)*(λ1[j] + δλ1j)/((λ1[i])*(λ1[j]))
            if rand() > prob #true -> do not cross
                # f[bi], f[bj] = f[bj], f[bi]
                # b[i], b[j] = b[j], b[i]
                # f[b[i]-1], i, j, f[b[j] + 1] 
                v[i] *= -1.0
                v[j] *= -1.0
            else # cross
                u = lambdas!(i, u, δλ1i, δλ2i)
                u = lambdas!(j, u, δλ1j, δλ2j)
                f[bi], f[bj] = f[bj], f[bi]
                b[i], b[j] = b[j], b[i]
            end
            if b[i] < b[j]
                k = f[b[i]-1], i, j, f[b[j] + 1] 
            else
                k = f[b[j]-1], j, i, f[b[i] + 1]
            end  
        end
    end
    Q = update_queue!(Q, k, t, u, p)
    return u, Q 
end

function deltalambdasstate2(u::State, i::Int64, k::Int64, p::Pressure) 
    x, v, s = state(u)
    y, tag, id = obstacle(k, u)
    if v[i] > 0.0
        if tag == 2 # τi -> τ⋆j 
            return -(1-p.γ)*p.G0[id, i], -(1-p.γ)*p.G0[id, i] 
        elseif tag == 3 #τ⋆j - τi -> τ∘j 
            return -p.γ*p.G0[id, i], -p.γ*p.G0[id, i]
        else 
            error("tag = $(tag) cannot exists")
        end
    else # v[i] < 0.0
        if tag == 2  # τ⋆j <- τi - τ∘j  
             return (1-p.γ)*p.G0[id, i], (1-p.γ)*p.G0[id, i]
        elseif tag == 3  # τ∘j <- τi 
            return +p.γ*p.G0[id, i], +p.γ*p.G0[id, i]
        else
            error("tag = $(tag) cannot exists")
        end
    end
    error("why am I here")
end

function deltalambdasstate1(u::State, i::Int64, j::Int64, p::Pressure) 
    x, v, s = state(u)
    if v[i] < 0.0 # τj -> <- τi 
        return -p.G0[j,i], p.G0[i,j], -(p.G0[i,j] + p.G0[j,i]), p.G0[i,j] + p.G0[j,i] 
    else
        # τi -> <- τj
        return p.G0[j,i], -p.G0[i,j], (p.G0[i,j] + p.G0[j,i]), -(p.G0[i,j] + p.G0[j,i])
    end
end

function revent!(Q, t::Float64, i::Int64, u::State, p::Pressure)
    x, v, s = state(u)
    f, b = ordering(u)
    if v[i] == 0.0 
        if s[i] == 0
            error("how come?")
        else    # unstick
            v[i] = -1.0
            s[i] = 0
            ss = stuckdependentparticles(i, u, p)
            k = i, f[b[i] - 1], ss...
            # todo use graph G0 and s to determine particle affected
            for j1 in ss
                u.s1.λ1[j1] += p.G0[i,j1] 
            end
        end
    else
        v[i] *= -1
        k = f[b[i] - 1], i, f[b[i] + 1]
    end
    Q = update_queue!(Q, k, t, u, p)
    return u, Q 
end

function update_queue!(Q::PriorityQueue{Int64, Tuple{Float64, Bool}}, S, t::Float64, u::State, p::Pressure)
    x, v, s = state(u)
    for i in S
        if i > N
            continue
        else
            τ, h = next_event(i, t, u, p)
            Q[i] = (τ, h)
        end
    end
    Q
end