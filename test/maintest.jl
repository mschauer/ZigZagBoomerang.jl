
@testset "main" begin
# Local ZigZag
Random.seed!(2)
using SparseArrays
d = 8
S = 1.3I + 0.5sprandn(d, d, 0.1)
Γ = S*S'

∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation
∇ϕmoving(t, x, θ, i, t′, F, Γ) = ZigZagBoomerang.idot_moving!(Γ, i, t, x, θ, t′, F) # sparse computation


@testset "ZigZag" begin

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0, 1.0], d)


    c = .7*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(0.9Γ, x0*0)
    T = 1000.0

    trace, _, acc = @time pdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
    @show acc[1]/acc[2]
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))


    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end


@testset "SZigZag" begin

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0,-0.5,0.5,1.0], d)


    c = .7*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(0.9Γ, x0*0)
    T = 1000.0

    trace, _, acc = @time spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
    @show acc[1]/acc[2]
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))
    @testset "subtrace" begin
        J = 1:2:d
        ts2, xs2 = sep(collect(discretize(subtrace(trace, J), dt)))
        @test ts2 ≈ ts[eachindex(ts2)]
        @test (xs2) ≈ (getindex.(xs[eachindex(ts2)], Ref(J)))
    end
    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end


@testset "SZigZagSelfMoving" begin

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0,-0.5,0.5,1.0], d)


    c = .8*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(0.9Γ, x0*0)
    T = 1000.0

    trace, _, acc = @time spdmp(∇ϕmoving, t0, x0, θ0, T, c, Z, SelfMoving(), Γ)
    @show acc[1]/acc[2]
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end





@testset "FactBoomerang" begin

    t0 = 0.0
    x0 = 0.2rand(d)

    #Γ0 = sparse(I, d, d)
    c = [norm(Γ[:, i], 2) for i in 1:d]
    Γ0 = copy(Γ)
    for i in 1:d
        #Γ0[d,d] = 1
    end
    Z = FactBoomerang(0.85Γ0, x0*0, 0.3)
    θ0 = sqrt(Diagonal(Z.Γ))\randn(d)

    T = 3000.0

    trace, _, acc = @time pdmp(∇ϕ, t0, x0, θ0, T, c, Z, Z.Γ)
    @show acc[1]/acc[2]
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))

    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 4.5/sqrt(T)
end

@testset "SFactBoomerang" begin

    t0 = 0.0
    x0 = rand(d)

    #Γ0 = sparse(I, d, d)
    c = [norm(Γ[:, i], 2) for i in 1:d]
    Γ0 = copy(Γ)
    for i in 1:d
        #Γ0[d,d] = 1
    end
    Z = FactBoomerang(1.2Γ0, x0*0, 0.3)
    θ0 = sqrt(Diagonal(Z.Γ))\randn(d)
    T = 3000.0

    trace, _, acc = @time spdmp(∇ϕ, t0, x0, θ0, T, c, Z, Z.Γ)
    @show acc[1]/acc[2]
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))


    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 4/sqrt(T)
end

@testset "Boomerang" begin
    t0 = 0.0
    θ0 = randn(d)
    x0 = randn(d)
    c = 16.0
    Γ0 = copy(Γ)
    B = Boomerang(Γ0, x0*0, 0.5)
    ∇ϕ!(y, x) = mul!(y, Γ, x)
    T = 3000.0
    trace, _, acc = @time pdmp(∇ϕ!, t0, x0, θ0, T, c, B)
    @show acc[1]/acc[2]
    dt = 0.1
    ts, xs = sep(collect(discretize(trace, dt)))
    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test_broken mean(abs.(cov(xs) - inv(Matrix(Γ0)))) < 2.5/sqrt(T)
end

@testset "Bouncy Particle Sampler" begin
    t0 = 0.0
    θ0 = randn(d)
    x0 = randn(d)
    c = 1.1
    Γ0 = copy(Γ)
    B = BouncyParticle(Γ0, x0*0, 0.5)
    ∇ϕ!(y, x) = mul!(y, Γ, x)
    T = 300.0
    trace, _, acc, more = @time pdmp(∇ϕ!, t0, x0, θ0, T, c, B, progress=true)
    @show more
    @show acc[1]/acc[2]
    dt = 0.1
    ts, xs = sep(collect(discretize(trace, dt)))
    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ0)))) < 2/sqrt(T)
end

@testset "Bouncy Particle Sampler (arbitrary mass matrix)" begin
    Random.seed!(2)
    t0 = 0.0
    θ0 = randn(d)
    x0 = randn(d)
    M = UpperTriangular(I + 0.4randn(d,d)')
    c = 20.0
    B = BouncyParticle(missing, missing, # ignored
        1.0, # momentum refreshment rate 
        0.9, # momentum correlation / only gradually change momentum in refreshment/momentum update
        ZigZagBoomerang.InvChol(M), # metric
        missing
    ) 

    ∇ϕ!(y, t, x, args...) = mul!(y, Γ, x)
    dϕ(t, x, v, args...) =  dot(v, Γ, x), dot(v, Γ, v)
    n = 800
    trace, _, acc, more = @time pdmp(
        dϕ, # return first two directional derivatives of negative target log-likelihood in direction v
        ∇ϕ!, # return gradient of negative target log-likelihood
        t0, x0, θ0, # initial state and duration
        n, # number of samples
        ZigZagBoomerang.LocalBound(c), # use Hessian information 
        B; # sampler
        adapt=false, # adapt bound c
        progress=true, # show progress bar
    )
    @show more
    @show acc[1]/acc[2]
    ts, xs = sep(trace)
    @show length(ts)
    @test mean(abs.(mean(xs))) < 3/sqrt(length(ts))
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 3/sqrt(length(ts))
end

@testset "Bouncy Particle Sampler (arbitrary mass matrix 2)" begin
    Random.seed!(2)
    t0 = 0.0
    θ0 = randn(d)
    x0 = randn(d)
    M = LowerTriangular(I + 0.4randn(d,d))
    c = 20.0
    B = BouncyParticle(missing, missing, # ignored
        1.0, # momentum refreshment rate 
        0.9, # momentum correlation / only gradually change momentum in refreshment/momentum update
        missing, # metric
        M
    ) 

    ∇ϕ!(y, t, x, args...) = mul!(y, Γ, x)
    dϕ(t, x, v, args...) =  dot(v, Γ, x), dot(v, Γ, v)
    n = 800
    trace, _, acc, more = @time pdmp(
        dϕ, # return first two directional derivatives of negative target log-likelihood in direction v
        ∇ϕ!, # return gradient of negative target log-likelihood
        t0, x0, θ0, # initial state and duration
        n, # number of samples
        ZigZagBoomerang.LocalBound(c), # use Hessian information 
        B; # sampler
        adapt=false, # adapt bound c
        progress=true, # show progress bar
    )
    @show more
    @show acc[1]/acc[2]
    ts, xs = sep(trace)
    @show length(ts)
    @test mean(abs.(mean(xs))) < 3/sqrt(length(ts))
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 3/sqrt(length(ts))
end

@testset "Bouncy Particle Sampler (adapted mass matrix)" begin
    Random.seed!(2)
    t0 = 0.0
    θ0 = randn(d)
    x0 = randn(d)
    M = ZigZagBoomerang.PDMats.PDiagMat(ones(d))
    c = 1.1
    B = BouncyParticle(missing, missing, # ignored
        1.0, # momentum refreshment rate 
        0.9, # momentum correlation / only gradually change momentum in refreshment/momentum update
        M, # metric
        missing # cholesky of momentum precision
    ) 

    ∇ϕ!(y, t, x, args...) = mul!(y, Γ, x)
    dϕ(t, x, v, args...) =  dot(v, Γ, x), dot(v, Γ, v)
    n = 800
    trace, _, acc, more = @time pdmp(
        dϕ, # return first two directional derivatives of negative target log-likelihood in direction v
        ∇ϕ!, # return gradient of negative target log-likelihood
        t0, x0, θ0, # initial state and duration
        n, # number of samples
        ZigZagBoomerang.LocalBound(c), # use Hessian information 
        B; # sampler
        adapt=false, # adapt bound c
        adapt_mass=true,
        progress=true, # show progress bar
    )
    @show more
    @show acc[1]/acc[2]
    ts, xs = sep(trace)
    @show length(ts)
    @test mean(abs.(mean(xs))) < 2/sqrt(length(ts))
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2/sqrt(length(ts))
end

@testset "ZigZag (independent)" begin

    t0 = 0.0
    x0 = rand(d)
    θ0 = rand([-1.0, 1.0], d)


    c = 10.0*[norm(Γ[:, i], 2) for i in 1:d]
    Γ0 = sparse(I, d, d)
    Z = ZigZag(Γ0, x0*0)

    T = 1000.0

    trace, _, acc = @time pdmp(∇ϕ, t0, x0, θ0, T, c, Z, Γ)
    @show acc[1]/acc[2]
    dt = 0.5
    ts, xs = sep(collect(discretize(trace, dt)))


    @test mean(abs.(mean(xs))) < 2/sqrt(T)
    @test mean(abs.(cov(xs) - inv(Matrix(Γ)))) < 2.5/sqrt(T)
end

@testset "FactBoomerang1" begin
    ϕ(x) = [cos(π*x[1]) + x[1]^2/2] # not needed
    # gradient of ϕ(x)
    ∇ϕ(x) = [-π*sin(π*x[1]) + x[1]]
    ∇ϕ(x, i) = -π*sin(π*x[1]) + x[1] # (REPLACE IT WITH AUTOMATIC DIFFERENTIATION)
    c = [3.5π]
    λref = 1.5
    n = 1
    x0 = randn(n)
    θ0 = randn(n)
    t0 = 0.0
    T = 1000.0
    Γ = sparse(Matrix(1.0I, n, n))
    B = FactBoomerang(Γ, x0*0, λref)
    trace, _,  acc = pdmp(∇ϕ, t0, x0, θ0, T, c, B)
    @show acc[1]/acc[2]
    m = mean(last.(collect(trace)))
    dt = 0.1
    ts, xs = sep(collect(discretize(trace, dt)))
    @test mean(xs)[1] < 5/sqrt(T)
end
end