using ZigZagBoomerang
const ZZB = ZigZagBoomerang
using Test
using SparseArrays
using Revise
using SparseArrays
using Statistics
using LinearAlgebra
using ZigZagBoomerang: sep
d = 20
S = 1.3I + 0.5sprandn(d, d, 0.1)
const Γ = S*S'


@testset "Sticky SZigZag" begin
    global Γ
    d = size(Γ, 1)   
    ∇ϕ(x, i, Γ) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation   
    t0 = 0.0
    x0 = rand(d)
    κ = 10.0*ones(d) # dont stop, actually
    θ0 = rand([-1.0,-0.5,0.5,1.0], d)


    c = .8*[norm(Γ[:, i], 2) for i in 1:d]

    Z = ZigZag(0.9Γ, x0*0)
    T = 2000.0
   
    trace, _, acc = @time sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ, Γ)
    tm = @elapsed @time sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ, Γ)
    al = @allocated @time sspdmp(∇ϕ, t0, x0, θ0, T, c, Z, κ, Γ)

    @test tm < 0.2
    @test al < 50_000_000

    dt = 0.5
    global ts, xs = sep(collect(discretize(trace, dt)))

    @test mean(abs.(mean(xs))) < d/sqrt(T)

end

@testset "New Sticky ZigZag" begin
    global Γ
    d = size(Γ, 1)
    κ = 10.0

    ∇ϕ(x, i) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation
    x0 = randn(d)
    u0 = ZZB.stickystate(x0)
    target = ZZB.StructuredTarget(Γ, ∇ϕ)
    barriers = [ZZB.StickyBarriers((0.0,0.0),(:sticky, :sticky),(κ, κ)) for i in 1:d]
    flow = ZZB.StickyFlow()
    strong = false

    c = .8*[norm(Γ[:, i], 2) for i in 1:d]
    adapt = false
    c = 1.0*ones(d)
    factor = 1.5
    T = 2000.0
    upper_bounds = ZZB.StickyUpperBounds(target.G, 0.9Γ, strong, adapt, c, factor)
    end_time = ZZB.EndTime(T)
    ZZB.stickyzz(u0, target, flow, upper_bounds, barriers, end_time)
end
