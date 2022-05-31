using ZigZagBoomerang
const ZZB = ZigZagBoomerang
using Test
using SparseArrays
using Statistics
using LinearAlgebra
using ZigZagBoomerang: sep
using Random
using Dictionaries
Random.seed!(1)
d = 40
S = 1.3I + 0.5sprandn(d, d, 0.1)
const Γ = S*S'


@testset "Asynchronous ZigZag" begin
    global Γ
    d = size(Γ, 1)
    κ = 10.0
    ∇ϕ(x, i) = ZigZagBoomerang.idot(Γ, i, x) # sparse computation
    x0 = randn(d)
    u0 = ZZB.stickystate(x0)
    target = ZZB.StructuredTarget(Γ, ∇ϕ)
    barriers = [ZZB.StickyBarriers((0.0, 0.0), (:sticky, :sticky), (κ, κ)) for i in 1:d]
    flow = ZZB.StickyFlow(ZigZag(0.9Γ, x0*0))
    strong = false
    c = 4ones(d)
    adapt = false
    multiplier = 1.5
    T = 2000.0
    G = target.G
    upper_bounds = ZZB.StrongUpperBounds(G, adapt, c, multiplier)
    end_time = ZZB.EndTime(T)
    trace, _, _, acc = @time ZZB.asynchzz(u0, target, flow, upper_bounds, barriers, end_time; progress=true)
    println("acc ", acc.acc/acc.num)

    tm = @elapsed ZZB.asynchzz(u0, target, flow, upper_bounds, barriers, end_time)
    @time al = @allocated ZZB.asynchzz(u0, target, flow, upper_bounds, barriers, end_time)
    @test tm < 1.0 
    dt = 0.5
    global ts2, xs2 = sep(collect(trace))
end