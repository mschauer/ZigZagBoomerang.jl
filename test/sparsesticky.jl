using ZigZagBoomerang
const ZZB = ZigZagBoomerang
using Test
using SparseArrays
using Revise
using SparseArrays
using Statistics
using LinearAlgebra
using ZigZagBoomerang: sep
using ProfileView
using Random
using Dictionaries
Random.seed!(1)


@testset "New Sparse Sticky ZigZag" begin
    d = 10_000_000
    

    global xs = Float64[1, 2, 1, 0, 0, 1]
    global u = dictionary(10-2i=> (x,x,x) for (i,x) in  enumerate(xs) if x ≠ 0)
    global Γ = sparse(Tridiagonal(-ones(d-1), 2ones(d), -ones(d-1))) 
    Γ[1,1] = Γ[end,end] = 1
    Γ += 0.1I
    

    d = size(Γ, 1)
    κ = 2000/d
    global ∇ϕ(u, i) = ZigZagBoomerang.idot(Γ, i, u) # sparse computation
    x0 = sprandn(d, 0.0)
    global u0 = ZZB.sparsestickystate(x0)
    @test nnz(u0) == nnz(x0)
    #@test ZZB.stickystate(u0)[2] == x0

    for i in 1:3
        @test u0[i][2] == x0[i]
        has, ι = ZZB.gettoken(u0, 1)
    end
   

    barrier = ZZB.StickyBarriers(0.0, :reversible, κ)

    target = ZZB.StructuredTarget(Γ, ∇ϕ)
    flow = ZZB.StickyFlow(ZigZag(I(d), nothing))

    c = 2.5
    adapt = false
    multiplier = 1.5
    T = 500.0

    upper_bounds = ZZB.SparseStickyUpperBounds(c; adapt=adapt, multiplier= multiplier)
    
    end_time = ZZB.EndTime(T)
    global trace, _, uT, acc = @time ZZB.sparsestickyzz(u0, target, flow, upper_bounds, barrier, end_time; progress=true)
    println("acc ", acc.acc/acc.num)
    k = min(trace.events[1][2], d-100+1)
    global ts1, xs1 = sep(collect(ZZB.subtrace(trace, k:k+100-1)))


end
