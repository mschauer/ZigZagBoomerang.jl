using DataStructures
using ZigZagBoomerang: LinearQueue, PriorityQueues, rcat

@testset "Queues" begin
    L = LinearQueue(0:2, [3.0, 1.0, 0.5])
    @test L[0] == 3.0
    Q = PriorityQueues(L, PriorityQueue(3=>1.0, 4=>0.6))
    @test peek(Q) == (2, 0.5)
    L[1] = 0.0
    @test peek(Q) == (1, 0.)


    @test keys(Q)[1] == 0:2
    @test rcat(1:10, 11:20) == 1:20

    @test Q[1] == L[1]
    Q[0] = 5.0
    @test Q[0] == L[0] == 5
    @test_throws KeyError Q[6]
    @test Q[4] == 0.6
    Q[3] = 11.0
    @test Q[3] == 11.0
    @test (Q[3] = 13) == 13
end

using Test
@testset "PartialQueue" begin
    vals = [1.0, 0.1, 1.0, 0.1, 0.3]
    G = Graph(Edge.(([1=>2, 2=>3, 3=>4, 4=>5])))

    Q = PartialQueue(G, vals)

    Q[2] = 3
end