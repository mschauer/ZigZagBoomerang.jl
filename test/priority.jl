using DataStructures
using ZigZagBoomerang: LinearQueue, PriorityQueues, rcat
using ZigZagBoomerang, Graphs
using ZigZagBoomerang: PartialQueue, dequeue!, saturate, rkey, div1
using Graphs: Edge
using Random
using Base.Threads

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




    Random.seed!(1)

    # number of regions
    nregions = 2
    # number of coordinates/keys
    d = 10

    # time surface
    times = rand(d)

    # undirected graph of neighbours
    G = Graph(Edge.(([i=>i+1 for i in 1:d-1])))

    # Make a Queue-structure keeping track of local minima
    # (a coordinate is a local minima if it's time is smaller than all neighbours' (via the graph) times)
    Q = PartialQueue(G, times, nregions)

    # key 1:5 is in region 1, keys 6:10 in region 2
    # we can work on different regions in parallel... see below
    @test rkey(Q,5) == 1 
    @test rkey(Q,6) == 2 


    # dequeue! some local minima to work with
    minima = dequeue!(Q) 

    @test peek(Q) == [[],[]] # currently all local minima are removed from the queue to be worked on

    # update/increment local time, (in parallel thanks to the threads makro)
    for r in 1:nregions
        for (i,t) in minima[r]
            println("work with $i, $t on $(Threads.threadid())")
            Q[i] = t + rand()
        end
    end
    
    # internal check
    @test try 
            ZigZagBoomerang.checkqueue(Q, fullcheck=true) == nothing
          catch 
            false 
          end  

    # By now we have a new set of local minimas we can handle in threads 
    @test peek(Q) != [[],[]]
end