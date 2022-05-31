var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ZigZagBoomerang","category":"page"},{"location":"#ZigZagBoomerang","page":"Home","title":"ZigZagBoomerang","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Back to repository: https://github.com/mschauer/ZigZagBoomerang.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ZigZagBoomerang]","category":"page"},{"location":"#ZigZagBoomerang.Boomerang","page":"Home","title":"ZigZagBoomerang.Boomerang","text":"Boomerang(μ, λ) <: ContinuousDynamics\n\nDynamics preserving the N(μ, Σ) measure (Boomerang) with refreshment time λ\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.Boomerang1d","page":"Home","title":"ZigZagBoomerang.Boomerang1d","text":"Boomerang1d(Σ, μ, λ) <: ContinuousDynamics\n\n1-d toy boomerang samper. Dynamics preserving the N(μ, Σ) measure with refreshment time λ.\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.BouncyParticle","page":"Home","title":"ZigZagBoomerang.BouncyParticle","text":"BouncyParticle(λ) <: ContinuousDynamics\n\nInput: argument Γ, a sparse precision matrix approximating target precision. Bouncy particle sampler,  λ is the refreshment rate, which has to be strictly positive.\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.ContinuousDynamics","page":"Home","title":"ZigZagBoomerang.ContinuousDynamics","text":"ContinuousDynamics\n\nAbstract type for the deterministic dynamics of PDMPs\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.ExtendedForm","page":"Home","title":"ZigZagBoomerang.ExtendedForm","text":"ExtendedForm()\n\nIndicates as args[1] that ∇ϕ  depends on the extended arguments\n\n∇ϕ(t, x, θ, i, t′, Z, args...)\n\ninstead of \n\n∇ϕ(x, i, args...)\n\nCan be used to implement ∇ϕ depending on random coefficients.\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.FactBoomerang","page":"Home","title":"ZigZagBoomerang.FactBoomerang","text":"FactBoomerang(Γ, μ, λ) <: ContinuousDynamics\n\nFactorized Boomerang dynamics preserving the N(μ, inv(Diagonal(Γ))) measure with refreshment time λ.\n\nExploits the conditional independence structure of the target measure, in form the argument Γ, a sparse precision matrix approximating target precision. μ is the approximate target mean.\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.FactTrace","page":"Home","title":"ZigZagBoomerang.FactTrace","text":"FactTrace\n\nSee Trace.\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.PDMPTrace","page":"Home","title":"ZigZagBoomerang.PDMPTrace","text":"FactTrace\n\nSee Trace.\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.SelfMoving","page":"Home","title":"ZigZagBoomerang.SelfMoving","text":"SelfMoving()\n\nIndicates as args[1] that ∇ϕ depends only on few coeffients and takes responsibility to call smove_forward!.\n\nReplaced by ExtendedForm.\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.Trace-Union{Tuple{T}, Tuple{T, Any, Any, Union{FactBoomerang, ZigZagBoomerang.JointFlow, ZigZag}}} where T","page":"Home","title":"ZigZagBoomerang.Trace","text":"Trace(t0::T, x0, θ0, F::Union{ZigZag,FactBoomerang})\n\nTrace object for exact trajectory of pdmp samplers. Returns an iterable FactTrace object. Note that iteration iterates pairs t => x where the vector x is modified inplace, so copies have to be made if the x is to be saved. collect applied to a trace object automatically copies x. discretize returns a discretized version.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.ZigZag","page":"Home","title":"ZigZagBoomerang.ZigZag","text":"struct ZigZag(Γ, μ) <: ContinuousDynamics\n\nLocal ZigZag sampler which exploits any independence structure of the target measure, in form the argument Γ, a sparse precision matrix approximating target precision. μ is the approximate target mean.\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.ZigZag1d","page":"Home","title":"ZigZagBoomerang.ZigZag1d","text":"ZigZag1d <: ContinuousDynamics\n\n1-d toy ZigZag sampler, dynamics preserving the Lebesgue measure.\n\n\n\n\n\n","category":"type"},{"location":"#ZigZagBoomerang.ab-Tuple{Any, Any, Any, Any, Any, ZigZag, Vararg{Any}}","page":"Home","title":"ZigZagBoomerang.ab","text":"ab(G, i, x, θ, c, Flow)\n\nReturns the constant term a and linear term b when computing the Poisson times from the upper upper bounding rates λᵢ(t) = max(a + b*t)^2. The factors a and b can be function of the current position x, velocity θ, tuning parameter c and the Graph G\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.conditional_trace-Tuple{Any, Any}","page":"Home","title":"ZigZagBoomerang.conditional_trace","text":"conditional_trace(trace::Trace, (p, n))\n\nConditionalTrace trace on hyperplane H through p with norml n. Returns iterable object iterating pairs t => x such that x ∈ H.\n\nIteration changes the vector x inplace, collect creates necessary copies.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.discretize-Tuple{Any, Any}","page":"Home","title":"ZigZagBoomerang.discretize","text":"discretize(trace::Trace, dt)\n\nDiscretize trace with step-size dt. Returns iterable object iterating pairs t => x.\n\nIteration changes the vector x inplace, collect creates necessary copies.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.discretize-Tuple{Vector, Union{ZigZag1d, Boomerang1d}, Any}","page":"Home","title":"ZigZagBoomerang.discretize","text":"discretize(x::Vector, Flow::Union{ZigZag1d, Boomerang1d}, dt)\n\nTransform the output of the algorithm (a skeleton of points) to a trajectory. Simple 1-d version.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.freezing_time-Tuple{Any, Any, Union{BouncyParticle, ZigZag}}","page":"Home","title":"ZigZagBoomerang.freezing_time","text":"τ = freezing_time(x, θ)\n\ncomputes the hitting time of a 1d particle with constant velocity θ to hit 0 given the position x\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.idot-Tuple{Any, Any, Any}","page":"Home","title":"ZigZagBoomerang.idot","text":"idot(A, j, x) = dot(A[:, j], x)\n\nCompute column-vector dot product exploiting sparsity of A.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.idot_moving!-Tuple{SparseArrays.SparseMatrixCSC, Any, Any, Any, Any, Any, Any}","page":"Home","title":"ZigZagBoomerang.idot_moving!","text":"idot_moving!(A::SparseMatrixCSC, j, t, x, θ, t′, F)\n\nCompute column-vector dot product exploiting sparsity of A. Move all coordinates needed to their position at time t′\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.move_forward!-Tuple{Any, Any, Any, Any, Union{Boomerang, FactBoomerang}}","page":"Home","title":"ZigZagBoomerang.move_forward!","text":"move_forward!(τ, t, x, θ, B::Boomerang)\n\nUpdates the position x, velocity θ and time t of the process after a time step equal to τ according to the deterministic dynamics of the Boomerang sampler which are the Hamiltonian dynamics preserving the Gaussian measure: : xt = μ +(x0 − μ)cos(t) + v_0sin(t), vt = −(x0 − μ)sin(t) + v_0cos(t) x: current location, θ: current velocity, t: current time.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.move_forward!-Tuple{Any, Any, Any, Any, Union{BouncyParticle, ZigZag}}","page":"Home","title":"ZigZagBoomerang.move_forward!","text":"move_forward!(τ, t, x, θ, Z::Union{BouncyParticle, ZigZag})\n\nUpdates the position x, velocity θ and time t of the process after a time step equal to τ according to the deterministic dynamics of the Bouncy particle sampler (BouncyParticle) and ZigZag: (x(τ), θ(τ)) = (x(0) + θ(0)*t, θ(0)). x: current location, θ: current velocity, t: current time,\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.move_forward-Tuple{Any, Any, Any, Any, Boomerang1d}","page":"Home","title":"ZigZagBoomerang.move_forward","text":"move_forward(τ, t, x, θ, B::Boomerang1d)\n\nUpdates the position x, velocity θ and time t of the process after a time step equal to τ according to the deterministic dynamics of the Boomerang1d sampler: xt = μ +(x0 − μ)cos(t) + v_0sin(t), vt = −(x0 − μ)sin(t) + v_0cos(t) x: current location, θ: current velocity, t: current time.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.move_forward-Tuple{Any, Any, Any, Any, ZigZag1d}","page":"Home","title":"ZigZagBoomerang.move_forward","text":"move_forward(τ, t, x, θ, ::ZigZag1d)\n\nUpdates the position x, velocity θ and time t of the process after a time step equal to τ according to the deterministic dynamics of the ZigZag1d sampler: (x(τ), θ(τ)) = (x(0) + θ(0)*t, θ(0)). x: current location, θ: current velocity, t: current time,\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.neighbours-Tuple{Vector{<:Pair}, Any}","page":"Home","title":"ZigZagBoomerang.neighbours","text":"neighbours(G::Vector{<:Pair}, i) = G[i].second\n\nReturn extended neighbourhood of i including i. G: graphs of neightbourhoods\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.normsq-Tuple{Real}","page":"Home","title":"ZigZagBoomerang.normsq","text":"normsq(x)\n\nSquared 2-norm.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.pdmp-Tuple{Any, Any, Any, Any, Any, Any, Union{FactBoomerang, ZigZagBoomerang.JointFlow, ZigZag}, Vararg{Any}}","page":"Home","title":"ZigZagBoomerang.pdmp","text":"pdmp(∇ϕ, t0, x0, θ0, T, c, F::Union{ZigZag, FactBoomerang}, args..., args) = Ξ, (t, x, θ), (acc, num), c\n\nOuter loop of the factorised samplers, the Factorised Boomerang algorithm and the Zig-Zag sampler. Inputs are a function ∇ϕ giving ith element of gradient of negative log target density ∇ϕ(x, i, args...), starting time and position t0, x0, velocities θ0, and tuning vector c for rejection bounds and final clock T.\n\nThe process moves to time T with invariant mesure μ(dx) ∝ exp(-ϕ(x))dx and outputs a collection of reflection points which, together with the initial triple t, x θ are sufficient for reconstructuing continuously the continuous path. It returns a FactTrace (see Trace) object Ξ, which can be collected into pairs t => x of times and locations and discretized with discretize. Also returns the number of total and accepted Poisson events and updated bounds c (in case of adapt==true the bounds are multiplied by factor if they turn out to be too small.)\n\nThis version does not assume that ∇ϕ has sparse conditional dependencies, see spdmp.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.pdmp-Tuple{Any, Any, Any, Any, Any, ZigZagBoomerang.Bound, Union{Boomerang, BouncyParticle}, Vararg{Any}}","page":"Home","title":"ZigZagBoomerang.pdmp","text":"pdmp(∇ϕ!, t0, x0, θ0, T, c::Bound, Flow::Union{BouncyParticle, Boomerang}; adapt=false, factor=2.0)\n\nRun a Bouncy particle sampler (BouncyParticle) or Boomerang sampler from time, location and velocity t0, x0, θ0 until time T. ∇ϕ!(y, x) writes the gradient of the potential (neg. log density) into y. c is a tuning parameter for the upper bound of the Poisson rate. If adapt = false, c = c*factor is tried, otherwise an error is thrown.\n\nIt returns a PDMPTrace (see Trace) object Ξ, which can be collected into pairs t => x of times and locations and discretized with discretize. Also returns the number of total and accepted Poisson events and updated bounds c (in case of adapt==true the bounds are multiplied by factor if they turn out to be too small.)\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.pdmp-Tuple{Any, Any, Any, Any, Any, ZigZagBoomerang.ContinuousDynamics}","page":"Home","title":"ZigZagBoomerang.pdmp","text":"pdmp(∇ϕ, x, θ, T, Flow::ContinuousDynamics; adapt=true,  factor=2.0)\n\nRun a piecewise deterministic process from location and velocity x, θ until time T. c is a tuning parameter for the upper bound of the Poisson rate. If adapt = false, c = c*factor is tried, otherwise an error is thrown.\n\nReturns vector of tuples (t, x, θ) (time, location, velocity) of direction change events.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.poisson_time-Tuple{Any, Any, Any}","page":"Home","title":"ZigZagBoomerang.poisson_time","text":"poisson_time(a, b, u)\n\nObtaining waiting time for inhomogeneous Poisson Process with rate of the form λ(t) = (a + b*t)^+, a,b ∈ R, u uniform random variable\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.poisson_time-Tuple{Number, Number}","page":"Home","title":"ZigZagBoomerang.poisson_time","text":"poisson_time(a[, u])\n\nObtaining waiting time for homogeneous Poisson Process with rate of the form λ(t) = a, a ≥ 0, u uniform random variable\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.poisson_time-Tuple{Tuple{T, T, T} where T, Any}","page":"Home","title":"ZigZagBoomerang.poisson_time","text":"poisson_time((a, b, c), u)\n\nObtaining waiting time for inhomogeneous Poisson Process with rate of the form λ(t) = c + (a + b*t)^+,  where c> 0 ,a, b ∈ R, u uniform random variable\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.pos-Tuple{Any}","page":"Home","title":"ZigZagBoomerang.pos","text":"pos(x)\n\nPositive part of x (i.e. max(0,x)).\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.queue_time!-Tuple{Any, Any, Any, Any, Any, Any, Any, Any, ZigZag}","page":"Home","title":"ZigZagBoomerang.queue_time!","text":"queue_time!(Q, t, x, θ, i, b, f, Z::ZigZag)\n\nComputes the (proposed) reflection time and the freezing time of the ith coordinate and enqueue the first one. f[i] = true if the next time is a freezing time.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.reflect!-Tuple{Any, Any, Any, BouncyParticle}","page":"Home","title":"ZigZagBoomerang.reflect!","text":"reflect!(∇ϕx, θ, F::BouncyParticle, Boomerang)\n\nReflection rule of sampler F at reflection time. x: position,θ`: velocity\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.reflect!-Tuple{Any, Number, Any, Any, Union{FactBoomerang, ZigZag}}","page":"Home","title":"ZigZagBoomerang.reflect!","text":"    reflect!(i, x, θ, F)\n\nReflection rule of sampler F at reflection time. i: coordinate which flips sign, x: position, θ: velocity (position not used for the ZigZag and FactBoomerang.)\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.sdiscretize-Tuple{Vector, Union{Boomerang, BouncyParticle}, Any}","page":"Home","title":"ZigZagBoomerang.sdiscretize","text":"discretize(x::Vector, Flow::Union{BouncyParticle, Boomerang}, dt)\n\nTransform the output of the algorithm (a skeleton of points) to a trajectory. multi-dimensional version.\n\nOld version that would not work with the sticky Boomerang sampler not centered in 0\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.spdmp-Tuple{Any, Any, Any, Any, Any, Any, Any, Union{FactBoomerang, ZigZagBoomerang.JointFlow, ZigZag}, Vararg{Any}}","page":"Home","title":"ZigZagBoomerang.spdmp","text":"spdmp(∇ϕ, t0, x0, θ0, T, c, [G,] F::Union{ZigZag,FactBoomerang}, args...;\n    factor=1.5, adapt=false)\n    = Ξ, (t, x, θ), (acc, num), c\n\nVersion of spdmp which assumes that i only depends on coordinates x[j] for j in neighbours(G, i).\n\nIt returns a FactTrace (see Trace) object Ξ, which can be collected into pairs t => x of times and locations and discretized with discretize. Also returns the number of total and accepted Poisson events and updated bounds c (in case of adapt==true the bounds are multiplied by factor if they turn out to be too small.) The final time, location and momentum at T can be obtained with smove_forward!(t, x, θ, T, F).\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.spdmp-Tuple{Any, Any, Any, Any, Any, ZigZagBoomerang.LocalBound, Any, Union{FactBoomerang, ZigZagBoomerang.JointFlow, ZigZag}, Vararg{Any}}","page":"Home","title":"ZigZagBoomerang.spdmp","text":"spdmp(∇ϕ, t0, x0, θ0, T, c, [G,] F::Union{ZigZag,FactBoomerang}, args...;     factor=1.5, adapt=false)     = Ξ, (t, x, θ), (acc, num), c\n\nVersion of spdmp which assumes that i only depends on coordinates x[j] for j in neighbours(G, i).\n\nIt returns a FactTrace (see Trace) object Ξ, which can be collected into pairs t => x of times and locations and discretized with discretize. Also returns the number of total and accepted Poisson events and updated bounds c (in case of adapt==true the bounds are multiplied by factor if they turn out to be too small.) The final time, location and momentum at T can be obtained with smove_forward!(t, x, θ, T, F).\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.spdmp_inner!-Tuple{Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Union{FactBoomerang, ZigZag}, Vararg{Any}}","page":"Home","title":"ZigZagBoomerang.spdmp_inner!","text":"spdmp_inner!(rng, G, G1, G2, ∇ϕ, t, x, θ, Q, c, b, t_old, (acc, num),\n\nF::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false, adaptscale=false)\n\n[Outdated] Inner loop of the factorised samplers: the factorised Boomerang algorithm and the Zig-Zag sampler. Given a dependency graph G, gradient ∇ϕ, current position x, velocity θ, Queue of events Q, time t, tuning parameter c, terms of the affine bounds a,b and time when the upper bounds were computed t_old\n\nThe sampler 1) extracts from the queue the first event time. 2) moves deterministically according to its dynamics until event time. 3) Evaluates whether the event time is a accepted reflection or refreshment time or shadow time. 4) If it is a reflection time, the velocity reflects according its reflection rule, if it is a refreshment time, the sampler updates the velocity from its prior distribution (Gaussian). In both cases, updates Q according to the dependency graph G. The sampler proceeds until the next accepted reflection time or refreshment time. (num, acc) incrementally counts how many event times occour and how many of those are real reflection times.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.spdmp_inner!-Tuple{Any, Any, Any, Any, Any, Any, Any, Any, ZigZagBoomerang.LocalBound, Any, Any, Any, Union{FactBoomerang, ZigZagBoomerang.JointFlow, ZigZag}, Vararg{Any}}","page":"Home","title":"ZigZagBoomerang.spdmp_inner!","text":"spdmpinner!(rng, G, G1, G2, ∇ϕ, t, x, θ, Q, c, b, told, (acc, num), F::Union{ZigZag,FactBoomerang}, args...; factor=1.5, adapt=false, adaptscale=false)\n\n[Outdated] Inner loop of the factorised samplers: the factorised Boomerang algorithm and the Zig-Zag sampler. Given a dependency graph G, gradient ∇ϕ, current position x, velocity θ, Queue of events Q, time t, tuning parameter c, terms of the affine bounds a,b and time when the upper bounds were computed t_old\n\nThe sampler 1) extracts from the queue the first event time. 2) moves deterministically according to its dynamics until event time. 3) Evaluates whether the event time is a accepted reflection or refreshment time or shadow time. 4) If it is a reflection time, the velocity reflects according its reflection rule, if it is a refreshment time, the sampler updates the velocity from its prior distribution (Gaussian). In both cases, updates Q according to the dependency graph G. The sampler proceeds until the next accepted reflection time or refreshment time. (num, acc) incrementally counts how many event times occour and how many of those are real reflection times.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.splitpairs-Tuple{Any}","page":"Home","title":"ZigZagBoomerang.splitpairs","text":"splitpairs(tx) = t, x\n\nSplits a vector of pairs into a pair of vectors.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.ssmove_forward!-Tuple{Any, Any, Any, Any, Any, Any, Union{BouncyParticle, ZigZag}}","page":"Home","title":"ZigZagBoomerang.ssmove_forward!","text":"t, x, θ = ssmove_forward!(G, i, t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})\n\nmoves forward only the non_frozen particles neighbours of i\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.ssmove_forward!-Tuple{Any, Any, Any, Any, Union{BouncyParticle, ZigZag}}","page":"Home","title":"ZigZagBoomerang.ssmove_forward!","text":"t, x, θ = ssmove_forward!(t, x, θ, t′, Z::Union{BouncyParticle, ZigZag})\n\nmoves forward only the non_frozen particles\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.sspdmp_inner!-Tuple{Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, ZigZag, Any, Vararg{Any}}","page":"Home","title":"ZigZagBoomerang.sspdmp_inner!","text":"sspdmp_inner!(Ξ, G, G1, G2, ∇ϕ, t, x, θ, Q, c, b, t_old, f, θf, (acc, num),\n        F::ZigZag, κ, args...; strong_upperbounds = false, factor=1.5, adapt=false)\n\nInner loop of the sticky ZigZag sampler. G[i] are indices which have to be moved, G1[i] is the set of indices used to derive the bounding rate λbari and G2 are the indices k in Aj for all j : i in Aj (neighbours of neighbours)\n\nIs assumed that ∇ϕ[x, i] is function of x_i with i in G[i] or that ∇ϕ takes care of moving .\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.subtrace-Tuple{Any, Any}","page":"Home","title":"ZigZagBoomerang.subtrace","text":"subtrace(tr, J)\n\nCompute the trace of a subvector x[J], returns a trace object.\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.λ-Tuple{Any, Any, Any, Any, FactBoomerang}","page":"Home","title":"ZigZagBoomerang.λ","text":"λ(∇ϕ, i, x, θ, Z::FactBoomerang)\n\nith Poisson rate of the FactBoomerang sampler\n\n\n\n\n\n","category":"method"},{"location":"#ZigZagBoomerang.λ-Tuple{Any, Any, Any, Any, ZigZag}","page":"Home","title":"ZigZagBoomerang.λ","text":"λ(∇ϕ, i, x, θ, Z::ZigZag)\n\nith Poisson rate of the ZigZag sampler\n\n\n\n\n\n","category":"method"},{"location":"#Literature","page":"Home","title":"Literature","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Joris Bierkens, Paul Fearnhead, Gareth Roberts: The Zig-Zag Process and Super-Efficient Sampling for Bayesian Analysis of Big Data. The Annals of Statistics, 2019, 47. Vol., Nr. 3, pp. 1288-1320. https://arxiv.org/abs/1607.03188.\nJoris Bierkens, Sebastiano Grazzi, Kengo Kamatani and Gareth Robers: The Boomerang Sampler. ICML 2020. https://arxiv.org/abs/2006.13777.\nJoris Bierkens, Sebastiano Grazzi, Frank van der Meulen, Moritz Schauer: A piecewise deterministic Monte Carlo method for diffusion bridges.  2020. https://arxiv.org/abs/2001.05889.\nhttps://github.com/jbierkens/ICML-boomerang/ (code accompanying the paper \"The Boomerang Sampler\")","category":"page"}]
}
