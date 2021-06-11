
using Random
using Distributions
outer(x) = x*x'
d = 5
Random.seed!(3)
μ = μ1 = 0.7randn(d)
μ2 = 0.7randn(d)
Σ1 = 0.1*outer(randn(d,d))
Σ2 = 0.2*outer(randn(d,d))
P = (;μ1=μ1, μ2=μ2, Σ1=Σ1 , Σ2=Σ2)

ℓ_(x, P)  = log(0.5pdf(MvNormal(P.μ1, P.Σ1), x)+0.5pdf(MvNormal(P.μ2, P.Σ2), x)) #+ randn()
ℓ(x, P) = ℓ_(x, P)
    # Set the number of samples to draw and warmup iterations
    n_samples, n_adapts = 8_000, 2_000
if true 

    # Set the number of samples to draw and warmup iterations
    n_samples, n_adapts = 5_000, 1_500
    ℓ(x, P) = ℓ_(x[1:5], P) + ℓ_(x[6:10], P) +  ℓ_(x[11:15], P) 
    μ = [μ1; μ1; μ1]
    d = 3d
end

using AdvancedHMC, Distributions, ForwardDiff

# Choose parameter dimensionality and initial parameter value
D = d; initial_θ = μ

# Define the target distribution
ℓπ(θ) = ℓ(θ, P)

# Define a Hamiltonian system
metric = DiagEuclideanMetric(D)
hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)

# Define a leapfrog solver, with initial step size chosen heuristically
initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
integrator = Leapfrog(initial_ϵ)

# Define an HMC sampler, with the following components
#   - multinomial sampling scheme,
#   - generalised No-U-Turn criteria, and
#   - windowed adaption for step-size and diagonal mass matrix
proposal = AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

# Run the sampler to draw samples from the specified Gaussian, where
#   - `samples` will store the samples
#   - `stats` will store diagnostic statistics for each sample
samples, stats = @time sample(hamiltonian, proposal, initial_θ, n_samples, adaptor, n_adapts; progress=true)
r = eachindex(samples)
t = 1:n_samples
x = samples
if true
    using GLMakie
    fig4 = Figure(resolution=(2000,2000))
    e = min(d, 5)
    for i in 1:min(d^2, e^2)
        u = CartesianIndices((e,e))[i]
        if u[1] == u[2]
            GLMakie.scatter(fig4[u[1],u[2]], t[r], getindex.(x, u[1])[r], markersize=0.5)
            lines!(fig4[u[1],u[2]], t[r], fill(μ1[u[1]], length(r)), color=:green)
            lines!(fig4[u[1],u[2]], t[r], fill(μ2[u[1]], length(r)), color=:green)
            
        else
            GLMakie.scatter(fig4[u[1],u[2]], getindex.(x, u[1])[r],  getindex.(x, u[2])[r], markersize=0.5)
            GLMakie.scatter!(fig4[u[1],u[2]], [
                    μ1[u[1]]
                    μ2[u[1]] 
                    ],[
                    μ1[u[2]] 
                    μ2[u[2]]
                    ], markersize=5.0, color=:lightgreen)    
        end
    end
    display(fig4)
end