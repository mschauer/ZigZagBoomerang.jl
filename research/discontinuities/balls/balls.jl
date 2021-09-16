using  LinearAlgebra
# using CairoMakie

###
N = 50
dim = 2 # particles in a plane
x = zeros((N+1)*dim + dim)

# legal region 
function legal(i, x, epsilon)
    y = x[dim*i+1:dim*(i+1)] 
    norm(y - x[1:dim]) > epsilon
end


ϵ = 1.0
# initialize particles in a legal region
for i in 1:N
    while true 
        if legal(i, x, ϵ)
            break
        end
        x[i*dim + 1: dim*(i+1)] =  randn(dim)
    end
end

x0 = deepcopy(x) # initial position

# draw circle
function draw_circ(μ, rsq)
    r = sqrt(rsq)
    θ = LinRange(0,2π, 1000)
    μ[1] .+ r*sin.(θ), μ[2] .+ r*cos.(θ)
end

using CairoMakie
odd = 1:2:2*N-1
even = 2:2:2*N
p = Figure(Aspect = (1,1))
axes = Axis(p[1,1])
limits!(axes, -5, 5, -5, 5)
scatter!(x[odd], x[even])
x1_1, x1_2 = draw_circ(x[1:2], ϵ)
lines!(x1_1,x1_2)
current_figure()

# standard gaussian log-likelihood
ϕi(x) = sum(x) 
ϕ(x) = sum([ϕi(x[i,i+1]...) for i in 1:2:N-1])




