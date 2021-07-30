function augment(t, x)
   z = abs.(x) .>= 2*eps()  
   return z
end
z = augment.(t, x)

function inclusion_probability(ts0, z)
   T = ts0[end]
   dict = Dict(z[1] => 0.0)
   for (δt, i) in zip(diff(ts0), eachindex(ts0))
      if haskey(dict, z[i])
         dict[z[i]] += δt/T
      else
         push!(dict, z[i] => δt/T)
      end
   end
return dict
end

function inclusion_probability(trace)
   ts, xs = splitpairs(trace0) 
   zs = augment.(ts, xs)
   inclusion_probability(ts0, z)
end
      
Random.seed!(10)
function ϕ(x, i, μ)
    x[i] - μ[i]
end
κ = .5
n = 2
μ = zeros(n)
x0 = randn(2)
θ0 = [1.0, -1.0]

T = 10.0
c = 0.1*ones(n)
@time trace0, _ = ZigZagBoomerang.sspdmp(ϕ, 0.0, x0, θ0, T, c, ZigZag(sparse(1.0I,n,n), zeros(n)), [κ,κ], μ)
ts0, xs0 = splitpairs(trace0)
zs0 = augment.(ts0, xs0)
zz = inclusion_probability(ts0, zs0)
@assert sum(values(zz)) == 1.0


### Inclusion discrete time sampler
function inclusion_probability_discrete(z, N)
   dict = Dict(z[1] => 0.0)
   for i in eachindex(z)
      if haskey(dict, z[i])
         dict[z[i]] += 1/N
      else
         push!(dict, z[i] => 1/N)
      end
   end
return dict
end

error("")

