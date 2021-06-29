function augment(t, x)
   z = abs.(x) .>= 2*eps()  
   return z
end
z = augment.(t, x)

function inclusion_probability(ts0, z)
   dict = Dict(z[1] => 0.0)
   for (δt, i) in zip(diff(ts0), eachindex(ts0))
      if haskey(dict, z[i])
         dict[z[i]] += δt
      else
         push!(dict, z[i] => δt)
      end
   end
return dict
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
error("")

