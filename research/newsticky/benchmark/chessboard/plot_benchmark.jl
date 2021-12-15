using Pkg
Pkg.activate(@__DIR__)
cd(@__DIR__)
 using DataFrames, CSV, Tables,GLMakie
#  using GLMakie

# use filer from DataFrames for filtering data
FileName_df1 = "benchmark.csv"
df1 = DataFrame(CSV.File(FileName_df1))

FileName_df2 = "benchmark1.csv"
df2 = DataFrame(CSV.File(FileName_df2))



function std(ℓ, stat, df1, df2) 
    n_exp = 10
    d = maximum(filter(:ell => l -> l == ℓ, df1).index)
    y = zeros(n_exp*d, 2)    
    samplers = ["SZZ", "GIBBS"]
    for (j,sampler) in enumerate(samplers)
        for i in 1:d
            x̂ = filter([:index, :ell, :stat] => (i0, l, s) -> i0 == i && l == ℓ && s == stat  , df1).val
            xi = filter([:index, :ell, :stat, :sampler] => (i0, l, st, m) -> i0 == i && l == ℓ && st == stat && m == sampler, df2).val
            x₀ = xi .- x̂
            y[(i-1)*n_exp + 1: i*n_exp, j] = x₀ 
        end
    end
    y
end

ℓ = 3
stat = 1
box1 = std(ℓ, stat, df1, df2) 

ℓ = 5
box2 = std(ℓ, stat, df1, df2)

ℓ = 7
box3 = std(ℓ, stat, df1, df2)




x1 = [fill(3, length(box1)); fill(5, length(box2)); fill(7, length(box3))]
x2 = [box1[:]; box2[:]; box3[:]]   
x3 = [fill(1, size(box1,1)); fill(2, size(box1,1)); 
        fill(1, size(box2,1)); fill(2, size(box2,1))
        fill(1, size(box3,1)); fill(2, size(box3,1))]
boxplot(x1,x2, dodge = x3, show_notch = true, color = x3)


