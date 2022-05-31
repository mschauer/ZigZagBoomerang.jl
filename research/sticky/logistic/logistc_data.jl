# FILE DOWNLOADED FROM: https://jundongl.github.io/scikit-feature/datasets.html
using MAT, SparseArrays
file = matopen("Prostate_GE.mat")
data = read(file)
# WARNING: everything seems to be shift by 1
y = data["Y"][:] .- 1.0 # rescale by 1
A = (data["X"])  .- 1.0 # rescale by 1
sA = sparse(A)
sA
n, p = size(A)
@show n, p
sΓ = sA'*sA
@show sparsity(sA), sparsity(sΓ)
