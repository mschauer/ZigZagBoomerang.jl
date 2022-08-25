using LinearAlgebra
struct InvChol{T} <: AbstractPDMat{Float64}
    R::T
    InvChol(R::S) where {S<:UpperTriangular} = new{S}(R)
end
PDMats.dim(M::InvChol) = size(M.R,1)
Base.size(M::InvChol) = size(M.R)
function PDMats.unwhiten!(r::Vector{Float64}, M::InvChol{UpperTriangular{Float64, Matrix{Float64}}}, z::Vector{Float64})
    r .= z
    LinearAlgebra.naivesub!(M.R, r)
    r
end    
function LinearAlgebra.mul!(r::AbstractVector, M::InvChol, x::AbstractVector, alpha::Number, beta::Number)
    @assert beta==0
    @. r = alpha*x
    LinearAlgebra.naivesub!(M.R', r)
    LinearAlgebra.naivesub!(M.R, r)
    r
end

function PDMats.whiten!(r::Vector{Float64}, M::InvChol{UpperTriangular{Float64, Matrix{Float64}}}, x::Vector{Float64})
    mul!(r, M.R, x)
    r
end    
function Base.show(io::IOContext, m::MIME{Symbol("text/plain")}, M::InvChol) 
    print(io, "CholInv: ")
    show(io, m, M.R)
end
Base.Matrix(M::InvChol) = M.R'*M.R