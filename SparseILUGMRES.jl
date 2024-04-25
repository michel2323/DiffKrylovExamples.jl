using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules
using DiffKrylov
using Krylov
using LinearAlgebra
using SparseArrays
using Test
using FiniteDifferences
using IncompleteLU

include("get_div_grad.jl")
include("utils.jl")

# Pick solver
solver = Krylov.gmres

# Build Laplacian
function sparse_laplacian(n :: Int=16; FC=Float64)
    A = get_div_grad(n, n, n)
    b = ones(n^3)
    return A, b
end
A, b = sparse_laplacian(4, FC=Float64)

# Central Differences
fdm = central_fdm(8, 1);
function A_one_one(x)
    _A = copy(A)
    _A[1,1] = x
    solver(_A,b)
end

function b_one(x)
    _b = copy(b)
    _b[1] = x
    solver(A,_b)
end

fda = FiniteDifferences.jacobian(fdm, a -> A_one_one(a)[1], copy(A[1,1]))
fdb = FiniteDifferences.jacobian(fdm, a -> b_one(a)[1], copy(b[1]))
fd =fda[1] + fdb[1]

# Build a structured shadow of a sparse matrix, setting the values to 0
function duplicate(A::SparseMatrixCSC)
    dA = copy(A)
    fill!(dA.nzval, zero(eltype(A)))
    return dA
end

# Forward AD
dA = Duplicated(A, duplicate(A))
db = Duplicated(b, zeros(length(b)))
dA.dval[1,1] = 1.0
db.dval[1] = 1.0
dx = Enzyme.autodiff(
    Forward,
    solver,
    dA,
    db
)

# Check against FD
@test isapprox(dx[1][1], fd, atol=1e-4, rtol=1e-4)
# Reverse AD
function driver!(x, A, b, P)
    x .= gmres(A,b; M=P, N=I,
        verbose=1,
        ldiv = isa(P, IncompleteLU.ILUFactorization) ? true : false
    )[1]
    nothing
end
# Without preconditioner
println("GMRES with no preconditioner")
dA = Duplicated(A, duplicate(A))
db = Duplicated(b, zeros(length(b)))
dx = Duplicated(zeros(length(b)), zeros(length(b)))
dx.dval[1] = 1.0
P = I
Enzyme.autodiff(
    Reverse,
    driver!,
    dx,
    dA,
    db,
    Const(P) # nopreconditioning
)
# With preconditioner
println("GMRES with ILU")
dA = Duplicated(A, duplicate(A))
db = Duplicated(b, zeros(length(b)))
dx = Duplicated(zeros(length(b)), zeros(length(b)))
dx.dval[1] = 1.0
P = ilu(A)
Enzyme.autodiff(
    Reverse,
    driver!,
    dx,
    dA,
    db,
    Const(P) # nopreconditioning
)

# Check against FD
@test isapprox(db.dval[1], fdb[1][1], atol=1e-4, rtol=1e-4)
@test isapprox(dA.dval[1,1], fda[1][1], atol=1e-4, rtol=1e-4)