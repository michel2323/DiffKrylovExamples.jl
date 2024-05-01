using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules
using DiffKrylov
using LinearAlgebra
using FiniteDifferences
using Krylov
using Random
using SparseArrays
using Test
using IncompleteLU
using SuiteSparseMatrixCollection, MatrixMarket

SimpleStats() = Krylov.SimpleStats(0, false, false, Float64[], Float64[], Float64[], 0.0, "unknown")

include("get_div_grad.jl")
include("utils.jl")

ssmc = ssmc_db(verbose=false)
cases  = [
    ["sherman5", "mc0", "bicgstab", "jacobi", nothing],
    ["powersim", "mc0", "gmres", "jacobi", nothing],
    ["Ill_Stokes", "mc0", "gmres", "jacobi", nothing],
    ["rma10", "mc0", "gmres", "jacobi", nothing],
    ["venkat50", "mc0", "bicgstab", "jacobi", nothing],
    ["sme3Dc", "mc0", "gmres", "ilu2", nothing],
    ["ecl32", "mc0", "gmres", "ilu2", nothing],
    ["poisson3Db", "mc0", "bicgstab", "ilu2", nothing],
    ["ohne2", "mc0", "gmres", "ilu2", nothing],
    ["thermomech_dK", "mc0", "gmres", "ilu2", nothing],
    ["marine1", "mc0", "gmres", "ilu2", nothing],
    ["Freescale1", "mc1", "gmres", "ilu2", nothing],
]

case = cases[1][1]
data = ssmc_matrices(ssmc, "", cases[1][1])
path = fetch_ssmc(data, format="MM")[1]
A = MatrixMarket.mmread(path * "/$(cases[1][1]).mtx")
b = ones(size(A,1))
# Build Laplacian
# function sparse_laplacian(n :: Int=16; FC=Float64)
#     A = get_div_grad(n, n, n)
#     b = ones(n^3)
#     return A, b
# end
# A, b = sparse_laplacian(4, FC=Float64)
solver = Krylov.gmres

# Build a structured shadow of a sparse matrix, setting the values to 0
function duplicate(A::SparseMatrixCSC)
    dA = copy(A)
    fill!(dA.nzval, zero(eltype(A)))
    return dA
end

# Reverse AD
function driver!(x, A, b, P, stats)
    (_x, _stats) = gmres(A,b; M=P, N=I,
        verbose=0,
        ldiv = isa(P, IncompleteLU.ILUFactorization) ? true : false
    )
    copyto!(x, _x)
    copyto!(stats, _stats)
    nothing
end
# Without preconditioner
println("GMRES with no preconditioner")
x = zeros(length(b))
stats = SimpleStats()
dupA = Duplicated(A, duplicate(A))
dupb = Duplicated(b, zeros(length(b)))
dupx = Duplicated(x, zeros(length(x)))
dupstats = Duplicated(stats, SimpleStats())
dupx.dval .= 1.0
P = I

driver!(x, A, b, P, stats)

Enzyme.autodiff(
    Reverse,
    driver!,
    dupx,
    Const(A),
    dupb,
    Const(P), # nopreconditioning
    dupstats
)
println("Krylov forward iterations: ", dupstats.val.niter)
println("Krylov reverse iterations: ", dupstats.dval.niter)
# With preconditioner
println("GMRES with ILU")
x = zeros(length(b))
stats = SimpleStats()
dupA = Duplicated(A, duplicate(A))
dupb = Duplicated(b, zeros(length(b)))
dupx = Duplicated(x, zeros(length(x)))
dupstats = Duplicated(stats, SimpleStats())
dupx.dval .= 1.0
P = ilu(A)
Enzyme.autodiff(
    Reverse,
    driver!,
    dupx,
    Const(A),
    dupb,
    Const(P), # nopreconditioning
    dupstats
)

println("Krylov forward iterations: ", dupstats.val.niter)
println("Krylov reverse iterations: ", dupstats.dval.niter)

# Check against FD

# @test isapprox(db.dval[1], fdb[1][1], atol=1e-4, rtol=1e-4)
# @test isapprox(dA.dval[1,1], fda[1][1], atol=1e-4, rtol=1e-4)
x = A\b
db = ones(length(b))
# db[1] = 1.0
dx = adjoint(A)\db

all(isapprox.(dx, dupb.dval, atol=1e-4, rtol=1e-4))