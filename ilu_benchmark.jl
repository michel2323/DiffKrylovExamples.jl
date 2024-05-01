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
using DelimitedFiles

include("get_div_grad.jl")
include("utils.jl")

ssmc = ssmc_db(verbose=false)
cases  = [
    ["sherman5"],
    ["powersim"],
    ["Ill_Stokes"],
    ["rma10"],
    ["venkat50"],
    ["sme3Dc"],
    ["ecl32"],
    ["poisson3Db"],
    ["ohne2"],
    ["thermomech_dK"],
    ["marine1"],
    ["Freescale1"],
]

function load_case(case)
    data = ssmc_matrices(ssmc, "", case)
    path = fetch_ssmc(data, format="MM")[1]
    A = MatrixMarket.mmread(path * "/$(case).mtx")
    b = ones(size(A,1))
    return A, b
end
# Build Laplacian

function sparse_laplacian(n :: Int=16; FC=Float64)
    A = get_div_grad(n, n, n)
    b = ones(n^3)
    return A, b
end

# Build a structured shadow of a sparse matrix, setting the values to 0
function duplicate(A::SparseMatrixCSC)
    dA = copy(A)
    fill!(dA.nzval, zero(eltype(A)))
    return dA
end

# Reverse AD
function driver!(solver, A, b, P)
    gmres!(solver, A,b; M=P, N=I,
        verbose=0,
        ldiv = isa(P, IncompleteLU.ILUFactorization) ? true : false
    )
    return nothing
end
# Without preconditioner
function compare_adjoint_ilu(A,b)
    # println("GMRES with no preconditioner")
    solver = GmresSolver(A,b)
    dupsolver = Duplicated(solver, GmresSolver(A,b))
    dupA = Duplicated(A, duplicate(A))
    dupb = Duplicated(b, zeros(length(b)))
    dupsolver.dval.x .= 1.0
    P = I

    # solver = GmresSolver(A,b)
    # driver!(solver, A, b, P)

    # Enzyme.autodiff(
    #     Reverse,
    #     driver!,
    #     dupsolver,
    #     Const(A),
    #     dupb,
    #     Const(P), # nopreconditioning
    # )
    # With preconditioner
    println("GMRES with ILU")
    solver = GmresSolver(A,b)
    dupsolver = Duplicated(solver, GmresSolver(A,b))
    dupA = Duplicated(A, duplicate(A))
    dupb = Duplicated(ones(length(b)), zeros(length(b)))
    dupsolver.dval.x .= 1.0
    P = ilu(A)
    Enzyme.autodiff(
        Reverse,
        driver!,
        dupsolver,
        Const(A),
        dupb,
        Const(P), # nopreconditioning
    )

    # iters = vcat(iters, [dupsolver.val.stats.niter, dupsolver.dval.stats.niter])
    iters = [dupsolver.val.stats.niter, dupsolver.dval.stats.niter]

    # x = A\b
    # db = ones(length(b))
    # dx = adjoint(A)\db

    # all(isapprox.(dx, dupb.dval, atol=1e-4, rtol=1e-4))
end

results_file = "results.csv"
results = Matrix(undef, size(cases,1), 3)
fill!(results, zero(Int))
for (i,case) in enumerate(cases)
    A, b = load_case(case[1])
    iters = compare_adjoint_ilu(A,b)
    results[i,:] .= case[1], iters[1], iters[2]
end
writedlm(results_file, results)
