using MKL
using Enzyme
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules
using DiffKrylov
using LinearAlgebra
BLAS.set_num_threads(16)
using FiniteDifferences
using Krylov
using Random
using SparseArrays
using Test
using IncompleteLU
using DelimitedFiles

include("../src/utils.jl")

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
# Without and with preconditioner
function compare_precond_noprecond_ilu(A,b)
    # println("GMRES with no preconditioner")
    solver = GmresSolver(A,b)
    dupsolver = Duplicated(solver, GmresSolver(A,b))
    dupA = Duplicated(A, duplicate(A))
    dupb = Duplicated(b, zeros(length(b)))
    dupsolver.dval.x .= 1.0
    P = I
    Enzyme.autodiff(
        Reverse,
        driver!,
        dupsolver,
        Const(A),
        dupb,
        Const(P), # nopreconditioning
    )
    iters = [dupsolver.val.stats.niter, dupsolver.dval.stats.niter]
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

    iters = vcat(iters, [dupsolver.val.stats.niter, dupsolver.dval.stats.niter])

    # x = A\b
    # db = ones(length(b))
    # dx = adjoint(A)\db

    # all(isapprox.(dx, dupb.dval, atol=1e-4, rtol=1e-4))
end

include("../cases/xyz_cases.jl")
cases = xyz_cases
results_file = "results.csv"
results = Matrix(undef, 0, 6)
# A,b = load_case(cases[1][1])
# fill!(results, zero(Int))
for (i,case) in enumerate(cases)
    if i < 47
        continue
    end
    case = cases[i]
    A, b = load_case(case[1])
    n = size(A,1)
    try
        iters = compare_precond_noprecond_ilu(A,b)
    catch
        iters = [0,0,0,0]
    end
    println("GMRES iterations")
    println("----------------")
    println("        \\wo P \\w P")
    println("Forward: $(iters[1]) $(iters[3])")
    println("Reverse: $(iters[2]) $(iters[4])")
    results = vcat(results, reshape([case[1], iters[1], iters[2], iters[3], iters[4], n], (1,6)))
    writedlm(results_file, results)
end
