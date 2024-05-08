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
function driver!(solver::GmresSolver, A, b, P, restart=false)
    gmres!(solver, A,b; M=P, N=I,
        verbose=0, restart=restart,
        ldiv = isa(P, IncompleteLU.ILUFactorization) ? true : false
    )
    return nothing
end

function driver!(solver::BicgstabSolver, A, b, P, restart=false)
    bicgstab!(solver, A,b; M=P, N=I,
        verbose=0,
        ldiv = isa(P, IncompleteLU.ILUFactorization) ? true : false
    )
    return nothing
end

function init_solver(fsolver, A, b, restart)
    if restart
        fsolver(A, b, 30)
    else
        fsolver(A, b)
    end

end
# Without and with preconditioner
function compare_precond_noprecond_ilu(fsolver, A, b, restart)
    solver = init_solver(fsolver, A, b, restart)
    dupsolver = Duplicated(solver, init_solver(fsolver, A, b, restart))
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
        Const(restart)
    )
    iters = [dupsolver.val.stats.niter, dupsolver.dval.stats.niter]
    stats = [dupsolver.val.stats.status, dupsolver.dval.stats.status]
    # With preconditioner
    solver = init_solver(fsolver, A, b, restart)
    dupsolver = Duplicated(solver, init_solver(fsolver, A, b, restart))
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
        Const(restart)
    )

    iters = vcat(iters, [dupsolver.val.stats.niter, dupsolver.dval.stats.niter])
    stats = vcat(stats, [dupsolver.val.stats.status, dupsolver.dval.stats.status])

    return iters, stats
end

include("../src/load_case.jl")
include("../cases/xyz_cases.jl")
include("../src/scaling.jl")
for setup in [(GmresSolver, false), (GmresSolver, true), (BicgstabSolver, false)]
    # setup = (GmresSolver, false)
    fsolver, restart = setup
    cases = xyz_cases
    results_file = "./results/iters_$(fsolver)_$(restart).csv"
    results = Matrix(undef, 0, 6)
    stats_file = "./results/stats_$(fsolver)_$(restart).csv"
    stats = Matrix(undef, 0, 5)
    for (i,case) in enumerate(cases)
        # i = 1
        # case = cases[i]
        A, b = load_case(case[1])
        if eltype(A) <: Complex
            continue
        end
        if eltype(A) <: Int
            A = SparseMatrixCSC{Float64,Int}(A)
            b = Vector{Float64}(b)
        end
        DAD, D = scaleSID(A)
        A = DAD
        b = D*b
        n = size(A,1)
        iters, stat = compare_precond_noprecond_ilu(fsolver,A,b,restart)
        results = vcat(results, reshape([case[1], iters[1], iters[2], iters[3], iters[4], n], (1,6)))
        stats = vcat(stats, reshape([case[1], stat[1], stat[2], stat[3], stat[4]], (1,5)))
        writedlm(results_file, results)
        writedlm(stats_file, stats)
    end
end
