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
using CSV
using DataFrames

include("../src/utils.jl")

# Build a structured shadow of a sparse matrix, setting the values to 0
function duplicate(A::SparseMatrixCSC)
    dA = copy(A)
    fill!(dA.nzval, zero(eltype(A)))
    return dA
end

# Reverse AD
function driver!(solver::GmresSolver, A, b, P, restart=false, atol=1e-6, rtol=1e-6)
    gmres!(solver, A,b; M=P, N=I,
        verbose=0, restart=restart,
        ldiv = isa(P, IncompleteLU.ILUFactorization) ? true : false,
        atol = atol, rtol = rtol,
        itmax = length(b)
    )
    return nothing
end

function driver!(solver::BicgstabSolver, A, b, P, restart=false, atol=1e-6, rtol=1e-6)
    bicgstab!(solver, A,b; M=P, N=I,
        verbose=0,
        ldiv = isa(P, IncompleteLU.ILUFactorization) ? true : false,
        atol = atol, rtol = rtol,
        itmax = length(b)
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
function compare_precond_noprecond_ilu(fsolver, A, b, restart, atol, rtol, noprecond)
    A, Dr, Dc = unsymmetric_scaling(A)
    b = Dr * ones(length(b))
    db = copy(b)
    # iters, stats = if noprecond
        solver = init_solver(fsolver, A, b, restart)
        dupsolver = Duplicated(solver, init_solver(fsolver, A, b, restart))
        dupA = Duplicated(A, duplicate(A))
        dupb = Duplicated(b, zeros(length(b)))
        dupsolver.dval.x .= db
        P = I
        Enzyme.autodiff(
            Reverse,
            driver!,
            dupsolver,
            Const(A),
            dupb,
            Const(P), # nopreconditioning
            Const(restart),
            Const(atol),
            Const(rtol),
        )
        # (
        #     [dupsolver.val.stats.niter, dupsolver.dval.stats.niter],
        #     [dupsolver.val.stats.status, dupsolver.dval.stats.status]
        # )
    # else
    #     [0, 0], [0, 0]
    # end
    iters = [dupsolver.val.stats.niter, dupsolver.dval.stats.niter]
    stats = [dupsolver.val.stats.status, dupsolver.dval.stats.status]
    res = [norm(dupb.val - A*dupsolver.val.x), norm(db - adjoint(A)*dupsolver.dval.x)]
    # With preconditioner
    solver = init_solver(fsolver, A, b, restart)
    dupsolver = Duplicated(solver, init_solver(fsolver, A, b, restart))
    dupA = Duplicated(A, duplicate(A))
    dupb = Duplicated(ones(length(b)), zeros(length(b)))
    dupsolver.dval.x .= db
    P = ilu(A)
    rb = copy(b)
    normr = norm(ldiv!(P,rb))
    rtol = atol/normr
    println("rtol for precond. systems: $rtol")
    Enzyme.autodiff(
        Reverse,
        driver!,
        dupsolver,
        Const(A),
        dupb,
        Const(P), # nopreconditioning
        Const(restart),
        Const(atol),
        Const(rtol),
    )
    iters = vcat(iters, [dupsolver.val.stats.niter, dupsolver.dval.stats.niter])
    stats = vcat(stats, [dupsolver.val.stats.status, dupsolver.dval.stats.status])
    res = vcat(res, [norm(dupb.val - A*dupsolver.val.x), norm(db - adjoint(A)*dupsolver.dval.x)])
    return (
        iters, stats, res
    )
end

include("../src/load_case.jl")
include("../cases/xyz_cases.jl")
include("../src/scaling.jl")

results_header = ["case", "nit_noP", "nit_P", "adj_nit_noP", "adj_nit_P", "n", "res", "Pres", "adj_res", "adj_Pres"]
stats_header = ["case", "nit_noP", "nit_P", "adj_nit_noP", "adj_nit_P"]
# for tol in [1e-2] #[1e-2, 1e-5, 1e-8]
for tol in [(1e-10, 1e-10)]
# for setup in [(GmresSolver, false)]
# for setup in [(GmresSolver, true)]#, (BicgstabSolver, false)]
for setup in [(GmresSolver, false), (GmresSolver, true), (BicgstabSolver, false)]
    atol = tol[1]
    rtol = tol[2]
    fsolver, restart = setup
    cases = xyz_cases
    mkpath("./results/$(tol)")
    # mkpath("./results/draft")
    results_file = "./results/$(tol)/iters_$(fsolver)_$(restart).csv"
    # results_file = "./results/draft/iters_$(fsolver)_$(restart).csv"
    stats_file = "./results/$(tol)/stats_$(fsolver)_$(restart).csv"
    # stats_file = "./results/draft/stats_$(fsolver)_$(restart).csv"
    results = Matrix(undef, 0, 10)
    stats = Matrix(undef, 0, 5)
    # for (i, case) in enumerate([cases[9]])
    for (i,case) in enumerate(cases)
        println("$(case[1]) with solver=$fsolver, restart=$restart, tol=$tol")
        A, b = load_case(case[1])
        if eltype(A) <: Complex
            continue
        end
        if eltype(A) <: Int
            A = SparseMatrixCSC{Float64,Int}(A)
            b = Vector{Float64}(b)
        end
        n = size(A,1)
        iters, stat, res = compare_precond_noprecond_ilu(fsolver,A,b,restart,atol, rtol, false)
        results = vcat(results, reshape([case[1], iters[1], iters[3], iters[2], iters[4], res[1], res[3], res[2], res[4], n], (1,10)))
        stats = vcat(stats, reshape([case[1], stat[1], stat[2], stat[3], stat[4]], (1,5)))
        df_results = DataFrame(results, Symbol.(results_header))
        df_stats = DataFrame(stats, Symbol.(stats_header))
        CSV.write(results_file, df_results)
        CSV.write(stats_file, df_stats)
    end
end
end
