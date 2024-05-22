using MKL
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


include("../src/load_case.jl")
include("../cases/xyz_cases.jl")
include("../src/scaling.jl")

function seeds(A, b, p)
    n = size(A, 1)
    B = zeros(n, p)
    for i = 1:p
        B[i, i] = 1
    end
    return B
end

function run_case(A, b, ps)

    # Forward differentiation

    # x = A\b

    # X = dB - dA*x

    # Forward Run

    x = A \ b

    kA, Dr, Dc = unsymmetric_scaling(A)
    kb = Dr * b
    M = ilu(kA)

    # n = size(M.U, 1)
    # for i = 1:n
    #     M.U[i,i] += 1e-8
    # end

    ky, stats = gmres(kA, kb; atol=atol, rtol=rtol, M=M, ldiv=true, itmax=2000)
    kx = Dc * ky
    # @assert isapprox(norm(kx), norm(x), atol=1e-8)

    # Reverse run

    dx = ones(length(x))
    # db = adjoint(A)\dx

    kdx = Dc*dx
    kw, stats = gmres(adjoint(kA),kdx; atol=atol, rtol=rtol, M=adjoint(M), ldiv=true, itmax=2000)
    kdb = Dr*kw
    # @assert isapprox(norm(kdb), norm(db), atol=1e-8)

    println("Reverse run -- Iterations: $(stats.niter)")
    iters = [stats.niter,]
    status = [stats.status,]

    # Vector reverse run
    for p in ps
        dX = seeds(A, b, p)
        # dB = adjoint(A)\dX

        kdX = Dc * dX
        kW, stats = block_gmres(adjoint(kA),kdX; atol=atol, rtol=rtol, M=adjoint(M), ldiv=true, itmax=2000)
        kdB = Dr * kW
        println("Vector Reverse run -- Iterations: $(stats.niter)")
        push!(iters, stats.niter)
        push!(status, stats.status)

        # @assert isapprox(norm(kdB), norm(dB), atol=1e-8)
    end
    return iters, status
end

cases = xyz_cases

atol = 1e-10
rtol = 1e-10
ps = [8, 16, 32]
results = Matrix(undef, 0, 5)
stats = Matrix(undef, 0, 5)
results_file = "./results/iters_block_gmres.csv"
stats_file = "./results/stats_block_gmres.csv"
results_header = ["case", "gmres", "bgmres8", "bgmres16", "bgmres32"]
stats_header = ["case", "gmres", "bgmres8", "bgmres16", "bgmres32"]
for (i,case) in enumerate(cases)
    println("$(case[1])")
    A, b = load_case(case[1])
    if eltype(A) <: Complex
        continue
    end
    if eltype(A) <: Int
        A = SparseMatrixCSC{Float64,Int}(A)
        b = Vector{Float64}(b)
    end
    iters, stat = run_case(A, b, ps)
    # n = size(A,1)
    # iters, stat, res = compare_precond_noprecond_ilu(fsolver,A,b,restart,atol, rtol, false)
    @show iters
    global results = vcat(results, reshape([case[1], iters[1], iters[2], iters[3], iters[4]], (1,5)))
    global stats = vcat(stats, reshape([case[1], stat[1], stat[2], stat[3], stat[4]], (1,5)))
    df_results = DataFrame(results, Symbol.(results_header))
    df_stats = DataFrame(stats, Symbol.(stats_header))
    CSV.write(results_file, df_results)
    CSV.write(stats_file, df_stats)
end
