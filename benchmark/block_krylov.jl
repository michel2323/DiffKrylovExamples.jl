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

function seeds(A, b)
    n = size(A, 1)
    B = zeros(n, n)
    for i = 1:n
        B[i, i] = 1
    end
    return B
end

cases = xyz_cases
A, b = load_case(cases[1][1])

B = seeds(A, b)

# Forward differentiation

# x = A\b

# X = dB - dA*x

# Forward Run

x = A\b

kA, Dr, Dc = unsymmetric_scaling(A)
kb = Dr*b
M = ilu(kA)

atol = 1e-10
rtol = 1e-10
ky, stats = gmres(kA,kb; atol=atol, rtol=rtol, M=M, ldiv=true) 
kx = Dc*ky
@assert isapprox(norm(kx), norm(x), atol=1e-8)

# Reverse run

dx = ones(length(x))
db = adjoint(A)\dx

kdx = Dc*dx
kw, stats = gmres(adjoint(kA),kdx; atol=atol, rtol=rtol, M=adjoint(M), ldiv=true)
kdb = Dr*kw
@assert isapprox(norm(kdb), norm(db), atol=1e-8)

println("Reverse run -- Iterations: $(stats.niter)")

# Vector reverse run

dX = seeds(A, b)
dB = adjoint(A)\dX

kdX = Dc*dX
kW, stats = block_gmres(adjoint(kA),kdX; atol=atol, rtol=rtol, M=adjoint(M), ldiv=true)
kdB = Dr*kW
println("Reverse run -- Iterations: $(stats.niter)")

@assert isapprox(norm(kdB), norm(dB), atol=1e-8)
