using IncompleteLU
using LinearAlgebra
using Random
using SparseArrays
using Test

# Testing ILU preconditioners for the adjoint for sparse matrix
# Check whether adjoint(ILU(A)) is the same as ILU(transpose(A))
function test_sparse_matrix_ilu(A)
    iluA = ilu(A)
    tA = sparse(transpose(A))
    ilutA = ilu(tA)
    iluadjA = adjoint(iluA)

    xs = []
    x = ones(size(A,1))
    ldiv!(ilutA, x)
    push!(xs, x)
    x = ones(size(A,1))
    ldiv!(iluadjA, x)
    push!(xs, x)

    @testset "sparse matrix adjoint(ILU(A) vs ILU(transpose(A))" begin
        @test isapprox(norm(xs[1] - xs[2]), zero(eltype(A)), atol=1e-10)
    end
end

# Testing ILU preconditioners for the adjoint for dense matrix
# Compare ILU vs full LU
# norm(ilu(A)\ones) - ldiv!(lu(A), ones)
function test_dense_matrix_ilu(A)
    iluA = ilu(A)
    tA = sparse(transpose(A))
    ilutA = ilu(tA)
    iluadjA = adjoint(iluA)

    xs = []
    b = ones(size(A,1))
    x = similar(b)
    x = similar(b)
    ldiv!(x, ilutA, b)
    push!(xs, x)


    c = ones(size(A,1))
    y = similar(c)
    y = similar(c)
    ldiv!(y, iluadjA, c)
    push!(xs, y)
    @testset "dense matrix ilu vs lu" begin
        @test isapprox(norm(xs[1] - xs[2]), zero(eltype(A)), atol=1e-10)
        @test isapprox(norm(b - A' * x), zero(eltype(A)), atol=1e-6)
        @test isapprox(norm(c - A' * y), zero(eltype(A)), atol=1e-6)
    end
end

# Test random dense matrix
Random.seed!(1)
A = sparse(randn(100,100))
test_dense_matrix_ilu(A)
test_sparse_matrix_ilu(A)

# Use a SSMC matrix
include("../src/load_case.jl")
include("../cases/alexis_cases.jl")

cases = alexis_cases

# Load sherman5
println("Load $(cases[1][1])")
A, _ = load_case(cases[1][1])
test_dense_matrix_ilu(A)
test_sparse_matrix_ilu(A)
