using IncompleteLU
using LinearAlgebra
using Krylov
using Random
using SparseArrays
using Test

include("../src/scaling.jl")
include("../src/load_case.jl")

# scaling callback
function custom_stopping_condition(solver::KrylovSolver, A, b, r, tol)
    mul!(r, A, solver.x)
    r .-= b               # r := b - Ax
    bool = norm(r) ≤ tol  # tolerance based on the 2-norm of the residual
    return bool
end
A, b = load_case("cdde4")
r = similar(b)
b = ones(length(b))

# Apply scaling
A, Dr, Dc = unsymmetric_scaling(A)
b = Dr*b
# Use two separate RHS for adjoint(A) and transpose(A) solve
dx = copy(b)
bt = copy(b)
# Create tranpose system to check against
tA = sparse(transpose(A))
# Create preconditioner
iluA = ilu(A)
ilutA = ilu(tA)

# Create solvers
solver = GmresSolver(A, b)
adjsolver = GmresSolver(A, b)
tsolver = GmresSolver(tA, tb)

# Set up callbacks
tol = 1e-6
krylov_callback(_solver) = custom_stopping_condition(_solver, A, b, r, tol)
tkrylov_callback(_solver) = custom_stopping_condition(_solver, tA, bt, r, tol)
adjkrylov_callback(_solver) = custom_stopping_condition(_solver, adjoint(A), dx, r, tol)

# atol and rtol set to max
atol = eps(Float64)
rtol = eps(Float64)

# Solve
gmres!(solver, A, b; M=iluA, N=I, verbose=0, callback=krylov_callback, ldiv=true, atol=atol, rtol=rtol)
gmres!(tsolver, tA, tb; M=ilutA, N=I, verbose=1, callback=tkrylov_callback, ldiv=true, atol=atol, rtol=rtol)
gmres!(adjsolver, adjoint(A), dx; M=adjoint(iluA), N=I, verbose=1, callback=adjkrylov_callback, ldiv=true, atol=atol, rtol=rtol)

# Should be the same
@test tsolver.stats.niter == adjsolver.stats.niter

println("solve(A) iterations: $(solver.stats.niter)")
println("solve(tA) iterations: $(tsolver.stats.niter)")
println("solve(adjoint(A)) iterations: $(adjsolver.stats.niter)")
# Transpose A for testing against adjoint(A)
tA = sparse(transpose(A))
