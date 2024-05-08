using IncompleteLU
using LinearAlgebra
using Krylov
using Random
using SparseArrays
using Test

include("../src/scaling.jl")
include("../src/load_case.jl")

# Stopping condition for GMRES against norm(b-Ax)
function custom_stopping_condition(solver::KrylovSolver, A, b, r, tol)
    z = solver.z
    k = solver.inner_iter
    nr = sum(1:k)
    V = solver.V
    R = solver.R
    y = copy(z)

    # Solve Rk * yk = zk
    for i = k : -1 : 1
        pos = nr + i - k
        for j = k : -1 : i+1
        y[i] = y[i] - R[pos] * y[j]
        pos = pos - j + 1
        end
        y[i] = y[i] / R[pos]
    end

    # xk = Vk * yk
    xk = sum(V[i] * y[i] for i = 1:k)
    mul!(r, A, xk)
    r .-= b               # r := b - Ax
    bool = norm(r) ≤ tol  # tolerance based on the 2-norm of the residual
    return bool
end


A, _ = load_case("cdde4")
b = ones(size(A,1))
r = similar(b)
# Use two separate RHS for adjoint(A) and transpose(A) solve
dx = deepcopy(b)
tb = deepcopy(b)
# Create tranpose system to check against
tA = sparse(transpose(deepcopy(A)))

# Apply scaling
A, Dr, Dc = unsymmetric_scaling(A)
tA, tDr, tDc = unsymmetric_scaling(tA)
# Scale all RHS
b = Dr*b
dx = Dr*dx
tb = tDr*tb
# Create preconditioner
iluA = ilu(A)
ilutA = ilu(tA)

# Create solvers
solver = GmresSolver(A, b)
tsolver = GmresSolver(tA, tb)
adjsolver = GmresSolver(A, dx)

# Set up callbacks
tol = 1e-10
krylov_callback(_solver) = custom_stopping_condition(_solver, A, b, r, tol)
tkrylov_callback(_solver) = custom_stopping_condition(_solver, tA, tb, r, tol)
adjkrylov_callback(_solver) = custom_stopping_condition(_solver, adjoint(A), dx, r, tol)

# atol and rtol set to max
atol = eps(Float64)
rtol = eps(Float64)

# Solve
gmres!(solver, A, b; M=iluA, N=I, verbose=0, callback=krylov_callback, ldiv=true, atol=atol, rtol=rtol)
gmres!(tsolver, tA, tb; M=ilutA, N=I, verbose=0, callback=tkrylov_callback, ldiv=true, atol=atol, rtol=rtol)
gmres!(adjsolver, adjoint(A), dx; M=adjoint(iluA), N=I, verbose=0, callback=adjkrylov_callback, ldiv=true, atol=atol, rtol=rtol)

# Should be the same
@test tsolver.stats.niter == adjsolver.stats.niter
@test isapprox(tsolver.x, adjsolver.x)

println("solve(A) iterations: $(solver.stats.niter)")
println("solve(tA) iterations: $(tsolver.stats.niter)")
println("solve(adjoint(A)) iterations: $(adjsolver.stats.niter)")
