using SuiteSparseMatrixCollection, MatrixMarket

ssmc = ssmc_db(verbose=false)

function load_case(case)
    data = ssmc_matrices(ssmc, "", case)
    path = fetch_ssmc(data, format="MM")[1]
    A = MatrixMarket.mmread(path * "/$(case).mtx")
    b = ones(size(A,1))
    return A, b
end
