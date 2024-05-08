function symmetric_scaling(A::SparseMatrixCSC)
    n, m = size(A)
    nnzA = nnz(A)

    scale = maximum(abs.(A), dims=1)[:]
    Iz = findall(scale .== 0)
    lenz = length(Iz)
    if lenz > 0
        println("A contains $lenz empty columns")
        scale[Iz] .= 1
    end

    d = diag(A)            # dense column vector
    scale .= 1.0 ./ scale  # dense column vector
    d .= scale .* d        # dense diagonal of DAD
    scale .= sqrt.(scale)  # dense elements of D
    D = Diagonal(scale)    # sparse diagonal D
    AL = tril(A, -1)
    AL2 = copy(AL)
    mul!(AL2, AL, D)
    mul!(AL, D, AL2)
    DAD = AL + AL' + spdiagm(0 => d)

    return DAD, D
end

function unsymmetric_scaling(A::SparseMatrixCSC)
    n, m = size(A)
    nnzA = nnz(A)

    absA = abs.(A)
    scale_row = maximum(absA, dims=2)[:]
    scale_column = maximum(absA, dims=1)[:]

    Ir = findall(scale_row .== 0)
    if length(Ir) > 0
        scale_row[Ir] .= 1
    end
    Ic = findall(scale_column .== 0)
    if length(Ic) > 0
        scale_column[Ic] .= 1
    end 

    scale_row .= 1.0 ./ scale_row
    scale_row .= sqrt.(scale_row)
    Dr = Diagonal(scale_row)
    scale_column .= 1.0 ./ scale_column
    scale_column .= sqrt.(scale_column)
    Dc = Diagonal(scale_column)
    A2 = copy(A)
    DAD = copy(A)
    mul!(A2, A, Dc)
    mul!(DAD, Dr, A2)

    return DAD, Dr, Dc
end
