function scaleSID(A::SparseMatrixCSC, iprint::Bool=false)
    n, m = size(A)
    if iprint
        println("\n scaleSID: Symmetric scaling of SID matrix.   n = $n m = $m  nnz = $(nnz(A))")
    end

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
    A2 = copy(A)
    DAD = copy(A)
    mul!(A2, A, D)
    mul!(DAD, D, A2)

    if iprint
        jmin = argmin(scale)
        jmax = argmax(scale)
        smin = scale[jmin]
        smax = scale[jmax]
        println("\n\n  Min scale                     Max scale")
        println("  Col $jmin $smin    Col $jmax $smax")
    end

    return DAD, D
end