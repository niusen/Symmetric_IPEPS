# Ordinary Z2 TensorMaps do not insert fermionic signs. Every actual exchange
# below explicitly applies 1 - 2 P_odd(i) P_odd(j), as in swap_funs.jl.
_rank(A) = length(codomain(A)) + length(domain(A))
_allout(A) = TensorKit.permute(A, (Tuple(1:_rank(A)), ()))

function _leg_map(A, M, axis)
    n = _rank(A)
    order = (axis, (i for i in 1:n if i != axis)...)
    B = TensorKit.permute(A, ((axis,), Tuple(i for i in 1:n if i != axis)))
    B = _allout(M * B)
    return TensorKit.permute(B, (Tuple(invperm(collect(order))), ()))
end

function _odd_projector(V)
    TensorKit.sectortype(V) == Z2Irrep || throw(ArgumentError("Z2 spaces required"))
    P = unitary(V, V)
    for (q,b) in blocks(P)
        q == Z2Irrep(0) && fill!(b, 0)
    end
    return P
end

function _crossing(A,i,j)
    odd = _leg_map(_leg_map(A,_odd_projector(space(A,i)),i),
                    _odd_projector(space(A,j)),j)
    return _allout(A)-2odd
end

function fpermute(A, order::Tuple)
    n = _rank(A)
    sort(collect(order)) == collect(1:n) || throw(ArgumentError("invalid leg order"))
    B = _allout(A)
    current = collect(1:n)
    for dst in 1:n
        src = findfirst(==(order[dst]), current)
        while src > dst
            i, j = src-1, src
            swap = collect(1:n)
            swap[i], swap[j] = swap[j], swap[i]
            B = TensorKit.permute(_crossing(B,i,j), (Tuple(swap), ()))
            current[i], current[j] = current[j], current[i]
            src -= 1
        end
    end
    return B
end

function _svd(A, Dmax, tol)
    if isdefined(TensorKit, :svd_trunc)
        truncation = TensorKit.truncrank(Dmax) & TensorKit.trunctol(; atol=tol*norm(A))
        u,s,v,_ = TensorKit.svd_trunc(A; trunc=truncation)
    else
        u,s,v = TensorKit.tsvd(A; trunc=TensorKit.truncdim(Dmax) &
                                       TensorKit.truncbelow(tol*norm(A)))
    end
    return u,s,v
end

function _normalized(A)
    n = norm(A)
    isfinite(n) && n > 0 || throw(ArgumentError("zero or nonfinite tensor"))
    return A/n
end

function _pinv_weight(S; rtol=1e-12)
    R = deepcopy(S)
    cutoff = rtol * maximum(abs, diag(convert(Array,S)))
    for (_,b) in blocks(R), i in axes(b,1)
        b[i,i] = abs(b[i,i]) > cutoff ? inv(b[i,i]) : zero(eltype(b))
    end
    return R
end
