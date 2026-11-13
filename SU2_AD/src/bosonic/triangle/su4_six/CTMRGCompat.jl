# Included only inside TriangleSU4Six after the original CTMRG source.
# More-specific SU(4) dispatch stays in this module; old model modules and the
# original CTMRG.jl / CTMRG_unitcell.jl files are left unchanged.

# The legacy threshold helper round-trips tensors through Dict/eval only to
# find a scalar. Use the same cutoff rule without requiring SU4 aliases in Main.
function truncate_multiplet(s::DiagonalTensorMap{T,S},chi,multiplet_tol,trun_tol) where {T,S<:GradedSpace{SU4Irrep}}
    values=sort(abs.(diag(convert(Array,s)));rev=true)
    boundary=length(values)>chi ? values[chi+1] : 0.0
    largest=first(values)
    kept=filter(values) do v
        v>boundary && v/largest>=trun_tol &&
            (boundary==0 || abs(v-boundary)/boundary>=multiplet_tol)
    end
    isempty(kept) && error("No complete nonzero CTMRG multiplet fits chi=$chi; increase chi")
    return minimum(kept)
end

function sdiag_inv_sqrt(S::DiagonalTensorMap{T,V}) where {T,V<:GradedSpace{SU4Irrep}}
    global chi,multiplet_tol,projector_trun_tol
    minimum_kept=truncate_multiplet(S,chi,multiplet_tol,projector_trun_tol)
    # Retain the intended boundary despite a few ulps of CG conversion error.
    cutoff=minimum_kept-32eps(float(minimum_kept))
    result=deepcopy(S)
    for (sector,b) in blocks(S)
        block(result,sector).=map(v->v>0 && v>=cutoff ? inv(sqrt(v)) : zero(v),b)
    end
    return result
end
