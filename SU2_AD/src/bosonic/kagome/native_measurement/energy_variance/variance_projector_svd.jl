# Exact, sector-preserving product SVD for VARIANCE projectors only.
# A=Qa*Ra, B'=Qb*Rb imply A*B=Qa*(Ra*Rb')*Qb'. QR retains every
# intermediate channel. Only C.native_svd applies the original chi/tolerance.
# Neither dense conversion nor changes to ordinary native CTMRG are required.

function variance_left_qr(A::AbstractTensorMap)
    if isdefined(TensorKit,:qr_compact)
        return TensorKit.qr_compact(A) # TensorKit 0.16/0.17: exact compact QR
    elseif isdefined(TensorKit,:leftorth)
        return TensorKit.leftorth(A;alg=TensorKit.QRpos(),atol=0.,rtol=0.)
    end
    error("TensorKit has no supported exact compact QR API")
end

function variance_reduces_product(up::AbstractTensorMap,low::AbstractTensorMap)
    domain(up)==codomain(low) || throw(DimensionMismatch("variance half-map spaces disagree"))
    lowerblocks=Dict(blocks(low))
    # Inspect multiplicity matrices, not just total physical dimensions. This
    # also finds savings in U1/SU2 blocks whose dimensions vary by sector.
    any(haskey(lowerblocks,c) && size(a,2)<min(size(a,1),size(lowerblocks[c],2))
        for (c,a) in blocks(up))
end

function variance_product_svd(up::AbstractTensorMap,low::AbstractTensorMap,chi,tol;
        algorithm::Symbol=:auto)
    algorithm in (:auto,:direct,:qr) || error("projector SVD must be :auto, :direct or :qr")
    domain(up)==codomain(low) || throw(DimensionMismatch("variance half-map spaces disagree"))
    use_qr=algorithm==:qr || (algorithm==:auto && variance_reduces_product(up,low))
    if !use_qr
        M=up*low;n=norm(M)
        isfinite(n) && n>0 || error("zero/non-finite variance projector product")
        return C.native_svd(M/n,chi,tol)
    end
    qa,ra=variance_left_qr(up)
    qb,rb=variance_left_qr(low')
    core=ra*rb';n=norm(core)
    isfinite(n) && n>0 || error("zero/non-finite variance projector core")
    u,s,v=C.native_svd(core/n,chi,tol)
    return qa*u,s,v*qb'
end

function variance_projector_svd(coord,d,Q,R,cs,ts,L,chi,tol;algorithm::Symbol=:auto)
    # Do NOT call C.get_M here: it already allocates the large up*low map.
    up=C.paired_half(coord,d,Q,R,cs,ts,L,L,true)
    low=C.paired_half(coord,d,Q,R,cs,ts,L,L,false)
    nu,nl=norm(up),norm(low)
    isfinite(nu) && isfinite(nl) && nu>0 && nl>0 || error("invalid variance projector half")
    up=up/nu;low=low/nl
    u,s,v=variance_product_svd(up,low,chi,tol;algorithm)
    return u,s,v,up,low
end
