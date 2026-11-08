# These physical positions are EXACTLY those of native_observables:
# Tu(R): B1(R),B2(R),B3(R); Td(R): B2(R),B3(R+ey),B1(R+ex).
bond_catalogue(g)=[(;representative=r,orientation=o,edge=e)
    for r in eachindex(g.representatives) for o in (:up,:down) for e in 1:3]
function bond_sites(x,y,orientation,edge)
    sites=orientation==:up ? ((x,y,1),(x,y,2),(x,y,3)) :
          orientation==:down ? ((x,y,2),(x,y+1,3),(x+1,y,1)) : error("invalid triangle")
    a,b=PAIRS[edge];sites[a],sites[b]
end

function operator_factors(Qa,Qb,ia,ib;shift=1.0)
    # Exact operator Schmidt representation of S.S + shift*I.
    # The positive shift prevents division by a vanishing central bond energy.
    # Cov(h+shift*I,h') == Cov(h,h'); it is removed by the connected subtraction.
    oa=N.physical_operators(Qa)
    u1=sectortype(space(Qa,1))==U1Irrep
    W=u1 ? U1Space(0=>2,2=>1,-2=>1) : ComplexSpace(4)
    Wb=W'
    if u1
        qplus=only(sectors(space(oa.plus[ia],7))).charge
        a0=collect(axes(W,U1Irrep(0)))
        b0=collect(axes(Wb,U1Irrep(0)))
        ai=(a0[1],a0[2],only(axes(W,U1Irrep(qplus))),only(axes(W,U1Irrep(-qplus))))
        bi=(b0[1],b0[2],only(axes(Wb,U1Irrep(-qplus))),only(axes(Wb,U1Irrep(qplus))))
    else
        ai=bi=(1,2,3,4)
    end
    Id=Matrix{ComplexF64}(I,8,8)
    matrices(i)=(Id,N.embedded_operator(i,N.SZ),N.embedded_operator(i,N.SP),N.embedded_operator(i,N.SP'))
    av=matrices(ia);bv=matrices(ib)
    build(Q,V,indices,ops)=begin
        P=space(Q,4)⊗space(Q,5)⊗space(Q,6)
        a=zeros(ComplexF64,2,2,2,2,2,2,4)
        for i in 1:4;a[:,:,:,:,:,:,indices[i]].=reshape(ops[i],ntuple(_->2,6));end
        TensorMap(a,P,P⊗V';tol=1e-11)
    end
    A=build(Qa,W,ai,av)
    B=build(Qb,Wb,bi,(shift*bv[1],bv[2],.5bv[4],.5bv[3]))
    # Dense basis of a dual space need not be in an assumed charge order.
    (A=A,B=B,W=W,identity_index=ai[1])
end

function fuse_operator(Q,op,leg)
    @tensor enlarged[:] := op[-4,-5,-6,1,2,3,-8]*Q[-1,-2,-3,1,2,3,-7]
    U=unitary(fuse(space(Q,leg)⊗space(op,7)),space(Q,leg)⊗space(op,7))
    if leg==1
        @tensor value[:] := U[-1,1,2]*enlarged[1,-2,-3,-4,-5,-6,-7,2]
    elseif leg==2
        @tensor value[:] := U[-2,1,2]*enlarged[-1,1,-3,-4,-5,-6,-7,2]
    else
        @tensor value[:] := U[-3,1,2]*enlarged[-1,-2,1,-4,-5,-6,-7,2]
    end
    value,U
end

function route_operator(R,Ua,Ub,ra,rb,W)
    other=only(setdiff(1:3,(ra,rb)))
    rp=permute(R,((ra,rb,other),()))
    a=N.conj_tensor(Ua);b=N.conj_tensor(Ub)
    wire=unitary(W',W')
    @tensor value[:] := a[-1,1,2]*b[-2,3,4]*rp[1,3,-3]*wire[2,4]
    permute(value,(Tuple(invperm([ra,rb,other])),()))
end

function identity_embedding(U,index)
    W=space(U,3)'
    a=zeros(ComplexF64,dim(W));a[index]=1
    v=TensorMap(a,W,one(W);tol=1e-12)
    @tensor emb[:] := U[-1,-2,1]*v[1]
    permute(emb,((1,),(2,)))
end

function mixed_Q(braQ,ketQ,U;operator=nothing,charged=false)
    if !isnothing(operator)
        if charged
            @tensor acted[:] := operator[-4,-5,-6,1,2,3,-8]*ketQ[-1,-2,-3,1,2,3,-7]
        else
            @tensor acted[:] := operator[-4,-5,-6,1,2,3]*ketQ[-1,-2,-3,1,2,3,-7]
        end
        ketQ=acted
    end
    bra=N.conj_tensor(braQ);l,m,u=U
    if charged
        @tensor order=(7,8,9,10,1,2,5,6,3,4) value[:] := bra[1,3,5,7,8,9,10]*ketQ[2,4,6,7,8,9,10,-4]*l[-1,1,2]*m[-2,5,6]*u[-3,3,4]
    else
        @tensor order=(7,8,9,10,1,2,5,6,3,4) value[:] := bra[1,3,5,7,8,9,10]*ketQ[2,4,6,7,8,9,10]*l[-1,1,2]*m[-2,5,6]*u[-3,3,4]
    end
    value
end
function mixed_R(braR,ketR,U)
    bra=N.conj_tensor(braR);d,r,m=U
    @tensor order=(1,2,3,4,5,6) value[:] := bra[1,3,5]*ketR[2,4,6]*d[-1,1,2]*r[-2,3,4]*m[-3,5,6]
    value
end
