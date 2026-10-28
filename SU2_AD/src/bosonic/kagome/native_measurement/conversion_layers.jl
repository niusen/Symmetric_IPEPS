struct DenseKagomeState
    geometry::BS.KagomeSUCellGeometry
    cells::Vector{BS.KagomeSUTensors}
    auxiliary_charges::Dict{Int,Int}
end
# TensorKit uses adjoint plus index permutation, rather than conj(TensorMap).
conj_tensor(t)=TensorKit.permute(t',(Tuple(numin(t)+1:numind(t)),Tuple(1:numin(t))))

function dense_space(V)
    W=ComplexSpace(dim(V))
    return isdual(V) ? W' : W
end
function dense_tensor(t)
    out=foldl(⊗,dense_space.(collect(codomain(t)));init=one(ComplexSpace(1)))
    incoming=foldl(⊗,dense_space.(collect(domain(t)));init=one(ComplexSpace(1)))
    TensorMap(convert(Array,t),out,incoming;tol=1e-12)
end
"""Exact unpadded dense TensorMaps, retaining the selected 1D auxiliary leg."""
function to_dense(source;component=1)
    source isa DenseKagomeState && return deepcopy(source)
    u=source isa KagomeU1Cells.U1KagomeState ? source : KagomeU1Cells.to_u1(source;component)
    fields=fieldnames(BS.KagomeSUTensors)
    cells=[BS.KagomeSUTensors((isnothing(getfield(c,n)) ? nothing : dense_tensor(getfield(c,n))
                              for n in fields)...) for c in u.cells]
    DenseKagomeState(u.geometry,cells,copy(u.auxiliary_charges))
end
function conversion_state(source;representation=:u1,component=1)
    representation==:dense && return to_dense(source;component)
    representation==:u1 || error("representation must be :u1 or :dense")
    source isa KagomeU1Cells.U1KagomeState ? deepcopy(source) : KagomeU1Cells.to_u1(source;component)
end

function prepare_device!(device)
    if device!="cpu" && !isdefined(CTM,:CUDA)
        Core.eval(CTM,:(using CUDA, cuTENSOR, Adapt))
    end
    if device=="cpu"
        return Base.invokelatest(CTM.square_J1_select_device!,device)
    end
    storage=redirect_stdout(devnull) do
        Base.invokelatest(CTM.square_J1_select_device!,device)
    end
    println("kagome TensorKit device = ",device);flush(stdout)
    storage
end
to_device(t)=CTM.square_J1_to_storage(CTM.SQUARE_J1_DEVICE_STORAGE[],t)
to_cpu(t)=CTM.square_J1_to_storage(Array,t)

struct NativeFactors
    state::Any
    Q::Vector{Any}                 # L, B3-down, B2-down, p1,p2,p3,aux
    R::Vector{Any}                 # B3(R+ey), B1(R+ex), B2(R)
    fusion::Vector{Any}            # B1/B2/B3 down-bond bra-ket fusers
end
owner(g,x,y)=BS.representative(g,mod1(x,g.Lx),mod1(y,g.Ly))
function build_factors(state)
    g=state.geometry;qs=Any[];rs=Any[];fusion=Any[]
    for c0 in state.cells
        c=BS.KagomeSUTensors((isnothing(getfield(c0,n)) ? nothing : to_device(complex(getfield(c0,n)/norm(getfield(c0,n))))
                            for n in fieldnames(BS.KagomeSUTensors))...)
        a,b,d=c.B_a,c.B_b,c.B_c;tu=c.T_u
        if numind(tu)==4
            @tensor Q[:] := a[-1,1,-4]*b[-3,2,-5]*d[-2,3,-6]*tu[1,2,3,-7]
        else
            @tensor small[:] := a[-1,1,-4]*b[-3,2,-5]*d[-2,3,-6]*tu[1,2,3]
            W=sectortype(space(small,1))==U1Irrep ? U1Space(0=>1) : ComplexSpace(1)
            spectator=to_device(TensorMap(ones(ComplexF64,1),W,one(W)))
            @tensor Q[:] := small[-1,-2,-3,-4,-5,-6]*spectator[-7]
        end
        R=TensorKit.permute(c.T_d,((3,1,2),()))
        norm(Q)>0 && norm(R)>0 || error("zero projected iPESS factor")
        push!(qs,Q/norm(Q));push!(rs,R/norm(R))
        push!(fusion,[to_device(complex(unitary(fuse(space(t,1)'⊗space(t,1)),space(t,1)'⊗space(t,1))))
                      for t in (c.B_a,c.B_b,c.B_c)])
    end
    NativeFactors(state,qs,rs,fusion)
end
function factor_fusers(f,r)
    g=f.state.geometry;x,y=g.representatives[r]
    lu=f.fusion[r][1];m=f.fusion[r][2];u=f.fusion[r][3]
    # The exact conjugate fusers ensure matching arrows/bases across every bond.
    rd=conj_tensor(f.fusion[owner(g,x,y+1)][3])
    rr=conj_tensor(f.fusion[owner(g,x+1,y)][1]);rm=conj_tensor(m)
    return (lu,m,u),(rd,rr,rm)
end
function double_Q(Q,U;operator=nothing,charged=false)
    if isnothing(operator)
        ket=Q
    elseif charged
        @tensor ket[:] := operator[-4,-5,-6,1,2,3,-8]*Q[-1,-2,-3,1,2,3,-7]
    else
        @tensor ket[:] := operator[-4,-5,-6,1,2,3]*Q[-1,-2,-3,1,2,3,-7]
    end
    bra=conj_tensor(Q);l,m,u=U
    if charged
        @tensor order=(7,8,9,10,1,2,5,6,3,4) layer[:] := bra[1,3,5,7,8,9,10]*ket[2,4,6,7,8,9,10,-4]*l[-1,1,2]*m[-2,5,6]*u[-3,3,4]
    else
        @tensor order=(7,8,9,10,1,2,5,6,3,4) layer[:] := bra[1,3,5,7,8,9,10]*ket[2,4,6,7,8,9,10]*l[-1,1,2]*m[-2,5,6]*u[-3,3,4]
    end
    layer
end
function double_R(R,U)
    bra=conj_tensor(R);d,r,m=U
    @tensor order=(1,2,3,4,5,6) layer[:] := bra[1,3,5]*R[2,4,6]*d[-1,1,2]*r[-2,3,4]*m[-3,5,6]
    layer
end
function fused_trace(U)
    @tensor v[:] := U[-1,1,1]
    conj_tensor(v)
end
function initial_environment(q,r,Uq,Ur)
    l,m,u=fused_trace.(Uq);d,right,rm=fused_trace.(Ur)
    # Trace the external ket/bra legs before Q/R are joined. No rank-4 AA.
    @tensor order=(1,2,3) C1[:] := q[1,3,2]*r[-1,-2,3]*l[1]*u[2]
    @tensor order=(1,2,3) C2[:] := q[-1,3,1]*r[-2,2,3]*u[1]*right[2]
    @tensor order=(1,2,3) C3[:] := q[-2,3,-1]*r[1,2,3]*d[1]*right[2]
    @tensor order=(2,1,3) C4[:] := q[2,3,-2]*r[1,-1,3]*l[2]*d[1]
    @tensor order=(1,2) T4[:] := q[1,2,-3]*r[-1,-2,2]*l[1]
    @tensor order=(1,2) T1[:] := q[-1,2,1]*r[-2,-3,2]*u[1]
    @tensor order=(1,2) T2[:] := q[-2,2,-1]*r[-3,1,2]*right[1]
    @tensor order=(1,2) T3[:] := q[-3,2,-2]*r[1,-1,2]*d[1]
    CTM.Cset_struc(C1,C2,C3,C4),CTM.Tset_struc(T1,T2,T3,T4)
end
