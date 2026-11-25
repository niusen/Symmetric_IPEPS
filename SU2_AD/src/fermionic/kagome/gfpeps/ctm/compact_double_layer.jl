# Include inside KagomeFermionicCTM.Legacy. No shared source is modified.
# Same bra swaps, fusion maps and final crossing signs as the shared routines.
# Contract the physical indices BEFORE fusion: the largest physical intermediate
# has D^8 entries instead of p*D^8, and there is no dense D^4-by-D^4 identity
# factorization of the left/down pair. This matters for p=64, D=8.

const compact_double_layer_enabled=Ref(false)

# Record the retained projector rank, including runs using the shared double layer.
const gfpeps_projector_ranks=Int[]
function sdiag_inv_sqrt(S::DiagonalTensorMap{Float64})
    R=invoke(sdiag_inv_sqrt,Tuple{DiagonalTensorMap},S)
    push!(gfpeps_projector_ranks,sum(count(!iszero,diag(b)) for (_,b) in blocks(R)))
    R
end

function compact_double_layer(Ap,A,extra::Bool=false)
    sectortype(space(A,1))==Z2Irrep || throw(ArgumentError("compact benchmark contraction is Z2 only"))
    gate=swap_gate(Ap,1,4)
    @tensor Ap[:]:=Ap[1,-2,-3,2,-5]*gate[-1,-4,1,2]
    gate=swap_gate(Ap,2,3)
    @tensor Ap[:]:=Ap[-1,1,2,-4,-5]*gate[-2,-3,1,2]
    gate=parity_gate(Ap,4)
    @tensor Ap[:]:=Ap[-1,-2,-3,1,-5]*gate[-4,1]
    gate=parity_gate(Ap,2)
    @tensor Ap[:]:=Ap[-1,1,-3,-4,-5]*gate[-2,1]
    Ap=permute(Ap,(1,2),(3,4,5))
    A=extra ? permute(A,(1,2),(3,4,5,6)) : permute(A,(1,2),(3,4,5))
    U_L=unitary(fuse(space(Ap,1)⊗space(A,1)),space(Ap,1)⊗space(A,1))
    U_D=unitary(fuse(space(Ap,2)⊗space(A,2)),space(Ap,2)⊗space(A,2))
    U_R=unitary(space(Ap,3)'⊗space(A,3)',fuse(space(Ap,3)'⊗space(A,3)'))
    U_U=unitary(space(Ap,4)'⊗space(A,4)',fuse(space(Ap,4)'⊗space(A,4)'))
    if extra
        @tensor raw[-1,-2,-3,-4,-5,-6,-7,-8,-9] := Ap[-1,-3,-5,-7,1]*A[-2,-4,-6,-8,1,-9]
        @tensor t1[-1,-2,-3,-4,-5,-6,-7,-8] := U_L[-1,1,2]*raw[1,2,-2,-3,-4,-5,-6,-7,-8]
        @tensor t2[-1,-2,-3,-4,-5,-6,-7] := U_D[-2,1,2]*t1[-1,1,2,-3,-4,-5,-6,-7]
        @tensor t3[-1,-2,-3,-4,-5,-6] := t2[-1,-2,1,2,-4,-5,-6]*U_R[1,2,-3]
        @tensor AA[-1,-2,-3,-4,-5] := t3[-1,-2,-3,1,2,-5]*U_U[1,2,-4]
    else
        @tensor raw[-1,-2,-3,-4,-5,-6,-7,-8] := Ap[-1,-3,-5,-7,1]*A[-2,-4,-6,-8,1]
        @tensor t1[-1,-2,-3,-4,-5,-6,-7] := U_L[-1,1,2]*raw[1,2,-2,-3,-4,-5,-6,-7]
        @tensor t2[-1,-2,-3,-4,-5,-6] := U_D[-2,1,2]*t1[-1,1,2,-3,-4,-5,-6]
        @tensor t3[-1,-2,-3,-4,-5] := t2[-1,-2,1,2,-4,-5]*U_R[1,2,-3]
        @tensor AA[-1,-2,-3,-4] := t3[-1,-2,-3,1,2]*U_U[1,2,-4]
    end
    # These projectors express the same two U_LU and U_DR crossing maps as
    # build_double_layer_swap_extra_leg, without forming their large products.
    P_odd_Lp,_=projector_parity(space(U_L',1))
    P_odd_Up,_=projector_parity(space(U_U',2))
    P_odd_U,_=projector_parity(space(U_U',3))
    @tensor isom_Lp[:]:=U_L[-1,4,3]*P_odd_Lp'[4,1]*P_odd_Lp[1,2]*U_L'[2,3,-2]
    @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2]
    @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6]
    if extra
        @tensor correction[:]:=AA[1,-2,-3,4,-5]*isom_Lp[-1,1]*isom_U[-4,4]
        AA=AA-2*correction
        @tensor correction[:]:=AA[-1,-2,-3,4,-5]*isom_Up_U[-4,4]
        AA=AA-2*correction
    else
        @tensor correction[:]:=AA[1,-2,-3,4]*isom_Lp[-1,1]*isom_U[-4,4]
        AA=AA-2*correction
        @tensor correction[:]:=AA[-1,-2,-3,4]*isom_Up_U[-4,4]
        AA=AA-2*correction
    end
    P_odd_Dp,_=projector_parity(space(U_D',1))
    P_odd_D,_=projector_parity(space(U_D',2))
    P_odd_R,_=projector_parity(space(U_R',3))
    @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2]
    @tensor isom_R[:]:=U_R[3,4,-1]*P_odd_R'[4,1]*P_odd_R[1,2]*U_R'[-2,3,2]
    @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2]
    if extra
        @tensor correction[:]:=AA[-1,2,-3,-4,-5]*isom_Dp_D[-2,2]
        AA=AA-2*correction
        @tensor correction[:]:=AA[-1,2,3,-4,-5]*isom_Dp[-2,2]*isom_R[-3,3]
        AA=AA-2*correction
    else
        @tensor correction[:]:=AA[-1,2,-3,-4]*isom_Dp_D[-2,2]
        AA=AA-2*correction
        @tensor correction[:]:=AA[-1,2,3,-4]*isom_Dp[-2,2]*isom_R[-3,3]
        AA=AA-2*correction
    end
    return AA,U_L,U_D,U_R,U_U
end

# Installation is explicit, so the unmodified shared functions remain available
# through invoke(..., Tuple{Any,Any}, ...) for independent numerical comparisons.
function install_compact_double_layer!()
    compact_double_layer_enabled[] && return nothing
    @eval begin
        build_double_layer_swap(Ap::AbstractTensorMap,A::AbstractTensorMap)=
            sectortype(space(A,1))==Z2Irrep ? compact_double_layer(Ap,A,false) : invoke(build_double_layer_swap,Tuple{Any,Any},Ap,A)
        build_double_layer_swap_extra_leg(Ap::AbstractTensorMap,A::AbstractTensorMap)=
            sectortype(space(A,1))==Z2Irrep ? compact_double_layer(Ap,A,true) : invoke(build_double_layer_swap_extra_leg,Tuple{Any,Any},Ap,A)
    end
    compact_double_layer_enabled[]=true
    nothing
end
