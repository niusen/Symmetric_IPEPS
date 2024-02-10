using LinearAlgebra
using TensorKit
using JSON
using HDF5, JLD2, MAT
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\GfPEPS_ctmrg\\swap_gate_ctmrg\\M1")

include("GfPEPS_CTMRG.jl")
include("GfPEPS_model.jl")
include("swap_funs.jl")



chi=30
tol=1e-6




CTM_ite_nums=500;
CTM_trun_tol=1e-10;


data=load("swap_gate_Tensor_M1.jld2")
A=data["A"];   #P1,P2,L,R,D,U

A_new=zeros(1,2,2,2,2,2,2)*im;
A_new[1,:,:,:,:,:,:]=A;
Vdummy=ℂ[U1Irrep](-3=>1);
V=ℂ[U1Irrep](0=>1,1=>1);
# Vdummy=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((-3,1)=>1);
# V=GradedSpace[Irrep[U₁]⊠Irrep[ℤ₂]]((0,0)=>1,(1,1)=>1);
A_new = TensorMap(A_new, Vdummy ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V');

@assert norm(convert(Array,A_new)[1,:,:,:,:,:,:]-A)/norm(A)<1e-14
A=A_new; # dummy,P1,P2,L,R,D,U


U_phy1=unitary(fuse(space(A,1)⊗space(A,2)⊗space(A,3)), space(A,1)⊗space(A,2)⊗space(A,3));
@tensor A[:]:=A[1,2,3,-2,-3,-4,-5]*U_phy1[-1,1,2,3]; # P,L,R,D,U

#Add bond:both parity gate and bond operator
bond=zeros(1,2,2); bond[1,1,2]=1;bond[1,2,1]=1; bond=TensorMap(bond, ℂ[U1Irrep](1=>1) ← V ⊗ V);
gate=parity_gate(A,3); @tensor A[:]:=A[-1,-2,1,-4,-5]*gate[-3,1];
@tensor A[:]:=A[-1,-2,1,2,-5]*bond[-6,-3,1]*bond[-7,-4,2];
U_phy2=unitary(fuse(space(A,1)⊗space(A,6)⊗space(A,7)), space(A,1)⊗space(A,6)⊗space(A,7));
@tensor A[:]:=A[1,-2,-3,-4,-5,2,3]*U_phy2[-1,1,2,3];
#P,L,R,D,U





gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2];           
A=permute(A,(1,2,3,5,4,));#P,L,R,U,D

gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; 
A=permute(A,(1,2,4,3,5,));#P,L,U,R,D

gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; 
A=permute(A,(2,1,3,4,5,));#L,P,U,R,D

gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; 
A=permute(A,(1,3,2,4,5,));#L,U,P,R,D

#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


#convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));









A_fused=A;


conv_check="singular_value";
CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true);
@time CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(AA_fused,chi,conv_check,tol,CTM,CTM_ite_nums,CTM_trun_tol);



display(space(CTM["Cset"][1]))
display(space(CTM["Cset"][2]))
display(space(CTM["Cset"][3]))
display(space(CTM["Cset"][4]))

Ident, NA, NB, NANB, CAdag, CA, CBdag, CB=Hamiltonians(U_phy1,U_phy2)

O1=NA;
O2=Ident;
direction="x";
is_odd=false;
NA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=NB;
O2=Ident;
direction="x";
is_odd=false;
NB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=NANB;
O2=Ident;
direction="x";
is_odd=false;
NANB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

@tensor O1[:]:=CAdag[1,-1,2]*CB[1,2,-2];
O2=Ident;
direction="x";
is_odd=false;
CAdagCB_onsite=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)





O1=CAdag;
O2=CA;
direction="x";
is_odd=true;
CAdag_CA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=CAdag;
O2=CB;
direction="x";
is_odd=true;
CAdag_CB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=CBdag;
O2=CA;
direction="x";
is_odd=true;
CBdag_CA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

O1=CBdag;
O2=CB;
direction="x";
is_odd=true;
CBdag_CB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)

println("NA=   "*string(NA))
println("NB=   "*string(NB))
println("NANB=   "*string(NANB))
println("CAdagCB_onsite=   "*string(CAdagCB_onsite))

println("CAdag_CA=   "*string(CAdag_CA))
println("CAdag_CB=   "*string(CAdag_CB))
println("CBdag_CA=   "*string(CBdag_CA))
println("CBdag_CB=   "*string(CBdag_CB))





