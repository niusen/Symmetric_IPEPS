using LinearAlgebra
using TensorKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
cd(@__DIR__)

include("..\\..\\src\\Settings.jl")
include("..\\..\\src\\AD_lib.jl")
include("..\\..\\src\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\square_1site_model.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")


chi=100



grad_ctm_setting=grad_CTMRG_settings();
grad_ctm_setting.CTM_conv_tol=1e-6;
grad_ctm_setting.CTM_ite_nums=100;
grad_ctm_setting.CTM_trun_tol=1e-8;
grad_ctm_setting.svd_lanczos_tol=1e-8;
grad_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
grad_ctm_setting.conv_check="singular_value";
grad_ctm_setting.CTM_ite_info=true;
grad_ctm_setting.CTM_conv_info=true;
grad_ctm_setting.CTM_trun_svd=false;
grad_ctm_setting.construct_double_layer=true;
grad_ctm_setting.grad_checkpoint=true;
dump(grad_ctm_setting);

global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol


# Vv=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(1,1/2)=>1,(2,0)=>1);
# Vp=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1);
Vv=SU2Space(0=>2,1/2=>1);
Vp=SU2Space(1/2=>1);
A=TensorMap(randn,Vv'*Vv*Vv*Vv',Vp)+TensorMap(randn,Vv'*Vv*Vv*Vv',Vp)*im;
A=permute(A,(1,2,3,4,5,));

H_Heisenberg, H123chiral_tensorkit, H12_tensorkit, H31_tensorkit, H23_tensorkit =Hamiltonians();



u,s,v=tsvd(H123chiral_tensorkit,(1,4,),(2,3,5,6,));
O1=u;
AA_op1=build_double_layer_swap_op(A,O1,true);
AA_open,U_p=build_double_layer_swap_open(A',A);
@tensor AA_op2[:]:=AA_open[-1,-2,-3,-4,1]*U_p'[2,3,1]*O1[2,3,-5];
println(norm(AA_op1-AA_op2)/norm(AA_op1))





