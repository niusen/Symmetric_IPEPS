using Revise,  TensorKit, Zygote
using LinearAlgebra, OptimKit
#using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using LineSearches
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)

include("..\\..\\src\\bosonic\\square\\square_spin_operator.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\bosonic\\square\\square_model.jl")
include("..\\..\\src\\bosonic\\square\\square_AD_SU2.jl")
include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")


Random.seed!(555)


D=3;



J1=2;
J2=0;
Jchi=2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);


grad_ctm_setting=grad_CTMRG_settings();
grad_ctm_setting.CTM_conv_tol=1e-6;
grad_ctm_setting.CTM_ite_nums=150;
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

LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=250;
LS_ctm_setting.CTM_trun_tol=1e-8;
LS_ctm_setting.svd_lanczos_tol=1e-8;
LS_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
LS_ctm_setting.conv_check="singular_value";
LS_ctm_setting.CTM_ite_info=true;
LS_ctm_setting.CTM_conv_info=true;
LS_ctm_setting.CTM_trun_svd=false;
LS_ctm_setting.construct_double_layer=true;
LS_ctm_setting.grad_checkpoint=true;
dump(LS_ctm_setting);

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);

optim_setting=Optim_settings();
optim_setting.init_statenm="Optim_LS_D_4_chi_87.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_Energy_settings();
energy_setting.model = "triangle_J1_J2_Jchi";
dump(energy_setting);




global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings






global Vv

data=load("elementary_tensors.jld2");
A1=data["A1"];
A2=data["A2"];
Achiral=data["Achiral"];

A1=A1/norm(A1);
A2=A2/norm(A2);
Achiral=Achiral/norm(Achiral);

lamda1=-0.520724601304734;        
lamda2=1.70735449077426;
lamda_chiral=-im*0.668655573111120

chi=54;




A=lamda1*A1+lamda2*A2+lamda_chiral*Achiral;


U=unitary(space(A,1)',space(A,1));
@tensor A[:]:=A[-1,-2,1,2,-5]*U[-3,1]*U[-4,2];

state_vec=Square_iPEPS(A);

global starting_time
starting_time=now();



init_complex_tensor=true;
init_C4_symetry=true;





init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);




println("chi="*string(chi));
E, E_T1, E_T2, E_T3, E_T4, ite_num,ite_err,Init_CTM=energy_CTM(state_vec, chi, parameters, LS_ctm_setting, energy_setting, init, Init_CTM);
#E=cost_fun(state_vec);
println(E);



init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

CTM, AA, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A,chi,init,[],grad_ctm_setting);
E_LU_RU_LD, E_LD_RU_RD, E_LU_LD_RD, E_LU_RU_RD=evaluate_chirality(A, AA, U_L,U_D,U_R,U_U, CTM, LS_ctm_setting);

Ex,Ey=evaluate_NN(A, AA, U_L,U_D,U_R,U_U, CTM, LS_ctm_setting);




