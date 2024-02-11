using Revise, TensorKit
using LinearAlgebra:diag,I,diagm 
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
using Random
using LineSearches,OptimKit
using Dates
cd(@__DIR__)

include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_correl.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD.jl")

Random.seed!(777)

M=1;
chi=40

t=1;
ϕ=pi/2;
μ=0;
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ)]);



grad_ctm_setting=grad_CTMRG_settings();
grad_ctm_setting.CTM_conv_tol=1e-6;
grad_ctm_setting.CTM_ite_nums=50;
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
LS_ctm_setting.CTM_ite_nums=50;
LS_ctm_setting.CTM_trun_tol=1e-8;
LS_ctm_setting.svd_lanczos_tol=1e-8;
LS_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
LS_ctm_setting.conv_check="singular_value";
LS_ctm_setting.CTM_ite_info=false;
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
optim_setting.init_statenm="nothing";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "spinless_triangle_lattice";
dump(energy_setting);



global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings






################################################
if M==1
    data=load("parton_state_M1.jld2")
    A=data["A"];   #P1,P2,L,R,D,U
elseif M==2
    data=load("parton_state_M2.jld2")
    A=data["A"];   #P1,P2,L,R,D,U
end


################################################
println("to get correct energy the following gauge transformation is required")
gauge_gate1=gauge_gate(A,4,-pi/4);
@tensor A[:]:=A[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
################################################
jldsave("parton_tensor_M"*string(M)*".jld2";A)
################################################


init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM, AA, U_L,U_D,U_R,U_U,ite_num,ite_err=fermi_CTMRG(A,chi,init,[],grad_ctm_setting);


Ident4, NA, NB, n_double_A, n_double_B,  CdagA_CB, Cdag_A, C_A, Cdag_B, C_B = @ignore_derivatives Hamiltonians_spinless_U1_SU2_2site(M);
ex1=hopping_x(CTM,Cdag_B,C_A,A,AA,grad_ctm_setting);
ex2=ob_onsite(CTM,CdagA_CB,A,AA,grad_ctm_setting);

ey1=hopping_y(CTM,Cdag_A,C_A,A,AA,grad_ctm_setting);
ey2=hopping_y(CTM,Cdag_B,C_B,A,AA,grad_ctm_setting);


e_right_top1=hopping_right_top(CTM,Cdag_B,C_A,A,AA,grad_ctm_setting);
e_right_top2=hopping_y(CTM,Cdag_A,C_B,A,AA,grad_ctm_setting);


E=im*ex1+im*ex2+ey1-ey2+e_right_top1-e_right_top2;
E=E+E';
dE=E/2+2.4020
println("dE= "*string(dE));


direction="x";
distance=20;
CAdag_CA_ob,CAdag_CB_ob,CBdag_CA_ob,CBdag_CB_ob=cal_correl(direction,M,A, AA, chi,CTM, distance,grad_ctm_setting);

direction="y";
distance=20;
CAdag_CA_ob,CAdag_CB_ob,CBdag_CA_ob,CBdag_CB_ob=cal_correl(direction,M,A, AA, chi,CTM, distance,grad_ctm_setting);