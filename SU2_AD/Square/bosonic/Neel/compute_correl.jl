using Revise, TensorKit, Zygote
using LinearAlgebra, OptimKit
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using LineSearches
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)

include("..\\..\\..\\src\\bosonic\\square\\square_spin_operator.jl")
include("..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\..\\src\\bosonic\\CTMRG_unitcell.jl")
include("..\\..\\..\\src\\bosonic\\square\\square_model.jl")
include("..\\..\\..\\src\\bosonic\\square\\square_model_cell.jl")
include("..\\..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\..\\src\\bosonic\\square\\simple_update_lib.jl")
include("..\\..\\..\\src\\bosonic\\square\\square_RVB_ansatz.jl")

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################

Random.seed!(1234)
symmetric_initial=false;

D_max=8;
J1=1;
J2=0;
Jchi=0;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);

symmetric_hosvd=false;
trun_tol=1e-6;


println("D_max= "*string(D_max))

chi=40;

"Unit-cell format:
ABABAB
CDCDCD
ABABAB
CDCDCD


A11  A21
A12  A22


actual unit-cell:
ABAB
BABA
ABAB
BABA
"

algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings

optim_setting=Optim_settings();
optim_setting.init_statenm="SU_SU2_D8_0.665389.jld2";#"Optim_cell_LS_D_4_chi_40_2.140901.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=50;
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

energy_setting=Square_Energy_settings();
energy_setting.model = "triangle_J1_J2_Jchi";
dump(energy_setting);


#################################
global Lx,Ly
Lx=2;
Ly=2;
##################################

data=load(optim_setting.init_statenm);
T_set=data["T_set"];
lambdax_set=data["lambdax_set"];
lambday_set=data["lambday_set"];

##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
LS_ctm_setting.CTM_ite_nums=50;

global chi, parameters, energy_setting, grad_ctm_setting
# A_cell=convert_to_iPEPS(Lx,Ly,T_set);
# init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
# CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell,chi,init,[],LS_ctm_setting);
# E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
# println(E_total/(Lx*Ly))





######################



A_cell=convert_to_iPEPS(Lx,Ly,T_set);



###############
#test for unitcell
# A11=A_cell[1][1];
# A21=A_cell[2][1];
# Vr=space(A11,3);
# Vr1=Vr⊕Rep[SU₂](0=>1)';
# UU=TensorMap(randn,Vr1,Vr);
# @tensor A11[:]:=A11[-1,-2,1,-4,-5]*UU[-3,1];
# @tensor A21[:]:=A21[1,-2,-3,-4,-5]*UU'[1,-1];
# A_cell=fill_tuple(A_cell, A11, 1,1);
# A_cell=fill_tuple(A_cell, A21, 2,1);
###############
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell,chi,init,[],LS_ctm_setting);
E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(E_total/(Lx*Ly))



distance=40;
cal_correl(CTM_cell,A_cell,AA_cell,D_max,chi,parameters,distance)





