using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit

using LinearAlgebra, OptimKit
#using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using LineSearches
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)

include("..\\src\\square_spin_operator.jl")
include("..\\src\\iPEPS_ansatz.jl")
include("..\\src\\CTMRG.jl")
#include("..\\src\\CTMRG_unitcell.jl")
include("..\\src\\square_model.jl")
#include("..\\src\\square_model_cell.jl")
#include("..\\src\\square_AD_SU2_cell.jl")
include("..\\src\\Settings.jl")
#include("..\\src\\Settings_cell.jl")
include("..\\src\\AD_lib.jl")
include("..\\src\\line_search_lib.jl")
#include("..\\src\\line_search_lib_cell.jl")
include("..\\src\\optimkit_lib.jl")

include("..\\src\\square_SimpleUpdate_lib.jl")
include("..\\src\\square_RVB_ansatz.jl")

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

J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);

symmetric_hosvd=false;
trun_tol=1e-6;

D=3;
chi=40;


# algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
# algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
# dump(algrithm_CTMRG_settings);
# global algrithm_CTMRG_settings

LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=150;
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





global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
A=RVB_ansatz(1,1,im);
state=Square_iPEPS(A);


######################







##############




global chi, parameters, energy_setting, grad_ctm_setting
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);


# Vp=SU2Space(1/2=>1);
# Vv=SU2Space(0=>1,1/2=>1);
# Vv2=SU2Space(0=>1,1/2=>2);

# A=TensorMap(randn,Vv*Vv2,Vv'*Vv2'*Vp);A=permute(A,(1,2,3,4,5,));

CTM, AA, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A,chi,init,[],LS_ctm_setting)




Ex,Ey=evaluate_NN(A::TensorMap, AA, U_L,U_D,U_R,U_U, CTM, LS_ctm_setting);
E_LD_RU,E_LU_RD=evaluate_NNN(A::TensorMap, AA, U_L,U_D,U_R,U_U, CTM, LS_ctm_setting);
E_LU_RU_LD, E_LD_RU_RD, E_LU_LD_RD, E_LU_RU_RD=evaluate_chirality(A::TensorMap, AA, U_L,U_D,U_R,U_U, CTM, LS_ctm_setting);


println([real(Ex),real(Ey)])
println([real(E_LD_RU),real(E_LU_RD)])
println([real(E_LU_RU_LD), real(E_LD_RU_RD), real(E_LU_LD_RD), real(E_LU_RU_RD)])

Ex*J1+Ey*J1+E_LD_RU*J2*2+E_LU_RU_LD*Jchi*4


H_Heisenberg, H123chiral, H12, H31, H23 =@ignore_derivatives Hamiltonians();
H_triangle=J1/4*(H12+H31)+J2/2*H23+Jchi*H123chiral;
E_T1, E_T2, E_T3, E_T4=evaluate_triangle(H_triangle, A::TensorMap, AA, U_L,U_D,U_R,U_U, CTM, LS_ctm_setting);

E_T1*4
