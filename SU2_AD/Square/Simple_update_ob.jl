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
include("..\\src\\CTMRG_unitcell.jl")
include("..\\src\\square_model.jl")
include("..\\src\\square_model_cell.jl")

include("..\\src\\Settings.jl")
include("..\\src\\Settings_cell.jl")
include("..\\src\\AD_lib.jl")
include("..\\src\\line_search_lib.jl")
include("..\\src\\line_search_lib_cell.jl")
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
J1=1.78;
J2=0.84;
Jchi=0.375*2*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
D_max=3;
symmetric_hosvd=false;
trun_tol=1e-6;


println("D_max= "*string(D_max))

chi=80;

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


##################################
"""
       /| s3
      / |
     /  |
    /   |
 s2 ----- s1
"""

H_Heisenberg, H123chiral, H12, H31, H23 =Hamiltonians();
H_triangle=(J1/4)*H31+(J1/4)*H12+(J2/2)*H23+Jchi*H123chiral;


##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;

A=RVB_ansatz(1,1,im);
state=Square_iPEPS(A);

TA=deepcopy(A);
TB=deepcopy(A);
TC=deepcopy(A);
TD=deepcopy(A);
λ_A_L=unitary(space(A,1)',space(A,1)');
λ_A_D=unitary(space(A,2)',space(A,2)'); 
λ_A_R=unitary(space(A,3)',space(A,3)');
λ_A_U=unitary(space(A,4)',space(A,4)');
λ_D_L=unitary(space(A,1)',space(A,1)');
λ_D_D=unitary(space(A,2)',space(A,2)'); 
λ_D_R=unitary(space(A,3)',space(A,3)');
λ_D_U=unitary(space(A,4)',space(A,4)');



######################

tau=5;
dt=0.1;
TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U=itebd(TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U, H_triangle, trun_tol, tau, dt, D_max);

tau=2;
dt=0.05;
TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U=itebd(TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U, H_triangle, trun_tol, tau, dt, D_max);

tau=0.2;
dt=0.01;
TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U=itebd(TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U, H_triangle, trun_tol, tau, dt, D_max);


println(space(TA))
println(space(TB))
println(space(TC))
println(space(TD))

##############
@tensor TA[:]:=TA[1,2,3,4,-5]*λ_A_L[1,-1]*λ_A_D[2,-2]*λ_A_R[3,-3]*λ_A_U[4,-4];
@tensor TD[:]:=TD[1,2,3,4,-5]*λ_D_L[1,-1]*λ_D_D[2,-2]*λ_D_R[3,-3]*λ_D_U[4,-4];

##############
state_vec=Matrix{Square_iPEPS}(undef,2,2);
state_vec[1,1]=Square_iPEPS(TA);
state_vec[1,2]=Square_iPEPS(TB);
state_vec[2,1]=Square_iPEPS(TC);
state_vec[2,2]=Square_iPEPS(TD);

##############


global Lx,Ly,A_cell
A_cell=initial_tuple_cell(Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        global U_phy,A_cell
        A=state_vec[cx, cy].T;
        A_cell=fill_tuple(A_cell, A, cx,cy);
    end
end

global chi, parameters, energy_setting, grad_ctm_setting
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell,chi,init,[],LS_ctm_setting);

include("..\\src\\square_model_cell.jl")
E_total,  E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);



filenm="SU_D_"*string(D_max)*".jld2"
jldsave(filenm;x=state_vec);

# mat_filenm="SU_D_"*string(D_max)*".mat"
# matwrite(mat_filenm, Dict(
#     "E_plaquatte_cell" => E_plaquatte_cell,
#     "space_Tu" => string(codomain(T_u)),
#     "space_Td" => string(codomain(T_d))
# ); compress = false)


