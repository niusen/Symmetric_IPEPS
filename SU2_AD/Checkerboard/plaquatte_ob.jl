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

include("..\\src\\checkerboard_spin_operator.jl")
include("..\\src\\iPEPS_ansatz.jl")
include("..\\src\\CTMRG.jl")
include("..\\src\\CTMRG_unitcell.jl")
include("..\\src\\checkerboard_model_cell.jl")
include("..\\src\\checkerboard_AD_SU2_cell.jl")
include("..\\src\\Settings.jl")
include("..\\src\\Settings_cell.jl")
include("..\\src\\AD_lib.jl")
include("..\\src\\line_search_lib.jl")
include("..\\src\\line_search_lib_cell.jl")
include("..\\src\\optimkit_lib.jl")

include("..\\src\\checkerboard_SimpleUpdate_lib.jl")
include("..\\src\\checkerboard_plaquatte_ansatz.jl")




Random.seed!(1234)
symmetric_initial=false;
J1=1;
J2=1;
parameters=Dict([("J1", J1), ("J2", J2)]);

symmetric_hosvd=false;
trun_tol=1e-6;
D=12;



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

optim_setting=Optim_settings();
optim_setting.init_statenm="Optim_cell_LS_D_12_chi_40.jld2";#"D2_state.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);


global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;

#state_vec=plaquatte_empty()
#state_vec=plaquatte_cross()
###################################

global Vv

if D==2
    Vv=SU2Space(1/2=>1);
elseif D==6
    Vv=SU2Space(1/2=>1,3/2=>1);
elseif D==12
    Vv=SU2Space(1/2=>2,3/2=>2);
end
@assert dim(Vv)==D;


state_vec=initial_SU2_state(Vv, optim_setting.init_statenm, optim_setting.init_noise)
state_vec=normalize_tensor_group(state_vec);
######################







##############


global Lx,Ly,U_phy,A_unfused_cell,A_fused_cell
A_unfused_cell=initial_tuple_cell(Lx,Ly);
A_fused_cell=initial_tuple_cell(Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        global U_phy,A_unfused_cell,A_fused_cell
        A_unfused,A_fused,U_phy=build_A_checkerboard(state_vec[cx, cy]);
        A_unfused_cell=fill_tuple(A_unfused_cell, A_unfused, cx,cy);
        A_fused_cell=fill_tuple(A_fused_cell, A_fused, cx,cy);
    end
end

global chi, parameters, energy_setting, grad_ctm_setting
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_fused_cell,chi,init,[],LS_ctm_setting)

E_plaquatte_cell=bond_energy(U_phy, state_vec, A_fused_cell::Tuple, AA_fused_cell, CTM_cell, LS_ctm_setting);



mat_filenm="ob_D_"*string(D)*".mat"
matwrite(mat_filenm, Dict(
    "E_plaquatte_cell" => E_plaquatte_cell
); compress = false)


