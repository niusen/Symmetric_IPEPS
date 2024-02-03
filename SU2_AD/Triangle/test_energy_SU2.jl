using Revise, TensorKit
using LinearAlgebra, OptimKit
using TensorKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
using Random
using LineSearches
using Dates
cd(@__DIR__)

include("..\\src\\bosonic\\Settings.jl")
include("..\\src\\bosonic\\Settings_cell.jl")
include("..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\src\\bosonic\\AD_lib.jl")
include("..\\src\\bosonic\\line_search_lib.jl")
include("..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\src\\bosonic\\optimkit_lib.jl")
include("..\\src\\bosonic\\CTMRG.jl")
include("..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\src\\fermionic\\swap_funs.jl")
include("..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\src\\fermionic\\double_layer_funs.jl")
include("..\\src\\fermionic\\square_Hubbard_AD_cell.jl")

Random.seed!(555)

D=4;
chi=20

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
energy_setting.model = "spinful_triangle_lattice";
dump(energy_setting);

algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings


global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings




if D==4
    Vv=SU2Space(0=>2,1/2=>1);
elseif D==4
   
elseif D==6
     
end
@assert dim(Vv)==D;





global Lx,Ly
Lx=2;
Ly=1;






init_complex_tensor=true;

state_vec=initial_fPEPS_state_spinful_SU2(Vv_set, optim_setting.init_statenm, optim_setting.init_noise,init_complex_tensor)
state_vec=normalize_tensor_group(state_vec);

A1=state_vec[1].T;
A2=state_vec[2].T;


global save_filenm
save_filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"

global starting_time
starting_time=now();

################################################




A_cell=initial_tuple_cell(Lx,Ly);
A1=A1/norm(A1);
A2=A2/norm(A2);
A_cell=fill_tuple(A_cell, A1, 1,1);
A_cell=fill_tuple(A_cell, A2, 2,1);


init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init,[],grad_ctm_setting);


E_total,  ex_set, ey_set, e_right_bot, e0_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
#E_total,  ex_set, ey_set, e0_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);





