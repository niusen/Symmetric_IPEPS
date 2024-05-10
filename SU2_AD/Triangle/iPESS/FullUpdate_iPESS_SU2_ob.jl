using Revise, TensorKit, Zygote
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)




include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate_iPESS.jl")

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################
# let
Random.seed!(1234);
import LinearAlgebra.BLAS as BLAS
# n_cpu=10;
# BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))

D_max=4;

t1=1;
t2=1;
ϕ=pi/2;
μ=0;
U=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);



trun_tol=1e-6;



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
optim_setting.init_statenm="nothing";#"SU_iPESS_SU2_csl_D4.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

ENV_ctm_setting=LS_CTMRG_settings();
ENV_ctm_setting.CTM_conv_tol=1e-6;
ENV_ctm_setting.CTM_ite_nums=10;
ENV_ctm_setting.CTM_trun_tol=1e-8;
ENV_ctm_setting.svd_lanczos_tol=1e-8;
ENV_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
ENV_ctm_setting.conv_check="singular_value";
ENV_ctm_setting.CTM_ite_info=true;
ENV_ctm_setting.CTM_conv_info=true;
ENV_ctm_setting.CTM_trun_svd=false;
ENV_ctm_setting.construct_double_layer=true;
ENV_ctm_setting.grad_checkpoint=true;
dump(ENV_ctm_setting);


LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=10;
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

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "spinful_triangle_lattice";
dump(energy_setting);



##################################
"""
       /| s3
      / |
     /  |
    /   |
 s2 ----- s1
"""


##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;


LS_ctm_setting.CTM_ite_nums=30;
##############


global Lx,Ly,A_cell
global chi, parameters, energy_setting, grad_ctm_setting


if optim_setting.init_statenm=="nothing"
    V=Rep[SU₂](0=>2, 1/2=>1);
    Vp=Rep[SU₂](0=>2, 1/2=>1);
    B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS(Lx,Ly,V,Vp); 
    # B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS_uniform(Lx,Ly,V,Vp);    
else
    data=load(optim_setting.init_statenm);
    T_set=data["T_set"];
    B_set=data["B_set"];
    λ_set1=data["λ_set1"];
    λ_set2=data["λ_set2"];
    λ_set3=data["λ_set3"];
end



A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,LS_ctm_setting);
E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(E_total)
println(ex_set)
println(ey_set)
println(e_diagonala_set)
println(e0_set)
println(eU_set)




# tau=20;
# dt=0.1;
# B_set, T_set = FullUpdate(parameters, B_set, T_set, tau, dt,D_max, trun_tol,ENV_ctm_setting);

# tau=20;
# dt=0.05;
# B_set, T_set = FullUpdate(parameters, B_set, T_set, tau, dt,D_max, trun_tol,ENV_ctm_setting);

tau=20;
dt=0.01;
B_set, T_set = FullUpdate(parameters, B_set, T_set, tau, dt,D_max, trun_tol,ENV_ctm_setting);

tau=20;
dt=0.002;
B_set, T_set = FullUpdate(parameters, B_set, T_set, tau, dt,D_max, trun_tol,ENV_ctm_setting);



A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,LS_ctm_setting);
E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(E_total)
println(ex_set)
println(ey_set)
println(e_diagonala_set)
println(e0_set)
println(eU_set)


# filenm="SU_iPESS_SU2_csl_D"*string(D_max)*".jld2";
# jldsave(filenm; B_set, T_set, λ_set1, λ_set2, λ_set3)


# end
