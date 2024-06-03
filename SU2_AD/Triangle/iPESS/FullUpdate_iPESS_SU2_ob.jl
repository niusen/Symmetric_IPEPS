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
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_FullUpdate_iPESS.jl")
include("..\\..\\src\\fermionic\\simple_update\\Full_Update_lib.jl")
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

###########################
import LinearAlgebra.BLAS as BLAS
n_cpu=8;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C"*string(n_cpu)*"_"*"FU_U"*string(U)*"_D"*string(D_max))
pid=getpid();
println("pid="*string(pid));
###########################


function main(D_max,parameters)

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
optim_setting.init_statenm="nothing";#"SU_iPESS_SU2_csl_D6_3.93877.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

ENV_ctm_setting=LS_CTMRG_settings();
ENV_ctm_setting.CTM_conv_tol=1e-6;
ENV_ctm_setting.CTM_ite_nums=50;
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
projector_trun_tol=ENV_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;


ENV_ctm_setting.CTM_ite_nums=30;
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
    # λ_set1=data["λ_set1"];
    # λ_set2=data["λ_set2"];
    # λ_set3=data["λ_set3"];
end

# init_complex_tensor=true;
# state_vec=initial_fiPESS_spinful_SU2(optim_setting.init_statenm, optim_setting.init_noise,init_complex_tensor)
# state_vec=normalize_ansatz(state_vec);



A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,ENV_ctm_setting);
E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, ENV_ctm_setting, energy_setting);
println(E_total)
println(ex_set)
println(ey_set)
println(e_diagonala_set)
println(e0_set)
println(eU_set)

println("verifications:")
test_decomposition1(B_set, T_set,AA_cell,Lx,Ly);
test_decomposition2(B_set, T_set,AA_cell,CTM_cell,Lx,Ly);
test_decomposition3(B_set, T_set,AA_cell,CTM_cell,Lx,Ly,E_total);
test_positive_triangle_env(B_set, T_set,AA_cell,CTM_cell,Lx,Ly,E_total)
test_positive_triangle_env2(B_set, T_set,AA_cell,CTM_cell,Lx,Ly,E_total)




global save_filenm
save_filenm="FU_iPESS_LS_D_"*string(D_max)*"_chi_"*string(chi)*".jld2"

global starting_time
starting_time=now();

global E_history,E_all_history
E_history=[10000];
E_all_history=[10000];






trun_order="simultaneous";
trun_tol=1e-8;

n_sweep=10;




tau=20;
dt=0.1;
Bset, Tset=FullUpdate_iPESS(tau,dt,B_set, T_set,Lx,Ly, D_max, trun_order, trun_tol, n_sweep, ENV_ctm_setting)

tau=2;
dt=0.05;
Bset, Tset=FullUpdate_iPESS(tau,dt,B_set, T_set,Lx,Ly, D_max, trun_order, trun_tol, n_sweep, ENV_ctm_setting)

end



main(D_max,parameters)




