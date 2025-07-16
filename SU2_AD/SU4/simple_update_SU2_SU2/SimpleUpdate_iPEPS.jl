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
include("..\\..\\src\\bosonic\\CTMRG_unitcell.jl")
include("..\\..\\src\\bosonic\\triangle\\triangle_spin_model_cell.jl")
include("..\\..\\src\\bosonic\\triangle\\triangle_spin_AD_cell.jl")
include("..\\..\\src\\bosonic\\triangle\\simple_update\\triangle_SimpleUpdate.jl")
include("..\\..\\src\\bosonic\\triangle\\simple_update\\triangle_SimpleUpdate_iPEPS.jl")

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################
let
Random.seed!(1234);
import LinearAlgebra.BLAS as BLAS
n_cpu=6;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))

Dmax=9;


J=1;
K=0.1;
Φ=0.1;
@show parameters=Dict([("J", J),("K", K), ("Φ",  Φ)]);



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
optim_setting.init_statenm="nothing";#"Gutzwiller_stochastic_iPESS_LS_D_6_chi_40_2.3592.jld2";#"nothing";
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

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "triangle_SU4_spin";
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



##############


global Lx,Ly,A_cell
global chi, parameters, energy_setting, grad_ctm_setting


if optim_setting.init_statenm=="nothing"
    Vv=Rep[SU₂ × SU₂]((0,0)=>1, (0,1/2)=>1, (1/2,0)=>1, (1/2,1/2)=>1);
    Vp=Rep[SU₂ × SU₂]((1/2,1/2)=>1);
    T_set,lambdax_set,lambday_set=initial_iPEPS(Lx,Ly,Vp,Vv); 
    # B_set, T_set, λ_set1, λ_set2, λ_set3=initial_iPESS_uniform(Lx,Ly,V,Vp);    
else
    data=load(optim_setting.init_statenm);

    
    T_set=data["T_set"];
    lambdax_set=data["lambdax_set"];
    lambday_set=data["lambday_set"];
    
end








# A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
# init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
# CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,LS_ctm_setting);
# E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
# println(E_total)
# println(ex_set)
# println(ey_set)
# println(e_diagonala_set)
# println(e0_set)
# println(eU_set)

# D0set=[];
# for cc in eachindex(B_set)
#     D0set=vcat(D0set,[dim(space(B_set[cc],1)), dim(space(B_set[cc],2)), dim(space(B_set[cc],3))]);
# end
# D_max0=maximum(D0set);
# B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS_no_Hamiltonian(parameters, B_set, T_set, λ_set1, λ_set2, λ_set3, D_max0, trun_tol);

tau=1;
dt=0.1;
T_set,lambdax_set,lambday_set=@ignore_derivatives simple_update_triangle(parameters,T_set,lambdax_set,lambday_set,tau,dt,Dmax);


# tau=20;
# dt=0.02;
# #B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS(parameters, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);
# B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS_Hofstadter(energy_setting, parameters, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);

# tau=20;
# dt=0.002;
# B_set, T_set, λ_set1, λ_set2, λ_set3 = itebd_iPESS(parameters, B_set, T_set, λ_set1, λ_set2, λ_set3, tau, dt,D_max, trun_tol);

filenm="SU_iPEPS_D"*string(Dmax)*".jld2";
jldsave(filenm; T_set,lambdax_set,lambday_set)

A_cell_iPEPS=initial_tuple_cell(Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        A_cell_iPEPS=fill_tuple(A_cell_iPEPS, T_set[cx,cy], cx,cy);
    end
end
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,LS_ctm_setting);
E_total,  ex_set, ey_set, e_diagonal_set, triangle_right_bot_set, triangle_left_top_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(E_total)
println(ex_set)
println(ey_set)
println(e_diagonal_set)
println(triangle_right_bot_set)
println(triangle_left_top_set)





# end


end
