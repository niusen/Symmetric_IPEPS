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
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate_iPESS.jl")

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################

Random.seed!(1234)


D_max=4;

t1=1;
t2=1;
ϕ=pi/2;
μ=0;
U=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);

global expon
expon=1;

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
optim_setting.init_statenm="nothing";#"Optim_cell_LS_D_4_chi_40_2.140901.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);


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






##############


global Lx,Ly,A_cell



global chi, parameters, energy_setting, grad_ctm_setting



function initial_iPESS()
    V=Rep[SU₂](0=>1, 1/2=>1);
    Vp=Rep[SU₂](0=>2, 1/2=>1);
    BA=permute(TensorMap(randn,V*Vp,V*V),(1,),(2,3,4,));
    TA=TensorMap(randn,V*V,V);
    BB=permute(TensorMap(randn,V*Vp,V*V),(1,),(2,3,4,));
    TB=TensorMap(randn,V*V,V);
    BC=permute(TensorMap(randn,V*Vp,V*V),(1,),(2,3,4,));
    TC=TensorMap(randn,V*V,V);
    BD=permute(TensorMap(randn,V*Vp,V*V),(1,),(2,3,4,));
    TD=TensorMap(randn,V*V,V);
    b_A=BA;
    t_A=TA;
    b_B=BB;
    t_B=TB;
    b_C=BC;
    t_C=TC;
    b_D=BD;
    t_D=TD;
    λ_A_1=unitary(space(t_A,1)',space(t_A,1)');
    λ_A_2=unitary(space(t_A,2)',space(t_A,2)');
    λ_A_3=unitary(space(t_A,3)',space(t_A,3)');
    λ_B_1=unitary(space(t_B,1)',space(t_B,1)');
    λ_B_2=unitary(space(t_B,2)',space(t_B,2)');
    λ_B_3=unitary(space(t_B,3)',space(t_B,3)');
    λ_C_1=unitary(space(t_C,1)',space(t_C,1)');
    λ_C_2=unitary(space(t_C,2)',space(t_C,2)');
    λ_C_3=unitary(space(t_C,3)',space(t_C,3)');
    λ_D_1=unitary(space(t_D,1)',space(t_D,1)');
    λ_D_2=unitary(space(t_D,2)',space(t_D,2)');
    λ_D_3=unitary(space(t_D,3)',space(t_D,3)');
    return b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D, λ_A_1, λ_A_2, λ_A_3, λ_B_1, λ_B_2, λ_B_3, λ_C_1, λ_C_2, λ_C_3, λ_D_1, λ_D_2, λ_D_3
end


function compute_E(b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D)
    global Lx,Ly
    A_A=iPESS_to_iPEPS(Triangle_iPESS(b_A,t_A));
    A_B=iPESS_to_iPEPS(Triangle_iPESS(b_B,t_B));
    A_C=iPESS_to_iPEPS(Triangle_iPESS(b_C,t_C));
    A_D=iPESS_to_iPEPS(Triangle_iPESS(b_D,t_D));

    A_cell_iPEPS=initial_tuple_cell(Lx,Ly);
    A_cell_iPEPS=fill_tuple(A_cell_iPEPS, A_A.T, 1,1);
    A_cell_iPEPS=fill_tuple(A_cell_iPEPS, A_B.T, 2,1);
    A_cell_iPEPS=fill_tuple(A_cell_iPEPS, A_C.T, 1,2);
    A_cell_iPEPS=fill_tuple(A_cell_iPEPS, A_D.T, 2,2);

    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,LS_ctm_setting);
    E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
    println(E_total)
    println(ex_set)
    println(ey_set)
    println(e_diagonala_set)
    println(e0_set)
    println(eU_set)
end



b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D, λ_A_1, λ_A_2, λ_A_3, λ_B_1, λ_B_2, λ_B_3, λ_C_1, λ_C_2, λ_C_3, λ_D_1, λ_D_2, λ_D_3=initial_iPESS();
# include("..\\..\\src\\fermionic\\swap_funs.jl")
# include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate_iPESS.jl")
compute_E(b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D);

tau=5;
dt=0.1;
b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D, λ_A_1, λ_A_2, λ_A_3, λ_B_1, λ_B_2, λ_B_3, λ_C_1, λ_C_2, λ_C_3, λ_D_1, λ_D_2, λ_D_3 = itebd(parameters, b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D, λ_A_1, λ_A_2, λ_A_3, λ_B_1, λ_B_2, λ_B_3, λ_C_1, λ_C_2, λ_C_3, λ_D_1, λ_D_2, λ_D_3, tau, dt, trun_tol);
compute_E(b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D);

tau=4;
dt=0.05;
b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D, λ_A_1, λ_A_2, λ_A_3, λ_B_1, λ_B_2, λ_B_3, λ_C_1, λ_C_2, λ_C_3, λ_D_1, λ_D_2, λ_D_3 = itebd(parameters, b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D, λ_A_1, λ_A_2, λ_A_3, λ_B_1, λ_B_2, λ_B_3, λ_C_1, λ_C_2, λ_C_3, λ_D_1, λ_D_2, λ_D_3, tau, dt, trun_tol);
compute_E(b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D);

tau=4;
dt=0.01;
b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D, λ_A_1, λ_A_2, λ_A_3, λ_B_1, λ_B_2, λ_B_3, λ_C_1, λ_C_2, λ_C_3, λ_D_1, λ_D_2, λ_D_3 = itebd(parameters, b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D, λ_A_1, λ_A_2, λ_A_3, λ_B_1, λ_B_2, λ_B_3, λ_C_1, λ_C_2, λ_C_3, λ_D_1, λ_D_2, λ_D_3, tau, dt, trun_tol);
compute_E(b_A, b_B, b_C, b_D, t_A, t_B, t_C, t_D);



