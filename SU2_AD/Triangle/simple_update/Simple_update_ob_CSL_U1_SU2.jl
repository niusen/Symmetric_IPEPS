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
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate.jl")

# let
###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################

Random.seed!(888)
symmetric_initial=false;

D_max=8;

t1=1;
t2=1;
ϕ=pi/2;
μ=0;
U=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);
parameters_evolve=Dict([("t1", t1),("t2", t2*1), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);

println("parameters:")
println(parameters)
println("parameters_evolve:")
println(parameters_evolve)

global update_triangle1, update_triangle2
update_triangle1=true;
update_triangle2=false;
println("update_triangle1: "*string(update_triangle1));
println("update_triangle2: "*string(update_triangle2));



global D_max, SU_trun_tol
SU_trun_tol=1e-8;
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
optim_setting.init_statenm="SU_U1_SU2_csl_D6.jld2";#"Optim_cell_LS_D_4_chi_40_2.140901.jld2";#"nothing";
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
energy_setting.model = "spinful_triangle_lattice";
dump(energy_setting);




##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;

#######################
VDummytype=2;
global VDummy_set

if VDummytype==1
    VDummy1=Rep[U₁ × SU₂]((-1, 1/2)=>1);
    VDummy2=Rep[U₁ × SU₂]((-1, 1/2)=>1);
    VDummy_set=(VDummy1,VDummy2,);
elseif VDummytype==2
    VDummy1=Rep[U₁ × SU₂]((-2, 0)=>1);
    VDummy2=ProductSpace{GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}, 0}();
    VDummy_set=(VDummy1,VDummy2,);
end
############################

if optim_setting.init_statenm=="nothing"
    Vp=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1);
    Vv1=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1);
    Vv3=Rep[U₁ × SU₂]((-1, 1/2)=>1, (0, 0)=>2, (1, 1/2)=>1)';
    Vv_set=((Vv1,Vv1',Vv3,Vv1,),(Vv3',Vv1',Vv1',Vv1,),);
    T_set,lambdax_set,lambday_set=initial_iPEPS_U1_SU2(Lx,Ly,Vp,Vv_set);
else
    data=load(optim_setting.init_statenm);
    T_set=data["T_set"];
    lambdax_set=data["lambdax_set"];
    lambday_set=data["lambday_set"];
end


##############
LS_ctm_setting.CTM_ite_nums=10;



global chi, parameters, energy_setting, grad_ctm_setting
A_cell=convert_to_iPEPS(Lx,Ly,T_set);
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);
E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(E_total)


# tau=30;
# dt=0.1;
# T_set,lambdax_set,lambday_set=itebd(parameters_evolve, T_set,lambdax_set,lambday_set, tau, dt, D_max);


# tau=4;
# dt=0.05;
# T_set,lambdax_set,lambday_set=itebd(parameters_evolve, T_set,lambdax_set,lambday_set, tau, dt, D_max);

tau=20;
dt=0.01;
T_set,lambdax_set,lambday_set=itebd(parameters_evolve, T_set,lambdax_set,lambday_set, tau, dt, D_max);

tau=20;
dt=0.002;
T_set,lambdax_set,lambday_set=itebd(parameters_evolve, T_set,lambdax_set,lambday_set, tau, dt, D_max);



A_cell=convert_to_iPEPS(Lx,Ly,T_set);
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);
E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(E_total)
println(ex_set)
println(ey_set)
println(e_diagonala_set)
println(e0_set)
println(eU_set)



# filenm="SU_U1_SU2_csl_D"*string(D_max)*".jld2";
# jldsave(filenm;T_set,lambdax_set,lambday_set)


# end




