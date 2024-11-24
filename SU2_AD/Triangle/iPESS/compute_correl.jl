using Revise
using LinearAlgebra:diag,I,diagm 
using TensorKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Zygote:@ignore_derivatives
using Random
using LineSearches,OptimKit
using Dates
cd(@__DIR__)


include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\src\\bosonic\\stochastic_opt.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate_iPESS.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_correl_cell.jl")

D=4;
chi=40

t=1;
ϕ=pi/2;
μ=0;
U=15;
parameters=Dict([("t1", t),("t2", t), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);



LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=30;
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

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);

optim_setting=Optim_settings();
optim_setting.init_statenm="stochastic_iPESS_LS_D_8_chi_80_4.05745.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
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
projector_trun_tol=LS_ctm_setting.CTM_trun_tol

global backward_settings









init_complex_tensor=true;

state_vec=initial_fiPESS_spinful_SU2(optim_setting.init_statenm, optim_setting.init_noise,init_complex_tensor)
global Lx,Ly
Lx,Ly=size(state_vec);
state_vec=state_vec[1:Lx,1:Ly]
state_vec=normalize_ansatz(state_vec);




global chi, parameters, energy_setting, grad_ctm_setting

######################
A_cell=initial_tuple_cell(Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        A_cell=fill_tuple(A_cell, iPESS_to_iPEPS(state_vec[cx,cy]).T, cx,cy);
        # A=A_cell[cx][cy];
        # AA_,_=build_double_layer_swap(A',A);
        # println(norm(AA_))

    end
end

# data=load(optim_setting.init_statenm);
# x=data["x"];
# for cx=1:Lx
#     for cy=1:Ly
#         A=iPESS_to_iPEPS(x[cx,cy]).T;
#         AA_,_=build_double_layer_swap(A',A);
#         println(norm(AA_))
#     end
# end

###############
# # test for unitcell
# A11=A_cell[1][1];
# A21=A_cell[2][1];
# Vr=space(A11,3);
# Vr1=Vr⊕Rep[SU₂](0=>1)';
# UU=TensorMap(randn,Vr1,Vr);
# @tensor A11[:]:=A11[-1,-2,1,-4,-5]*UU[-3,1];
# @tensor A21[:]:=A21[1,-2,-3,-4,-5]*UU'[1,-1];
# A_cell=fill_tuple(A_cell, A11, 1,1);
# A_cell=fill_tuple(A_cell, A21, 2,1);

# A11=A_cell[1][1];
# A12=A_cell[1][2];
# Vd=space(A11,2);
# Vd1=Vd⊕Rep[SU₂](0=>1)';
# UU=TensorMap(randn,Vd1,Vd);
# @tensor A11[:]:=A11[-1,1,-3,-4,-5]*UU[-2,1];
# @tensor A12[:]:=A12[-1,-2,-3,1,-5]*UU'[1,-4];
# A_cell=fill_tuple(A_cell, A11, 1,1);
# A_cell=fill_tuple(A_cell, A12, 1,2);
###############


init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);
E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(E_total)
println(ex_set)
println(ey_set)
println(e_diagonala_set)
println(e0_set)
println(eU_set)

# jldname="CTM_iPESS_D"*string(D_max_)*"_chi"*string(chi)*".jld2";
# data=load(jldname);
# CTM_cell=data["CTM_cell"];



distance=40;
D=dim(space(A_cell[1][1],1));
direction="x";
SS_ob_set,CdagC_ob_set=cal_correl(CTM_cell,A_cell,AA_cell,D,chi,parameters,direction,distance)





