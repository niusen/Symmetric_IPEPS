using LinearAlgebra:diag,I,diagm 
using JLD2,ChainRulesCore,MAT, Zygote
using Zygote:@ignore_derivatives
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)



include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\mps_algorithms\\Projector_funs.jl")
include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")


filenm="stochastic_iPESS_LS_D_6_chi_40_3.4307.jld2";
data=load(filenm);
# A=data["x"][1].T;
# B=data["x"][2].T;
state=data["x"]
#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings

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


global Lx,Ly
Lx,Ly=size(state);
A_cell=initial_tuple_cell(Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        if isa(state[ca,cb],Square_iPEPS)
            A_cell=fill_tuple(A_cell, state[ca,cb].T, ca,cb);
        elseif isa(state[ca,cb],Triangle_iPESS)
            A0=iPESS_to_iPEPS(state[ca,cb]).T;
            A0=A0/norm(A0)*10;
            A_cell=fill_tuple(A_cell, A0, ca,cb);
        else
            error("unknown type ansatz")
        end
    end
end

##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################


chi=40;
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);
hop_x_set,hop_y_set,hop_diagonala_set,occu_set,doublon_set,holon_set,cdagupcdagdn_set,pairing_x_set,pairing_y_set,pairing_diagonala_set,ss_x_set,ss_y_set,ss_diagonala_set=evaluate_all_ob_cell(A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
1+1
#############################
