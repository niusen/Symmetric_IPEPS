using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using LinearAlgebra, OptimKit
#using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON, MAT
using Random
using LineSearches
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)


include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\..\\..\\src\\bosonic\\CTMRG_unitcell.jl")
include("..\\..\\..\\..\\src\\bosonic\\square\\square_AD_SU2_cell.jl")
include("..\\..\\..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\..\\..\\src\\bosonic\\optimkit_lib.jl")

include("..\\..\\..\\..\\src\\mps_algorithms\\ES_algorithms.jl")

Random.seed!(555)


D=4;
chi=20;


Nv=4;
EH_n=30;
group_index=true;
vison=true;


#permute_neighbour_ind=bosonic_permute_neighbour_ind;
#permute_neighbour_ind=fermionic_permute_neighbour_ind;





grad_ctm_setting=grad_CTMRG_settings();
grad_ctm_setting.CTM_conv_tol=1e-6;
grad_ctm_setting.CTM_ite_nums=10;
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



backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);

optim_setting=Optim_settings();
optim_setting.init_statenm="Optim_cell_LS_D_4_chi_100.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);




global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings






global Vv

if D==4
    #Vv=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)';
    Vv=SU2Space(0=>2,1/2=>1);  
end
@assert dim(Vv)==D;



global Lx,Ly
Lx=2;
Ly=1;


init_complex_tensor=true;

state_vec=initial_SU2_state(Vv, optim_setting.init_statenm, optim_setting.init_noise,init_complex_tensor)
state_vec=normalize_tensor_group(state_vec);



A_cell=initial_tuple_cell(Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        A=state_vec[cx,cy].T;
        norm_A=norm(A)
        A=A/norm_A;

        A_cell=fill_tuple(A_cell, A, cx,cy);
    end
end

#############################

init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_cell,chi,init,[],grad_ctm_setting);




T2=CTM_cell.Tset[2][1].T2;
T4=CTM_cell.Tset[1][1].T4;

Tset=(T2=T2,T4=T4);
CTM=(Tset=Tset,);

global U_L,U_R
U_L=U_L_cell[1][1];
U_R=U_R_cell[1][1];


#ES_CTMRG_ED_Kprojector(CTM,D,chi,Nv,EH_n,group_index,vison);
ES_CTMRG_ED(CTM,D,chi,Nv,EH_n,group_index,vison)


