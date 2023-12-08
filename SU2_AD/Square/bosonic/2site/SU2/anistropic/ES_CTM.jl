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

include("..\\..\\..\\..\\..\\src\\bosonic\\square\\square_spin_operator.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\square\\square_2site_model.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\square\\square_AD_2site.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\..\\..\\..\\src\\bosonic\\optimkit_lib.jl")
# include("..\\..\\..\\..\\..\\src\\mps_algorithms\\ES_CTM_algorithms_SU2.jl")
# include("..\\..\\..\\..\\..\\src\\mps_algorithms\\parity_funs.jl")
# include("..\\..\\..\\..\\..\\src\\mps_algorithms\\position_permute.jl")

include("..\\..\\..\\..\\..\\src\\mps_algorithms\\ES_algorithms.jl")

Random.seed!(555)


D=4;
chi=40;


Nv=8;
EH_n=30;
group_index=true;
vison=false;


#permute_neighbour_ind=bosonic_permute_neighbour_ind;
#permute_neighbour_ind=fermionic_permute_neighbour_ind;



J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);


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
optim_setting.init_statenm="Optim_LS_Dx_4_Dy_13_chi_80.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Square_2site_Energy_settings();
energy_setting.model = "triangle_J1_J2_Jchi";
dump(energy_setting);




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

global starting_time
starting_time=now();


data=load(optim_setting.init_statenm);

A=data["A"];






#############################

init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM, AA, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A,chi,init,[],grad_ctm_setting);




#ES_CTMRG_ED_Kprojector(CTM,D,chi,Nv,EH_n,group_index,vison);
ES_CTMRG_ED(CTM,D,chi,Nv,EH_n,group_index,vison)


