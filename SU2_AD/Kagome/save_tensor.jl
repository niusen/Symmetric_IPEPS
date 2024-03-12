using Revise, TensorKit, Zygote
using LinearAlgebra, OptimKit
#using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using LineSearches
using Zygote:@ignore_derivatives
using Dates
using MAT

cd(@__DIR__)

include("..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\src\\bosonic\\kagome_AD_SU2.jl")
include("..\\src\\bosonic\\Settings.jl")
include("..\\src\\bosonic\\line_search_lib.jl")



Random.seed!(555)




theta=0*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);



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
optim_setting.init_statenm="SimpleUpdate_D_6.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

state_vec=load(optim_setting.init_statenm)

x0=state_vec;

B1=x0["B_a"];
B2=x0["B_b"];
B3=x0["B_c"];
Tup=x0["T_u"];
Tdn=x0["T_d"];


@tensor PEPS_tensor[:] := B1[-1,1,-5]*B2[4,3,-6]*B3[-4,2,-7]*Tup[1,3,2]*Tdn[-3,4,-2];
B1=convert(Array, B1);
B2=convert(Array, B2);
B3=convert(Array, B3);
Tup=convert(Array, Tup);
Tdn=convert(Array, Tdn);

filenm="state_D6.mat"
matwrite(filenm, Dict(
	"B1" => B1,
    "B2" => B2,
    "B3" => B3,
    "Tup" => Tup,
    "Tdn" => Tdn
); compress = false)
