using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using LinearAlgebra, OptimKit
using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives


cd(@__DIR__)
include("resource_codes\\kagome_load_tensor.jl")
include("resource_codes\\kagome_CTMRG.jl")
include("resource_codes\\kagome_model.jl")
include("resource_codes\\kagome_IPESS.jl")
include("resource_codes\\kagome_AD.jl")
include("resource_codes\\Settings.jl")
include("resource_codes\\AD_lib.jl")



Random.seed!(555)


D=6;
chi=40;


theta=0*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);




Bond_irrep="A";
Triangle_irrep="A1+iA2";
nonchiral="A1_even";#"No", "A1_even"
ipess_irrep=IPESS_IRREP(Bond_irrep, Triangle_irrep, nonchiral);
dump(ipess_irrep);



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
LS_ctm_setting.grad_checkpoint=false;
dump(LS_ctm_setting);

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);

optim_setting=Optim_settings();
optim_setting.init_statenm="nothing";#"LS_A1even_D_6_chi_40.json";#"nothing";
optim_setting.init_noise=0;
optim_setting.grad_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Energy_settings()
energy_setting.kagome_method ="E_single_triangle";#"E_single_triangle", "E_triangle"
energy_setting.E_up_method = "1x1";#"1x1", "2x2"
energy_setting.cal_chiral_order = false;
dump(energy_setting);





global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=grad_ctm_setting.CTM_trun_tol

global backward_settings

run_FiniteDiff(parameters, D, chi, ipess_irrep, LS_ctm_setting, optim_setting, energy_setting)

#fun(state_vec)
# global E_tem, CTM_tem
# E,∂E,CTM_tem=get_grad(state_vec);
# println(E,∂E)


# E0, grad=FD(state_vec)
# println(grad)
# println(∂E./grad)




