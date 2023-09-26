using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
cd(@__DIR__)
include("..\\resource_codes\\kagome_load_tensor.jl")
include("..\\resource_codes\\kagome_CTMRG.jl")
include("..\\resource_codes\\kagome_model.jl")
include("..\\resource_codes\\kagome_IPESS.jl")
include("..\\resource_codes\\kagome_FiniteDiff.jl")
include("..\\resource_codes\\Settings.jl")


Random.seed!(12345)


D=3;
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





ctm_setting=CTMRG_settings();
ctm_setting.CTM_conv_tol=1e-6;
ctm_setting.CTM_ite_nums=50;
ctm_setting.CTM_trun_tol=1e-8;
ctm_setting.svd_lanczos_tol=1e-8;
ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
ctm_setting.conv_check="singular_value";
ctm_setting.CTM_ite_info=true;
ctm_setting.CTM_conv_info=true;
ctm_setting.CTM_trun_svd=false;
ctm_setting.construct_double_layer=true;

dump(ctm_setting);


optim_setting=Optim_settings();
optim_setting.init_statenm="nothing";#"LS_A1even_U1_D_6_chi_60.json";#"nothing";
optim_setting.init_noise=0;
optim_setting.grad_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"

dump(optim_setting);

energy_setting=Energy_settings()
energy_setting.kagome_method ="E_single_triangle";
energy_setting.E_up_method = "1x1";
energy_setting.cal_chiral_order = false;

dump(energy_setting);

A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
global A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb  
run_FiniteDiff(parameters,D,chi,Bond_irrep,Triangle_irrep,nonchiral,ctm_setting,optim_setting,energy_setting);








