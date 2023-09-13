using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random

cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("pyrochlore_load_tensor.jl")
include("pyrochlore_IPESS.jl")
include("square_CTMRG.jl")
include("spin_operator.jl")
include("pyrochlore_model.jl")
include("build_tensor.jl")
include("pyrochlore_correl.jl")
include("build_tensor.jl")
include("Settings.jl")

include("pyrochlore_FiniteDiff.jl")



Random.seed!(1234)


D=6;
chi=40;



J1=1;
J2=1;
parameters=Dict([("J1", J1), ("J2", J2)]);

H=plaquatte_Heisenberg(J1,J2);




Square_irrep="A1";#"A1", "B1"
Bond_irrep="A";



ctm_setting=CTMRG_settings();
ctm_setting.CTM_conv_tol=1e-6;
ctm_setting.CTM_ite_nums=30;
ctm_setting.CTM_trun_tol=1e-12;
ctm_setting.svd_lanczos_tol=1e-8;
ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
ctm_setting.conv_check="singular_value";
ctm_setting.CTM_ite_info=true;
ctm_setting.CTM_conv_info=true;
ctm_setting.CTM_trun_svd=false;
ctm_setting.construct_double_layer=true;

dump(ctm_setting);


optim_setting=Optim_settings();
optim_setting.init_statenm="nothing";
optim_setting.init_noise=0;
optim_setting.grad_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"

dump(optim_setting);

A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
global A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb
run_FiniteDiff(parameters,D,chi,Bond_irrep,Square_irrep,ctm_setting,optim_setting);









