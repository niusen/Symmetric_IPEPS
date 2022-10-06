using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model.jl")
include("kagome_IPESS.jl")
include("kagome_FiniteDiff.jl")



Random.seed!(1234)


D=3;
chi=40;


theta=0*pi;
J1=cos(theta);
J2=0.5;
J3=0.3;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);



#state_dict=read_json_state("LS_D_8_chi_40.json")
init_statenm=nothing;
#init_statenm="julia_LS_D_8_chi_40.json"
init_noise=0;
CTM_conv_tol=1e-6;
CTM_ite_nums=100;
CTM_trun_tol=1e-12;
grad_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
Bond_irrep="A";
Triangle_irrep="A1+iA2";
#nonchiral="A1_even";
nonchiral="No"
run_FiniteDiff(parameters,D,chi,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,grad_CTM_method,linesearch_CTM_method,Bond_irrep,Triangle_irrep,nonchiral,init_statenm,init_noise)









