using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model.jl")
include("kagome_IPESS.jl")
include("parameter_interpolation.jl")






D=8;
chi=60;


theta=0.055*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);



parameter=tan(theta);
parameterL=tan(0.045*pi);
parameterR=tan(0.06*pi);

init_statenmL="julia_LS_D_8_theta_0.045.json";
init_statenmR="julia_LS_D_8_theta_0.06.json"


CTM_conv_tol=1e-6;
CTM_ite_nums=100;
CTM_trun_tol=1e-12;
Bond_irrep="A";
Triangle_irrep="A1+iA2";
nonchiral="No"
run_parameter_interpolation(parameters,parameter,parameterL,parameterR,D,chi,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,Bond_irrep,Triangle_irrep,nonchiral,init_statenmL,init_statenmR)







