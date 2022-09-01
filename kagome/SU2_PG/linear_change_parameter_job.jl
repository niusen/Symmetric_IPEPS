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
include("linear_change_parameter.jl")






D=6;
chi=20;


theta=0.2*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);



#state_dict=read_json_state("LS_D_8_chi_40.json")

init_statenmL="julia_LS_D_6_chi_40.json"
init_statenmR="julia_LS_A1even_D_6_chi_20.json"

Np=15;
CTM_conv_tol=1e-6;
CTM_ite_nums=100;
CTM_trun_tol=1e-12;
Bond_irrep="A";
Triangle_irrep="A1+iA2";
nonchiral="No";
run_parameter_change(parameters,D,chi,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,Bond_irrep,Triangle_irrep,nonchiral,init_statenmL,init_statenmR,Np)








