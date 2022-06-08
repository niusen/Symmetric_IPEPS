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
include("kagome_FiniteDiff.jl")



D=6;
chi=36;



J1=0.80902;
J2=0;
J3=0;
Jchi=0;
Jtrip=0.5878;

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);



#state_dict=read_json_state("LS_D_8_chi_40.json")
#init_statenm=nothing;
init_statenm="LS_D_6_chi_40.json"
CTM_conv_tol=1e-6;
CTM_ite_nums=100;
CTM_trun_tol=1e-12;
Bond_irrep="A";
run_FiniteDiff(parameters,D,chi,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,Bond_irrep,init_statenm)



