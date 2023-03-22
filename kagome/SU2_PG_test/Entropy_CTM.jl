using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD

cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG_test")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model.jl")
include("kagome_IPESS.jl")
include("mps_algorithms\\ITEBD_algorithms.jl")
include("mps_algorithms\\TransfOp_decomposition.jl")
include("mps_algorithms\\PUMPS_algorithms.jl")
include("mps_algorithms\\ES_preliminary.jl")




D=3;

N=4;
EH_n=20;#number of entanglement spectrum
filenm="julia_LS_D_3_chi_20.json"

J1=1;
J2=0;
J3=0;
Jchi=0;
Jtrip=0;
parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);

CTM_conv_tol=1e-6;
CTM_ite_nums=200;
CTM_trun_tol=1e-12;
group_index=true;


chi=40;
ES_CTMRG_ED_Kprojector(filenm,parameters,D,chi,N,EH_n,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,group_index)
