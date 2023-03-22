using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD

cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model.jl")
include("kagome_IPESS.jl")
include("mps_algorithms\\ITEBD_algorithms.jl")
include("mps_algorithms\\TransfOp_decomposition.jl")
include("mps_algorithms\\Entropy.jl")
include("mps_algorithms\\ES_preliminary.jl")




D=3;

N=4;
N_eu=3;#number of eigenvalues for each sector, in order to detect degeneracy

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


chi=10;

#Entropy_finite_size(filenm,parameters,D,chi,N,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol);

Topo_entropy_Renyi2(filenm,parameters,D,chi,N_eu,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol)
