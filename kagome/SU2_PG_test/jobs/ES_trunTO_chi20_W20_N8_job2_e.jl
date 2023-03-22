using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD


include("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG/kagome_load_tensor.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG/kagome_CTMRG.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG/kagome_model.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG/kagome_IPESS.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG/mps_algorithms/ITEBD_algorithms.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG/mps_algorithms/TransfOp_decomposition.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG/mps_algorithms/PUMPS_algorithms.jl")
include("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG/mps_algorithms/ES_preliminary.jl")





D=8;
chi=20;
W=20;
N=8;
kset=4:4;
EH_n=6;#number of entanglement spectrum per k point
Dtrun_method="eigs";
Dtrun_init=5000;
Dtrun_max=5000;
Dtrun_tol=1e-6;
unitcell_size=1;

J1=0.80902;
J2=0;
J3=0;
Jchi=0;
Jtrip=0.5878;
parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);

filenm="julia_LS_D_8_chi_40.json";
cal_ES(filenm,parameters,D,chi,W,N,kset,EH_n,Dtrun_init,Dtrun_max,Dtrun_tol,Dtrun_method,unitcell_size,true)