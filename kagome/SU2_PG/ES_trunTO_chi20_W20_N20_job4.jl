using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD

cd("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model.jl")
include("kagome_IPESS.jl")
include("mps_algorithms/ITEBD_algorithms.jl")
include("mps_algorithms/TransfOp_decomposition.jl")
include("mps_algorithms/PUMPS_algorithms.jl")
include("mps_algorithms/ES_preliminary.jl")




D=8;
chi=20;
W=20;
N=20;
kset=15:19;
EH_n=3;#number of entanglement spectrum per k point
Dtrun_method="svds";
Dtrun_init=400;
Dtrun_max=400;

cal_ES(D,chi,W,N,kset,EH_n,Dtrun_init,Dtrun_max,Dtrun_method)