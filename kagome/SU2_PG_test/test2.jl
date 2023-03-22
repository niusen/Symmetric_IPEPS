using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD

#cd("/users/p1231/niu/Code/Julia_codes/Tensor_network/IPEPS_TensorKit/kagome/SU2_PG")
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model.jl")
include("kagome_IPESS.jl")
include("kagome_correl.jl")




D=8;

J1=0.80902;
J2=0;
J3=0;
Jchi=0;
Jtrip=0.5878;
parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);

CTM_conv_tol=1e-6;
distance=100;


chi=20;
CTM_ite_nums=50;
CTM_trun_tol=1e-12;
filenm="LS_D_8_test.json";
cal_correl(filenm,D,chi,parameters,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,distance)
