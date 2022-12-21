using LinearAlgebra
using TensorKit
using TensorKitAD
using Zygote
using KrylovKit
using JSON
using HDF5, JLD
using Random
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG_AD_test")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")


include("functions.jl")



Random.seed!(1234)


D=3;


println("D="*string(D));flush(stdout);





B1a,B1b,B2=construct_tensors(D);
state_vec=rand(Float64, 3);




costfun(state_vec)=cost_fun(state_vec,B1a,B1b,B2);
E=costfun(state_vec);
Grad_AD=costfun'(state_vec);









