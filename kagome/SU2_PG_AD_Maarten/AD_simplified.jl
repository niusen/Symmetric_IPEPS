using Revise,LinearAlgebra, TensorKit, TensorKitAD, Zygote, KrylovKit, Random
using Zygote:@ignore_derivatives

cd(@__DIR__)
include("functions.jl")



Random.seed!(1234)



D=3;
println("D="*string(D));flush(stdout);

B1a,B1b,B2=construct_tensors(D);
state_vec=rand(Float64, 3);

costfun(state_vec)=cost_fun(state_vec,B1a,B1b,B2);
E=costfun(state_vec);
Grad_AD=costfun'(state_vec);
