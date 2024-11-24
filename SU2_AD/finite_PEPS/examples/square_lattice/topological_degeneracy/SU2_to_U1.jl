using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")

"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""



data=load("CSL_D3_SU2.jld2");
A=data["A"];


V=Rep[U₁](0=>1, 1/2=>1,-1/2=>1);
Vp=Rep[U₁](1/2=>1,-1/2=>1);

A_dense=convert(Array,A);
A_U1=TensorMap(A_dense,V*V,V*V*Vp);
A_U1=permute(A_U1,(1,2,3,4,5,));
jldsave("CSL_D3_U1.jld2";A=A_U1);

