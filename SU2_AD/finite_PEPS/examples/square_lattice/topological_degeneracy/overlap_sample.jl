using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\mps_methods.jl")
include("..\\..\\..\\environment\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\truncations.jl")

"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=6;
Ly=6;


data=load("CSL_D3.jld2");
A=data["A"];

gate_up=parity_gate(A,4);
gate_left=parity_gate(A,1);

# Vv=U₁Space(0=>1,1=>1,-1=>1);
# Vp=U₁Space(1=>1,-1=>1);
Vv=ℤ₂Space(0=>1,1=>2);
Vp=ℤ₂Space(1=>2);
A=TensorMap(convert(Array,A),Vv*Vv,Vv*Vv*Vp);

