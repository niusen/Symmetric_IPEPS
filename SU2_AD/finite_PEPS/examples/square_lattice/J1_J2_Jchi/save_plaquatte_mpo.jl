using LinearAlgebra
using TensorKit,TensorKitAD
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
cd(@__DIR__)


include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\compress_plaquatte_mpo.jl")






J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;

mpo_set=get_plaquatte_mpo(J1,J2,Jchi);



filenm="mpo_J1_"*string(J1)*"_J2_"*string(J2)*"_Jchi_"*string(Jchi)*".jld2";



jldsave(save_opt_filenm; mpo_set);






