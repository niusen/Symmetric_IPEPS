using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
using Revise,TensorKitAD, Zygote, KrylovKit, Random
using Zygote:@ignore_derivatives
cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("pyrochlore_load_tensor.jl")
include("pyrochlore_IPESS.jl")
include("square_CTMRG.jl")
include("spin_operator.jl")
include("pyrochlore_model.jl")
include("build_tensor.jl")
include("functions.jl")

Random.seed!(1234)


D=2;

coe=[1,0];

A_set,E_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

Sigma=plaquatte_Heisenberg();

AKLT=plaquatte_AKLT(Sigma);



###################
PEPS_tensor,A_fused,U_phy=build_PEPS(A_set,E_set,coe);

CTM=[];
U_L=[];
U_D=[];
U_R=[];
U_U=[];

init=Dict([("CTM", []), ("init_type", "PBC")]);
conv_check="singular_value";
CTM_ite_info=true;
CTM_conv_info=true;
CTM_conv_tol=1e-6;
CTM_ite_nums=100;
CTM_trun_tol=1e-12;
chi=40;
@time CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);

##############################


costfun(coe)=cost_fun(coe,Sigma,A_set,E_set,CTM);
E=costfun(coe);

Grad_AD=costfun'(coe);