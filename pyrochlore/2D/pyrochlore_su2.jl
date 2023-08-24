using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("pyrochlore_load_tensor.jl")
include("pyrochlore_IPESS.jl")
include("square_CTMRG.jl")
include("spin_operator.jl")
include("pyrochlore_model.jl")
include("build_tensor.jl")

Random.seed!(1234)


D=2;

coe=[1,0];
virtual_type="tetrahedral";
PEPS_tensor,A_fused,U_phy=build_PEPS(D,coe1,virtual_type);



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






rho=build_density_op(U_phy, PEPS_tensor, AA_fused, U_L,U_D,U_R,U_U, CTM);#L',U',R',D',  L,U,R,D

Sigma=plaquatte_Heisenberg();

AKLT=plaquatte_AKLT(Sigma);


Ea=plaquatte_ob(rho,AKLT)
Eb=plaquatte_ob(rho,Sigma)

