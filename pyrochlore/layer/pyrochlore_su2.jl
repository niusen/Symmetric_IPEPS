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
include("kagome_CTMRG.jl")


Random.seed!(1234)


D=2;



A_set,E_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
bond_tensor=A_set[1];
tetrahedral_tensora=E_set[1];
tetrahedral_tensorb=E_set[2];

U=unitary(space(tetrahedral_tensora,4)',space(tetrahedral_tensora,4));
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*tetrahedral_tensora[1,3,2,5]*tetrahedral_tensorb[4,-2,-3,6]*U[5,6];



U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

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
