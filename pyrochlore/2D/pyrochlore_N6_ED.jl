using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
using Combinatorics
cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("pyrochlore_load_tensor.jl")
include("pyrochlore_IPESS.jl")
include("square_CTMRG.jl")
include("spin_operator.jl")
include("pyrochlore_model.jl")
include("build_tensor.jl")

Random.seed!(1234)

J1=1;
J2=1;
lambda=1;


#plaquettes: (1 2 3 4);(3 4 5 6);(1 2 5 6);

D=2;




Bond_irrep="A";
Square_irrep="A1";
init_statenm="nothing";
init_noise=0;
A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
json_state_dict, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=initial_state(Bond_irrep,Square_irrep,D,init_statenm,init_noise);
bond_tensor,square_tensor_A1=construct_su2_PG_IPESS(json_state_dict,A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb);
square_tensor_A1=square_tensor_A1/norm(square_tensor_A1);

Bond_irrep="A";
Square_irrep="B1";
init_statenm="nothing";
init_noise=0;
A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
json_state_dict, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=initial_state(Bond_irrep,Square_irrep,D,init_statenm,init_noise);
bond_tensor,square_tensor_B1=construct_su2_PG_IPESS(json_state_dict,A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb);
square_tensor_B1=square_tensor_B1/norm(square_tensor_B1);

PEPS_tensor,A_fused1,U_phy=build_PEPS(bond_tensor,square_tensor_A1);
PEPS_tensor,A_fused2,U_phy=build_PEPS(bond_tensor,square_tensor_B1);


A_set=Vector(undef,2);
A_set[1]=A_fused1;
A_set[2]=A_fused2;


Sigma=plaquatte_Heisenberg(J1,J2);
AKLT=plaquatte_AKLT(Sigma);

@tensor H_eff[:]:=AKLT[5,6,7,8,1,2,3,4]*U_phy[-1,1,2]*U_phy[-2,3,4]*U_phy'[5,6,-3]*U_phy'[7,8,-4];
Id=unitary(space(H_eff,1),space(H_eff,1));

@tensor H1[:]:=H_eff[-1,-2,-4,-5]*Id[-3,-6];
@tensor H2[:]:=H_eff[-1,-3,-4,-6]*Id[-2,-5];
@tensor H3[:]:=H_eff[-2,-3,-5,-6]*Id[-1,-4];

H=H1+H2+H3;
H=permute(H,(1,2,3,),(4,5,6,));

eu,ev=eigen(H)