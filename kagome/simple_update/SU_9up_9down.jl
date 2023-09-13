using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("..\\resource_codes\\kagome_load_tensor.jl")
include("..\\resource_codes\\kagome_CTMRG.jl")
include("..\\resource_codes\\kagome_model.jl")
include("..\\resource_codes\\kagome_IPESS.jl")
include("..\\resource_codes\\kagome_FiniteDiff.jl")

include("funs_9up_9down.jl")


Random.seed!(1234)

D_max=15;

trun_tol=1e-6;
D=3;#initial state bond dimension


theta=0*pi;
J1=cos(theta);
J2=0.4;
J3=J2;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);



#state_dict=read_json_state("LS_D_8_chi_40.json")
init_statenm=nothing;
#init_statenm="julia_LS_D_8_chi_40.json"
init_noise=0;
Bond_irrep="A";
Triangle_irrep="A1+iA2";
nonchiral="A1_even";
#nonchiral="No"

A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
state_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd=initial_state(Bond_irrep,Triangle_irrep,nonchiral,D,init_statenm,init_noise);
bond_tensor,triangle_tensor=construct_su2_PG_IPESS(state_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

T_u=deepcopy(triangle_tensor)*im;
T_d=deepcopy(triangle_tensor)*im;
B_a=deepcopy(bond_tensor)*im;
B_b=deepcopy(bond_tensor)*im;
B_c=deepcopy(bond_tensor)*im;
lambda_u_a=unitary(space(bond_tensor,1),space(bond_tensor,1));
lambda_u_a=lambda_u_a'*lambda_u_a;
lambda_u_b=deepcopy(lambda_u_a);
lambda_u_c=deepcopy(lambda_u_a);
lambda_d_a=deepcopy(lambda_u_a);
lambda_d_b=deepcopy(lambda_u_a);
lambda_d_c=deepcopy(lambda_u_a);


U_d=space(bond_tensor,3);
U_phy_3=unitary(fuse(U_d ⊗ U_d ⊗ U_d), U_d ⊗ U_d ⊗ U_d);
U_phy_2=unitary(fuse(U_d ⊗ U_d), U_d ⊗ U_d);
H_triangle, H_Heisenberg, H12_tensorkit, H31_tensorkit, H23_tensorkit=Hamiltonians(U_phy_3,parameters["J1"],parameters["J2"],parameters["J3"],parameters["Jchi"],parameters["Jtrip"])
@tensor H_triangle[:]:=U_phy_3[2,-1,-2,-3]*H_triangle[2,1]*U_phy_3'[-4,-5,-6,1];
H_triangle=permute(H_triangle,(1,2,3,),(4,5,6,));
H_Heisenberg=TensorMap(H_Heisenberg, U_d' ⊗ U_d' ← U_d' ⊗ U_d');




#T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, trun_tol, tau, dt, D_max);





T_u_set=Dict([("A", T_u), ("B", T_u), ("C", T_u), ("D", T_u), ("E", T_u), ("F", T_u), ("G", T_u), ("H", T_u), ("I", T_u)]);
T_d_set=Dict([("A", T_d), ("B", T_d), ("C", T_d), ("D", T_d), ("E", T_d), ("F", T_d), ("G", T_d), ("H", T_d), ("I", T_d)]);
B_a_set=Dict([("A", B_a), ("B", B_a), ("C", B_a), ("D", B_a), ("E", B_a), ("F", B_a), ("G", B_a), ("H", B_a), ("I", B_a)]);
B_b_set=Dict([("A", B_b), ("B", B_b), ("C", B_b), ("D", B_b), ("E", B_b), ("F", B_b), ("G", B_b), ("H", B_b), ("I", B_b)]);
B_c_set=Dict([("A", B_c), ("B", B_c), ("C", B_c), ("D", B_c), ("E", B_c), ("F", B_c), ("G", B_c), ("H", B_c), ("I", B_c)]);
lambda_a_u_set=Dict([("A", lambda_u_a), ("B", lambda_u_a), ("C", lambda_u_a), ("D", lambda_u_a), ("E", lambda_u_a), ("F", lambda_u_a), ("G", lambda_u_a), ("H", lambda_u_a), ("I", lambda_u_a)]);
lambda_b_u_set=Dict([("A", lambda_u_a), ("B", lambda_u_a), ("C", lambda_u_a), ("D", lambda_u_a), ("E", lambda_u_a), ("F", lambda_u_a), ("G", lambda_u_a), ("H", lambda_u_a), ("I", lambda_u_a)]);
lambda_c_u_set=Dict([("A", lambda_u_a), ("B", lambda_u_a), ("C", lambda_u_a), ("D", lambda_u_a), ("E", lambda_u_a), ("F", lambda_u_a), ("G", lambda_u_a), ("H", lambda_u_a), ("I", lambda_u_a)]);
lambda_a_d_set=Dict([("A", lambda_u_a), ("B", lambda_u_a), ("C", lambda_u_a), ("D", lambda_u_a), ("E", lambda_u_a), ("F", lambda_u_a), ("G", lambda_u_a), ("H", lambda_u_a), ("I", lambda_u_a)]);
lambda_b_d_set=Dict([("A", lambda_u_a), ("B", lambda_u_a), ("C", lambda_u_a), ("D", lambda_u_a), ("E", lambda_u_a), ("F", lambda_u_a), ("G", lambda_u_a), ("H", lambda_u_a), ("I", lambda_u_a)]);
lambda_c_d_set=Dict([("A", lambda_u_a), ("B", lambda_u_a), ("C", lambda_u_a), ("D", lambda_u_a), ("E", lambda_u_a), ("F", lambda_u_a), ("G", lambda_u_a), ("H", lambda_u_a), ("I", lambda_u_a)]);


#triangles around a hexagon, from left-top, clockwise
Cell_ind1=["B","E","A","A","F","B"];
Cell_ind2=["C","H","B","B","I","C"];
Cell_ind3=["A","G","C","C","D","A"];
Cell_ind4=["I","B","F","F","G","I"];
Cell_ind5=["D","C","I","I","E","D"];
Cell_ind6=["F","A","D","D","H","F"];
Cell_ind7=["E","I","G","G","A","E"];
Cell_ind8=["H","D","E","E","B","H"];
Cell_ind9=["G","F","H","H","C","G"];
Cell_ind=[Cell_ind1, Cell_ind2, Cell_ind3, Cell_ind4, Cell_ind5, Cell_ind6, Cell_ind7, Cell_ind8, Cell_ind9];

T_d_A_envs=["E","G","A"];
T_d_B_envs=["H","E","B"];
T_d_C_envs=["G","H","C"];
T_d_D_envs=["A","C","D"];
T_d_E_envs=["D","I","E"];
T_d_F_envs=["B","A","F"];
T_d_G_envs=["I","F","G"];
T_d_H_envs=["F","D","H"];
T_d_I_envs=["C","B","I"];
T_d_envs=Dict([("A", T_d_A_envs), ("B", T_d_B_envs), ("C", T_d_C_envs), ("D", T_d_D_envs), ("E", T_d_E_envs), ("F", T_d_F_envs), ("G", T_d_G_envs), ("H", T_d_H_envs), ("I", T_d_I_envs)]);

T_u_A_envs=["D","F","A"];
T_u_B_envs=["F","I","B"];
T_u_C_envs=["I","D","C"];
T_u_D_envs=["E","H","D"];
T_u_E_envs=["A","B","E"];
T_u_F_envs=["H","G","F"];
T_u_G_envs=["C","A","G"];
T_u_H_envs=["B","C","H"];
T_u_I_envs=["G","E","I"];
T_u_envs=Dict([("A", T_u_A_envs), ("B", T_u_B_envs), ("C", T_u_C_envs), ("D", T_u_D_envs), ("E", T_u_E_envs), ("F", T_u_F_envs), ("G", T_u_G_envs), ("H", T_u_H_envs), ("I", T_u_I_envs)]);






tau=1;
dt=0.2;

T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set=itebd(T_u_set,T_d_set,B_a_set,B_b_set,B_c_set,lambda_a_u_set,lambda_b_u_set,lambda_c_u_set,lambda_a_d_set,lambda_b_d_set,lambda_c_d_set, H_triangle, H_Heisenberg, J2, J3, trun_tol, tau, dt, D_max, Cell_ind, T_d_envs,T_u_envs);



println(space(T_u_set["A"]))
println(space(T_u_set["B"]))
println(space(T_u_set["C"]))
println(space(T_u_set["D"]))
println(space(T_u_set["E"]))
println(space(T_u_set["F"]))
println(space(T_u_set["G"]))
println(space(T_u_set["H"]))
println(space(T_u_set["I"]))

println(space(T_d_set["A"]))
println(space(T_d_set["B"]))
println(space(T_d_set["C"]))
println(space(T_d_set["D"]))
println(space(T_d_set["E"]))
println(space(T_d_set["F"]))
println(space(T_d_set["G"]))
println(space(T_d_set["H"]))
println(space(T_d_set["I"]))