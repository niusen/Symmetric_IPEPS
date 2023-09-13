using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random

cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("..\\resource_codes\\kagome_load_tensor.jl")
include("..\\resource_codes\\kagome_model.jl")
include("..\\resource_codes\\kagome_IPESS.jl")
include("..\\resource_codes\\kagome_correl.jl")
include("..\\resource_codes\\kagome_CTMRG.jl")

include("funs_1up_1down.jl")
include("cluster_state.jl")


Random.seed!(1234)


D_max=28;
symmetric_hosvd=false;
trun_tol=1e-6;
D=3;



theta=0*pi;
J1=cos(theta);
J2=0;
J3=0;
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

T_u=deepcopy(triangle_tensor);
T_d=deepcopy(triangle_tensor);
B_a=deepcopy(bond_tensor);
B_b=deepcopy(bond_tensor);
B_c=deepcopy(bond_tensor);
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

tau=5;
dt=0.02;
T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, trun_tol, tau, dt, D_max,symmetric_hosvd);

tau=2;
dt=0.01;
T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, trun_tol, tau, dt, D_max,symmetric_hosvd);

tau=1;
dt=0.005;
T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, trun_tol, tau, dt, D_max,symmetric_hosvd);


println(space(T_u))
println(space(T_d))


@tensor PEPS_tensor[:] := B_a[-1,1,-5]*B_b[4,3,-6]*B_c[-4,2,-7]*T_u[1,3,2]*T_d[-3,4,-2];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];




Size="2x4"
psi=built_cluster(A_fused,Size);








###############################



D=D_max
#init_statenm="LS_A1even_U1_D_6_chi_60.json"
init_statenm=nothing
init_noise=0;
Bond_irrep="A";
Triangle_irrep="A1+iA2";
nonchiral="A1_even";
#nonchiral="No"

A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, _, _, virtual_particle, _, _=construct_tensor(D);
global A_set,B_set,A1_set,A2_set,A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, virtual_particle
run_FiniteDiff(psi, D,Bond_irrep,Triangle_irrep,nonchiral,init_statenm,init_noise)


