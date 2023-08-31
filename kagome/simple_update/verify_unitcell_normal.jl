using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2
using Random

cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_load_tensor.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_CTMRG_unitcell.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_model_cell.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_IPESS.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_FiniteDiff.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_correl_cell.jl")


include("funs_1up_1down.jl")


Random.seed!(12345)


D_max=6;
symmetric_hosvd=false;
trun_tol=1e-6;
D=3;

println("D_max= "*string(D_max))

chis=[40];
#chis=[40,80,120,160];
CTM_conv_tol=1e-6;
CTM_ite_nums=200;
CTM_trun_tol=1e-12;

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

# tau=5;
# dt=0.02;
# T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, trun_tol, tau, dt, D_max,symmetric_hosvd);

# tau=2;
# dt=0.01;
# T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, trun_tol, tau, dt, D_max,symmetric_hosvd);

# tau=1;
# dt=0.005;
# T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, trun_tol, tau, dt, D_max,symmetric_hosvd);


println(space(T_u))
println(space(T_d))


@tensor PEPS_tensor[:] := B_a[-1,1,-5]*B_b[4,3,-6]*B_c[-4,2,-7]*T_u[1,3,2]*T_d[-3,4,-2];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];




#########################

id_L=unitary(space(A_fused,1),space(A_fused,1));
id_D=unitary(space(A_fused,2),space(A_fused,2));
id_phy=unitary(space(A_fused,5),space(A_fused,5));


@tensor A_LD[:]:=id_L[-1,-3]*id_D[-4,-2];
@tensor A_RU[:]:=id_L[-5,-1]*id_D[-3,-6]*id_phy[-2,-4];
U_L_phy=unitary(fuse(space(A_RU,1)⊗ space(A_RU,2)), space(A_RU,1)⊗ space(A_RU,2));
U_D_phy=unitary(fuse(space(A_fused,4)⊗ space(A_fused,5)), space(A_fused,4)⊗ space(A_fused,5));
@tensor A_LU[:]:=A_fused'[-1,-2,1,-4,2]*U_L_phy'[1,2,-3];
@tensor A_RD[:]:=A_fused[-1,-2,-3,1,2]*U_D_phy[-4,1,2];
@tensor A_RU[:]:=A_RU[1,2,3,4,-3,-4]*U_D_phy'[3,4,-2]*U_L_phy[-1,1,2];

# @tensor tt[:]:=A_RU[-1,1,-2,-3]*A_RD[-4,-5,-6,1];
# @tensor tt[:]:=A_LU[-1,-2,1,-3]*A_RU[1,-4,-5,-6];
# AA_fused, U_L,U_D,U_R,U_U=build_double_layer(A_fused,[]);
# @tensor AA_fused_[:]:=A_LU[2,1,7,8]*A_LD[3,11,10,1]*U_L[-1,2,3]*A_RU[7,4,5,9]*A_RD[10,12,6,4]*U_R[5,6,-3]*U_U[8,9,-4]*U_D[-2,11,12];
# @assert norm(AA_fused-permute(AA_fused_,(1,2,),(3,4,)))/norm(AA_fused)<1e-14

Lx=2;Ly=2;
A_cell=Matrix(undef,Lx,Ly);
A_cell[1,1]=A_LU;
A_cell[2,1]=A_RU;
A_cell[1,2]=A_LD;
A_cell[2,2]=A_RD;

#########################






AA_fused, U_L,U_D,U_R,U_U=build_double_layer(A_fused,[]);
@tensor AA_fused_[:]:=A_LU[2,1,7,8]*A_LD[3,11,10,1]*U_L[-1,2,3]*A_RU[7,4,5,9]*A_RD[10,12,6,4]*U_R[5,6,-3]*U_U[8,9,-4]*U_D[-2,11,12];
@assert norm(AA_fused-permute(AA_fused_,(1,2,),(3,4,)))/norm(AA_fused)<1e-14
########################








CTM=[];
U_L=[];
U_D=[];
U_R=[];
U_U=[];

chi=chis[1];
println("chi= "*string(chi));flush(stdout);

conv_check="singular_value";
CTM_ite_info=true;
CTM_conv_info=true;

Random.seed!(123456)
CTM_init, _,_,_,_,_=init_CTM_cell(chi,A_cell,"single_layer_random",CTM_ite_info)
init=Dict([("CTM", deepcopy(CTM_init)), ("init_type", "single_layer_random")]);




    CTM, _,_,_,_,_,ite_num,ite_err=CTMRG_cell(A_cell,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);

    # method1="E_triangle";
    # method2="full_cell";
    # E_up=evaluate_ob_UpTriangle_single_layer(parameters, U_phy, U_D_phy, A_cell, CTM, method1, method2);
    # energy=E_up*2/3;
    # println("Up triangle energy: "*string(energy))

    method1="E_triangle";
    method2="reduced_cell";
    E_up=evaluate_ob_UpTriangle_single_layer(parameters, U_phy, U_D_phy, A_cell, CTM, method1, method2);
    energy=E_up*2/3;
    println("Up triangle energy: "*string(energy))

    eu_allspin_x,allspin_x=solve_correl_length_single_layer(5,[],CTM,"x");
    eu_allspin_y,allspin_y=solve_correl_length_single_layer(5,[],CTM,"y");


  



