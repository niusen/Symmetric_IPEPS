using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
using Plots
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_load_tensor.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_CTMRG.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_model.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_IPESS.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_FiniteDiff.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_correl.jl")


include("funs_1up_1down.jl")


Random.seed!(1234)


D_max=8;
symmetric_hosvd=false;
trun_tol=1e-6;
D=3;

chi=60;
CTM_conv_tol=1e-6;
CTM_ite_nums=200;
CTM_trun_tol=1e-12;

theta=0.2*pi;
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


@tensor PEPS_tensor[:] := B_a[-1,1,-5]*B_b[4,2,-6]*B_c[-4,3,-7]*T_u[1,2,3]*T_d[-3,4,-2];
A_unfused=PEPS_tensor;

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
CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);
if (parameters["J2"]==0) & (parameters["J3"]==0)
    E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
    energy=(E_up+E_down)/3;
elseif parameters["Jtrip"]==0
    E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23,   E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b,  E_NNNN_11,E_NNNN_22,E_NNNN_33=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_bond");
    E_NN=parameters["J1"]*(E_up_12+E_up_31+E_up_23+E_down_12+E_down_31+E_down_23);
    E_NNN=parameters["J2"]*(E_NNN_23a+E_NNN_12a+E_NNN_31a+E_NNN_23b+E_NNN_12b+E_NNN_31b);
    E_NNNN=parameters["J3"]*(E_NNNN_11+E_NNNN_22+E_NNNN_33);
    energy=(E_NN+E_NNN+E_NNNN)/3;
    println(real([E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23]))
    println(real([E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b]))
    println(real([E_NNNN_11,E_NNNN_22,E_NNNN_33]))
else
    E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
    E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23,   E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b,  E_NNNN_11,E_NNNN_22,E_NNNN_33=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_bond");
    E_NN=parameters["J1"]*(E_up_12+E_up_31+E_up_23+E_down_12+E_down_31+E_down_23);
    E_NNN=parameters["J2"]*(E_NNN_23a+E_NNN_12a+E_NNN_31a+E_NNN_23b+E_NNN_12b+E_NNN_31b);
    E_NNNN=parameters["J3"]*(E_NNNN_11+E_NNNN_22+E_NNNN_33);
    energy=(E_up+E_down)/3+(E_NNN+E_NNNN)/3;
end



chiral_order_parameters=Dict([("J1", 0), ("J2", 0), ("J3", 0), ("Jchi", 0), ("Jtrip", 1)]);
chiral_order_up, chiral_order_down=evaluate_ob(chiral_order_parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");

println(energy)



distance=30;

_, _, SS12, SS31, SS23=Hamiltonians(U_phy,1,0,0,0,0);
S1L,S1R=single_spin_operator(U_phy,1,1);
S2L,S2R=single_spin_operator(U_phy,2,2);
S3L,S3R=single_spin_operator(U_phy,3,3);
AA_S1L,_,_,_,_=build_double_layer_extra_leg(A_fused,S1L);
AA_S1R,_,_,_,_=build_double_layer_extra_leg(A_fused,S1R);
AA_S2L,_,_,_,_=build_double_layer_extra_leg(A_fused,S2L);
AA_S2R,_,_,_,_=build_double_layer_extra_leg(A_fused,S2R);
AA_S3L,_,_,_,_=build_double_layer_extra_leg(A_fused,S3L);
AA_S3R,_,_,_,_=build_double_layer_extra_leg(A_fused,S3R);

AA_SS12, _,_,_,_=build_double_layer(A_fused,SS12);
AA_SS31, _,_,_,_=build_double_layer(A_fused,SS31);
AA_SS23, _,_,_,_=build_double_layer(A_fused,SS23);



norms=evaluate_correl_spinspin("x", AA_fused, AA_fused, AA_fused, CTM, "dimerdimer", 10);
norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
norms=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_fused, AA_fused, CTM, "dimerdimer", distance);
SS12_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SS12, AA_SS12, CTM, "dimerdimer", distance);
SS23_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SS23, AA_SS23, CTM, "dimerdimer", distance);
SS31_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SS12, AA_SS31, CTM, "dimerdimer", distance);
S1_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_S1L, AA_S1R, CTM, "spinspin", distance);
S2_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_S2L, AA_S2R, CTM, "spinspin", distance);
S3_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_S3L, AA_S3R, CTM, "spinspin", distance);


SS12_ob=SS12_ob./norms;
SS23_ob=SS23_ob./norms;
SS31_ob=SS31_ob./norms;
S1_ob=S1_ob./norms;
S2_ob=S2_ob./norms;
S3_ob=S3_ob./norms;


eu_allspin_x,allspin_x=solve_correl_length(5,AA_fused/norm_coe,CTM,"x");
eu_allspin_y,allspin_y=solve_correl_length(5,AA_fused/norm_coe,CTM,"y");


S1_ob=real(S1_ob);
plot(range(1,distance),log.(abs.(S1_ob))/log(10))
