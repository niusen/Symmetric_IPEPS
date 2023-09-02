using LinearAlgebra
using TensorKit
# using Suppressor
using KrylovKit
using JSON
using HDF5, JLD2
using Random

cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_load_tensor.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_CTMRG.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_model.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_IPESS.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_FiniteDiff.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\kagome_correl.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\simple_update\\resource_codes\\Settings.jl")


include("funs_1up_1down.jl")


Random.seed!(1234)


D_max=6;
symmetric_hosvd=false;
itebd_trun_tol=1e-6;
D=3;

println("D_max= "*string(D_max))

chis=[40];
#chis=[40,80,120,160];

ctm_setting=CTMRG_settings();
ctm_setting.CTM_conv_tol=1e-6;
ctm_setting.CTM_ite_nums=30;
ctm_setting.CTM_trun_tol=1e-8;
ctm_setting.svd_lanczos_tol=1e-8;
ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
ctm_setting.conv_check="singular_value";
ctm_setting.CTM_ite_info=true;
ctm_setting.CTM_conv_info=true;
ctm_setting.CTM_trun_svd=false;



dump(ctm_setting);


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
T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, itebd_trun_tol, tau, dt, D_max,symmetric_hosvd);

# tau=2;
# dt=0.01;
# T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, itebd_trun_tol, tau, dt, D_max,symmetric_hosvd);

# tau=1;
# dt=0.005;
# T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, itebd_trun_tol, tau, dt, D_max,symmetric_hosvd);


println(space(T_u))
println(space(T_d))


@tensor PEPS_tensor[:] := B_a[-1,1,-5]*B_b[4,3,-6]*B_c[-4,2,-7]*T_u[1,3,2]*T_d[-3,4,-2];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

CTM=[];
U_L=[];
U_D=[];
U_R=[];
U_U=[];

init=Dict([("CTM", []), ("init_type", "PBC")]);



global chis,D_max,init,A_fused

for cchi=1:length(chis)
    global chis,D_max,init,A_fused
    
    chi=chis[cchi];
    println("chi= "*string(chi));flush(stdout);
    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,ctm_setting);

    E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
    energy=(E_up+E_down)/3;
    println("Up triangle energy: "*string(energy))

    eu_allspin_x,allspin_x=solve_correl_length(5,AA_fused/norm(AA_fused),CTM,"x");
    eu_allspin_y,allspin_y=solve_correl_length(5,AA_fused/norm(AA_fused),CTM,"y");

    init=Dict([("CTM", CTM), ("init_type", "PBC")]);

    matwrite("SimpleUpdate_ob"*"_D"*string(D_max)*"_chi"*string(chi)*".mat", Dict(
        "energy" => energy,
        "eu_allspin_x"=>eu_allspin_x,
        "allspin_x"=>allspin_x,
        "eu_allspin_y"=>eu_allspin_y,
        "allspin_y"=>allspin_y,
        "space_T_u"=>string(space(T_u)),
        "space_T_d"=>string(space(T_d))
    ); compress = false)
end



