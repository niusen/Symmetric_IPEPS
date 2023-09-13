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
include("pyrochlore_correl.jl")
include("Settings.jl")

Random.seed!(1234)

J1=1;
J2=1;
D=12;


Bond_irrep="A";
Square_irrep="A1";#"A1", "A1+iB1"
init_statenm="nothing";
init_noise=0;

@time A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
json_state_dict, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=initial_state(Bond_irrep,Square_irrep,D,init_statenm,init_noise);
bond_tensor,square_tensor=construct_su2_PG_IPESS(json_state_dict,A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb);
PEPS_tensor,A_fused,U_phy=build_PEPS(bond_tensor,square_tensor);


chi=40;

ctm_setting=CTMRG_settings();
ctm_setting.CTM_conv_tol=1e-6;
ctm_setting.CTM_ite_nums=30;
ctm_setting.CTM_trun_tol=1e-12;
ctm_setting.svd_lanczos_tol=1e-8;
ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
ctm_setting.conv_check="singular_value";
ctm_setting.CTM_ite_info=true;
ctm_setting.CTM_conv_info=true;
ctm_setting.CTM_trun_svd=false;
ctm_setting.construct_double_layer=true;


CTM=[];
U_L=[];
U_D=[];
U_R=[];
U_U=[];

init=Dict([("CTM", []), ("init_type", "PBC")]);

CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,ctm_setting);



#@time rho=build_density_op(U_phy, PEPS_tensor, AA_fused, U_L,U_D,U_R,U_U, CTM);#L',U',R',D',  L,U,R,D


Sigma=plaquatte_Heisenberg(J1,J2);
Sigma_fused=fuse_H(Sigma,U_phy);

AKLT=plaquatte_AKLT(Sigma);




U_U_phy=unitary(fuse(space(U_phy,1)*space(U_phy,1)), space(U_phy,1)*space(U_phy,1));

#Ea=plaquatte_ob(rho,AKLT)
#@time Eb=plaquatte_ob(rho,Sigma)

@time Eb1=ob_efficient(Sigma_fused, U_phy, AA_fused, CTM,bond_tensor,square_tensor);#L',U',R',D',  L,U,R,D



eu_allspin_x,allspin_x=solve_correl_length(5,[],CTM,"x",ctm_setting);
eu_allspin_y,allspin_y=solve_correl_length(5,[],CTM,"y",ctm_setting);


matwrite("D"*string(D)*"_correl"*"_chi"*string(chi)*".mat", Dict(
    "energy" => Eb1,
    "eu_allspin_x"=>eu_allspin_x,
    "allspin_x"=>allspin_x,
    "eu_allspin_y"=>eu_allspin_y,
    "allspin_y"=>allspin_y,

); compress = false)

