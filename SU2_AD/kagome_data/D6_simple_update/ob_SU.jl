using Revise, TensorKit, TensorKitAD, Zygote
using LinearAlgebra
using ChainRulesCore
using KrylovKit
using JSON, MAT
using Zygote:@ignore_derivatives

using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2
using Random

cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")

include("../../../src/bosonic/CTMRG.jl")
include("../../../src/bosonic/kagome_model.jl")
include("../../../src/bosonic/kagome_IPESS.jl")
include("../../../src/bosonic/kagome_correl.jl")
include("../../../src/bosonic/Settings.jl")
include("../../../src/bosonic/iPEPS_ansatz.jl")
include("../../../src/bosonic/AD_lib.jl")



filenm="SimpleUpdate_D_6.jld2";

Random.seed!(1234)


D=6;

trun_tol=1e-6;

chis=[40,80,120,160,320,480,640,320*3,320*4];


ctm_setting=LS_CTMRG_settings();
ctm_setting.CTM_conv_tol=1e-6;
ctm_setting.CTM_ite_nums=100;
ctm_setting.CTM_trun_tol=1e-8;
ctm_setting.svd_lanczos_tol=1e-8;
ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
ctm_setting.conv_check="singular_value";
ctm_setting.CTM_ite_info=true;
ctm_setting.CTM_conv_info=true;
ctm_setting.CTM_trun_svd=false;
ctm_setting.construct_double_layer=true;
dump(ctm_setting);

energy_setting=Kagome_Energy_settings()
energy_setting.kagome_method ="E_triangle";
energy_setting.E_up_method = "2x2";
energy_setting.E_dn_method = "simplified";
energy_setting.cal_chiral_order = false;
dump(energy_setting);


theta=0*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);


global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=ctm_setting.CTM_trun_tol

global backward_settings




data=load(filenm);
B_a=data["B_a"];
B_b=data["B_b"];
B_c=data["B_c"];
T_u=data["T_u"];
T_d=data["T_d"];




@assert dim(space(T_u,1))==D;

@tensor PEPS_tensor[:] := B_a[-1,1,-5]*B_b[4,3,-6]*B_c[-4,2,-7]*T_u[1,3,2]*T_d[-3,4,-2];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

iPESS_tensors=Kagome_iPESS(B_a,B_b,B_c,T_u,T_d);
# norm_A=norm(A_fused)
# A_fused= A_fused/norm_A;
# A_unfused=A_unfused/norm_A;

CTM=[];
U_L=[];
U_D=[];
U_R=[];
U_U=[];

init=Dict([("CTM", []), ("init_type", "PBC")]);
conv_check="singular_value";
CTM_ite_info=true;
CTM_conv_info=true;

CTM_init=[];
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
global chis,D_max,init,A_fused,A_unfused,iPESS_tensors,conv_check,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info,CTM_init,init

for cchi=1:length(chis)
    global chis,D_max,init,A_fused,A_unfused,iPESS_tensors,conv_check,CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info,CTM_init,init

    
    chi=chis[cchi];
    global chi
    println("chi= "*string(chi));flush(stdout);



    
    #CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true,true);
    
    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,CTM_init,ctm_setting)
    E_up, E_down=evaluate_ob(parameters, U_phy, iPESS_tensors, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method, energy_setting.E_up_method, energy_setting.E_dn_method);
    energy=(E_up+E_down)/3;
    println(energy)







    eu_allspin_x,allspin_x=solve_correl_length(5,AA_fused/norm(AA_fused),CTM,"x",ctm_setting);
    eu_allspin_y,allspin_y=solve_correl_length(5,AA_fused/norm(AA_fused),CTM,"y",ctm_setting);


    init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
    CTM_init=deepcopy(CTM);


    matwrite("simpleupdate_ob"*"_D"*string(D)*"_chi"*string(chi)*".mat", Dict(
        "energy" => energy,
        "E_up" =>E_up,
        "E_down" => E_down,
        "eu_allspin_x"=>eu_allspin_x,
        "allspin_x"=>allspin_x,
        "eu_allspin_y"=>eu_allspin_y,
        "allspin_y"=>allspin_y,
        "space_T_u"=>string(space(T_u)),
        "space_T_d"=>string(space(T_d))
    ); compress = false)
end



