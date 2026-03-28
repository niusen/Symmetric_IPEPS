using Revise, TensorKit, Zygote
using LinearAlgebra:diag,I,diagm 
using ChainRulesCore
using KrylovKit
using JSON, MAT, HDF5, JLD2
using Zygote:@ignore_derivatives
using Random
using Dates

cd(@__DIR__)
include("../../src/bosonic/Settings.jl")
include("../../src/bosonic/iPEPS_ansatz.jl")
include("../../src/bosonic/kagome_load_tensor.jl")
include("../../src/bosonic/kagome_IPESS.jl")
# include("../../src/bosonic/kagome/kagome_CTMRG.jl")
include("../../src/bosonic/kagome_model.jl")
include("../../src/bosonic/kagome_IPESS.jl")
include("../../src/bosonic/kagome_FiniteDiff.jl")
include("../../src/bosonic/kagome_correl.jl")


include("../../src/bosonic/CTMRG.jl")
include("../../src/bosonic/kagome_correl.jl")
include("../../src/bosonic/AD_lib.jl")
include("../../src/bosonic/line_search_lib.jl")
include("../../src/bosonic/line_search_lib_cell.jl")
include("../../src/bosonic/kagome_AD_SU2.jl")
include("../../src/fermionic/simple_update/fermi_triangle_SimpleUpdate.jl")



include("../../src/bosonic/kagome/funs_1up_1down.jl")


Random.seed!(1234)


D_max=8;
symmetric_hosvd=false;
itebd_trun_tol=1e-6;
D=3;

println("D_max= "*string(D_max))

chis=[40,60];
#chis=[40,80,120,160];

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


optim_setting=Optim_settings();
optim_setting.init_statenm="nothing";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);

energy_setting=Kagome_Energy_settings()
energy_setting.kagome_method ="E_triangle";
energy_setting.E_up_method = "2x2";
energy_setting.E_dn_method = "open_leg";#"open_leg", "simplified"
energy_setting.cal_chiral_order = false;
dump(energy_setting);





theta=0*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);

global Lx,Ly
Lx=1;
Ly=1;
energy_setting.Lx=Lx;
energy_setting.Ly=Ly;


global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=ctm_setting.CTM_trun_tol

global backward_settings

global Vv

Vp=Rep[SU₂](1/2=>1);

if D==3
    Vv=SU2Space(0=>1,1/2=>1);
elseif D==6
    Vv=SU2Space(0=>1,1/2=>1,1=>1);
elseif D==8
    Vv=SU2Space(0=>1,1/2=>2,1=>1);
elseif D==12
    Vv=SU2Space(0=>1,1/2=>2,1=>1,3/2=>1);
elseif D==13
    Vv=SU2Space(0=>2,1/2=>2,1=>1,3/2=>1);    
elseif D==16
    Vv=SU2Space(0=>2,1/2=>2,1=>2,3/2=>1);
elseif D==18
    Vv=SU2Space(0=>2,1/2=>3,1=>2,3/2=>1);
elseif Vv==23
    Vv=SU2Space(0=>2,1/2=>3,1=>2,3/2=>1,2=>1);
end
@assert dim(Vv)==D;

global starting_time
starting_time=now();

is_complex=false;
state_vec=initial_SU2_state(Vv, optim_setting.init_statenm, optim_setting.init_noise, is_complex);
state_vec=normalize_ansatz(state_vec);




# B_a=state_vec.B1;
# B_b=state_vec.B2;
# B_c=state_vec.B3;
# T_u=state_vec.Tup;
# T_d=state_vec.Tdn;



B_a=state_vec.B1;
B_a=B_a-permute(B_a,(2,1,3,));
B_b=deepcopy(B_a);
B_c=deepcopy(B_a);

T_u=state_vec.Tup;
T_u=T_u+permute(T_u,(2,3,1,))+permute(permute(T_u,(2,3,1,)),(2,3,1,));
T_d=deepcopy(T_u);

lambda_u_a=unitary(space(B_a,2)',space(B_a,2)');
lambda_u_a=lambda_u_a'*lambda_u_a;
lambda_u_b=deepcopy(lambda_u_a);
lambda_u_c=deepcopy(lambda_u_a);
lambda_d_a=nothing;
lambda_d_b=nothing;
lambda_d_c=nothing;

# B_a=B_a+TensorMap(randn,codomain(B_a),domain(B_a))*0.1;
# B_b=B_b+TensorMap(randn,codomain(B_b),domain(B_b))*0.1;
# B_c=B_c+TensorMap(randn,codomain(B_c),domain(B_c))*0.1;
# T_u=T_u+TensorMap(randn,codomain(T_u),domain(T_u))*0.1;
# T_d=T_d+TensorMap(randn,codomain(T_d),domain(T_d))*0.1;


U_d=Vp';
U_phy_3=unitary(fuse(U_d ⊗ U_d ⊗ U_d), U_d ⊗ U_d ⊗ U_d);
U_phy_2=unitary(fuse(U_d ⊗ U_d), U_d ⊗ U_d);
H_triangle, H_Heisenberg, H12_tensorkit, H31_tensorkit, H23_tensorkit=Hamiltonians(U_phy_3,parameters["J1"],parameters["J2"],parameters["J3"],parameters["Jchi"],parameters["Jtrip"])
@tensor H_triangle[:]:=U_phy_3[2,-1,-2,-3]*H_triangle[2,1]*U_phy_3'[-4,-5,-6,1];
H_triangle=permute(H_triangle,(1,2,3,),(4,5,6,));
H_Heisenberg=TensorMap(H_Heisenberg, U_d' ⊗ U_d' ← U_d' ⊗ U_d');

tau=5;
dt=0.01;
T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, itebd_trun_tol, tau, dt, D_max,symmetric_hosvd);

# tau=2;
# dt=0.01;
# T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, itebd_trun_tol, tau, dt, D_max,symmetric_hosvd);

# tau=1;
# dt=0.005;
# T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c=itebd(T_u,T_d,B_a,B_b,B_c,lambda_u_a,lambda_u_b,lambda_u_c,lambda_d_a,lambda_d_b,lambda_d_c, H_triangle, itebd_trun_tol, tau, dt, D_max,symmetric_hosvd);


println(space(T_u))
println(space(T_d))

filenm="SimpleUpdate_D_"*string(D_max)*".jld2"
jldsave(filenm; B_a,B_b,B_c,T_u,T_d);

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
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
#for cchi=1:length(chis)
    cchi=1;
    global chis,D_max,init,A_fused
    
    chi=chis[cchi];
    println("chi= "*string(chi));flush(stdout);
    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,nothing,ctm_setting);

    E_up, E_dn=evaluate_ob(parameters, U_phy,state_vec, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting,energy_setting);
    energy=(E_up+E_dn)/3;
    println("Up triangle energy: "*string(energy))

    eu_allspin_x,allspin_x=solve_correl_length(5,[],CTM,"x",ctm_setting);
    eu_allspin_y,allspin_y=solve_correl_length(5,[],CTM,"y",ctm_setting);

    init=Dict([("CTM", CTM), ("init_type", "PBC"),("AA_fused",AA_fused),("U_L",U_L),("U_R",U_R),("U_U",U_U),("U_D",U_D)]);

    matwrite("SimpleUpdate_ob"*"_D"*string(D_max)*"_chi"*string(chi)*".mat", Dict(
        "energy" => energy,
        "eu_allspin_x"=>eu_allspin_x,
        "allspin_x"=>allspin_x,
        "eu_allspin_y"=>eu_allspin_y,
        "allspin_y"=>allspin_y,
        "space_T_u"=>string(space(T_u)),
        "space_T_d"=>string(space(T_d))
    ); compress = false)
#end



