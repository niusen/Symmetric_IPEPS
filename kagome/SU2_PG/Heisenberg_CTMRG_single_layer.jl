using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_CTMRG_unitcell.jl")
include("kagome_model.jl")
include("kagome_model_cell.jl")
include("kagome_IPESS.jl")
include("kagome_FiniteDiff.jl")

Random.seed!(1234)

D=8;

theta=0*pi;
J1=cos(theta);
J2=0;
J3=0;
Jchi=0;
Jtrip=sin(theta);

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);

#state_dict=read_json_state("LS_D_8_chi_40.json")
#init_statenm=nothing;
init_statenm="LS_A1even_U1_D_8_chi_60.json"
#init_statenm=nothing
init_noise=0;
CTM_conv_tol=1e-6;
CTM_ite_nums=100;
CTM_trun_tol=1e-12;
Bond_irrep="A";
Triangle_irrep="A1+iA2";
#nonchiral="A1_even";
nonchiral="No"


state_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, _, _=initial_state(Bond_irrep,Triangle_irrep,nonchiral,D,init_statenm,init_noise);



A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
    
bond_tensor,triangle_tensor=construct_su2_PG_IPESS(state_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);


PEPS_tensor=bond_tensor;
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
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
chi=40;
@time CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);

E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
#E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_bond");
energy=(E_up*2)/3;
println(energy)

#return energy,CTM,U_L,U_D,U_R,U_U

chiral_order_parameters=Dict([("J1", 0), ("J2", 0), ("J3", 0), ("Jchi", 0), ("Jtrip", 1)]);
chiral_order_up, chiral_order_down=evaluate_ob(chiral_order_parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");


# Lx=2;Ly=2;
# A_cell=Matrix(undef,Lx,Ly);
# A_cell[1,1]=A_fused;
# A_cell[1,2]=A_fused;
# A_cell[2,1]=A_fused;
# A_cell[2,2]=A_fused;

# init=Dict([("CTM", []), ("init_type", "PBC")]);
# conv_check="singular_value";
# CTM_ite_info=true
# CTM_conv_info=true;
# CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG_cell(A_cell,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);



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

Lx=2;Ly=2;
A_cell=Matrix(undef,Lx,Ly);
A_cell[1,1]=A_LU;
A_cell[2,1]=A_RU;
A_cell[1,2]=A_LD;
A_cell[2,2]=A_RD;

#########################

CTM_ite_nums=200;
init=Dict([("CTM", []), ("init_type", "single_layer_random")]);
conv_check="singular_value";
CTM_ite_info=true
CTM_conv_info=true;
chi=80;
@time CTM_chi_20, AA_fused_cell,U_L_cell,U_D_cell,U_R_cell,U_U_cell=CTMRG_cell(A_cell,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);


#######################

CTM_ite_nums=200;
init=Dict([("CTM", CTM_chi_20), ("init_type", "single_layer_random"),("AA_fused_cell",AA_fused_cell),("U_L_cell",U_L_cell),("U_R_cell",U_R_cell),("U_U_cell",U_U_cell),("U_D_cell",U_D_cell)]);
conv_check="singular_value";
CTM_ite_info=true
CTM_conv_info=true;
chi=120;
@time CTM, _, _,_,_,_=CTMRG_cell(A_cell,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);


method1="E_triangle";
method2="full_cell";
E_up=evaluate_ob_UpTriangle_single_layer(parameters, U_phy, U_D_phy, A_cell, CTM, method1, method2);
energy=E_up*2/3;
println(energy)

method1="E_triangle";
method2="reduced_cell";
E_up=evaluate_ob_UpTriangle_single_layer(parameters, U_phy, U_D_phy, A_cell, CTM, method1, method2);
energy=E_up*2/3;
println(energy)

###########################


