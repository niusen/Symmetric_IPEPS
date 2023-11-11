using Revise, PEPSKit, TensorKit, TensorKitAD, Zygote, MPSKit
using LinearAlgebra, OptimKit
#using PEPSKit: NORTH,SOUTH,WEST,EAST,NORTHWEST,NORTHEAST,SOUTHEAST,SOUTHWEST,@diffset
using JLD2,ChainRulesCore
using KrylovKit
using JSON
using Random
using LineSearches
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)

include("..\\src\\checkerboard_spin_operator.jl")
include("..\\src\\iPEPS_ansatz.jl")
include("..\\src\\CTMRG.jl")
include("..\\src\\CTMRG_unitcell.jl")
include("..\\src\\checkerboard_model_cell.jl")
include("..\\src\\checkerboard_AD_SU2_cell.jl")
include("..\\src\\Settings.jl")
include("..\\src\\Settings_cell.jl")
include("..\\src\\AD_lib.jl")
include("..\\src\\line_search_lib.jl")
include("..\\src\\line_search_lib_cell.jl")
include("..\\src\\optimkit_lib.jl")

include("..\\src\\checkerboard_SimpleUpdate_lib.jl")
include("..\\src\\checkerboard_plaquatte_ansatz.jl")




Random.seed!(1234)
symmetric_initial=false;
J1=1;
J2=1;
D_max=6;
symmetric_hosvd=false;
trun_tol=1e-6;
D=2;

println("D_max= "*string(D_max))

chi=40;

"Unit-cell format:
ABABAB
CDCDCD
ABABAB
CDCDCD


A11  A21
A12  A22


actual unit-cell:
ABAB
BABA
ABAB
BABA
"

###################################
global Lx,Ly
Lx=2;
Ly=2;

#state_vec=plaquatte_empty()
state_vec=plaquatte_cross()
###################################
T_d=state_vec[1,1].Tm;
B_a=state_vec[1,1].B_L;
B_d=state_vec[1,1].B_U;
B_a=permute(B_a,(2,1,3));
B_d=permute(B_d,(2,1,3));

T_u=state_vec[1,2].Tm;
B_c=state_vec[1,2].B_L;
B_b=state_vec[1,2].B_U;


######################

lambda_u_a=unitary(space(B_a,2),space(B_a,2));
lambda_u_b=unitary(space(B_b,2),space(B_b,2));
lambda_u_c=unitary(space(B_c,2),space(B_c,2));
lambda_u_d=unitary(space(B_d,2),space(B_d,2));

lambda_d_a=unitary(space(B_a,1),space(B_a,1));
lambda_d_b=unitary(space(B_b,1),space(B_b,1));
lambda_d_c=unitary(space(B_c,1),space(B_c,1));
lambda_d_d=unitary(space(B_d,1),space(B_d,1));



U_d=space(B_a,3);
U_phy_2=unitary(fuse(U_d ⊗ U_d), U_d ⊗ U_d);
H_plaquatte,_=plaquatte_Heisenberg(J1,J2);


tau=5;
dt=0.1;
T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d=itebd(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, H_plaquatte,U_phy_2, trun_tol, tau, dt, D_max,symmetric_hosvd);

tau=2;
dt=0.05;
T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d=itebd(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, H_plaquatte,U_phy_2, trun_tol, tau, dt, D_max,symmetric_hosvd);

tau=0.2;
dt=0.01;
T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d=itebd(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, H_plaquatte,U_phy_2, trun_tol, tau, dt, D_max,symmetric_hosvd);


println(space(T_u))
println(space(T_d))



##############
B_a_=permute(B_a,(2,1,3));
B_d_=permute(B_d,(2,1,3));
state_vec[1,1]=Checkerboard_iPESS(B_a_,B_d_,T_d);
state_vec[1,2]=Checkerboard_iPESS(B_c,B_b,T_u);
state_vec[2,2]=state_vec[1,1];
state_vec[2,1]=state_vec[1,2];

##############


global Lx,Ly,U_phy,A_unfused_cell,A_fused_cell
A_unfused_cell=initial_tuple_cell(Lx,Ly);
A_fused_cell=initial_tuple_cell(Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        global U_phy
        A_unfused,A_fused,U_phy=build_A_checkerboard(state_vec[cx, cy]);
        A_unfused_cell=fill_tuple(A_unfused_cell, A_unfused, cx,cy);
        A_fused_cell=fill_tuple(A_fused_cell, A_fused, cx,cy);
    end
end

global chi, parameters, energy_setting, grad_ctm_setting
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_fused_cell,chi,init,[],LS_ctm_setting)

E_plaquatte_cell=bond_energy(U_phy, state_vec, A_fused_cell::Tuple, AA_fused_cell, CTM_cell, LS_ctm_setting);

filenm="SU_D_"*string(D_max)*".jld2"
jldsave(filenm;x=state_vec);

mat_filenm="SU_D_"*string(D_max)*".mat"
matwrite(mat_filenm, Dict(
    "E_plaquatte_cell" => E_plaquatte_cell,
    "space_Tu" => string(codomain(T_u)),
    "space_Td" => string(codomain(T_d))
); compress = false)


