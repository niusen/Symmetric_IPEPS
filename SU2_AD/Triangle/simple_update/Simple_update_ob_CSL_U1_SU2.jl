using Revise, TensorKit, Zygote
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)




include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate.jl")

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################

Random.seed!(1234)
symmetric_initial=false;


t1=1;
t2=1;
ϕ=pi/2;
μ=0;
U=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);
parameters_evolve=Dict([("t1", t1),("t2", t2*1.5), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);

D_max=6;


global D_max, SU_trun_tol
SU_trun_tol=1e-8;
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

algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings

optim_setting=Optim_settings();
optim_setting.init_statenm="SU2_cell_LS_D_4_chi_40_2.368055.jld2";#"Optim_cell_LS_D_4_chi_40_2.140901.jld2";#"nothing";
optim_setting.init_noise=0.0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);


LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=50;
LS_ctm_setting.CTM_trun_tol=1e-8;
LS_ctm_setting.svd_lanczos_tol=1e-8;
LS_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
LS_ctm_setting.conv_check="singular_value";
LS_ctm_setting.CTM_ite_info=true;
LS_ctm_setting.CTM_conv_info=true;
LS_ctm_setting.CTM_trun_svd=false;
LS_ctm_setting.construct_double_layer=true;
LS_ctm_setting.grad_checkpoint=true;
dump(LS_ctm_setting);

energy_setting=Square_Hubbard_Energy_settings();
energy_setting.model = "spinful_triangle_lattice";
dump(energy_setting);



##################################
"""
       /| s3
      / |
     /  |
    /   |
 s2 ----- s1
"""

VDummytype=2;
global VDummy_set

if VDummytype==1
    VDummy1=Rep[U₁ × SU₂]((-1, 1/2)=>1);
    VDummy2=Rep[U₁ × SU₂]((-1, 1/2)=>1);
    VDummy_set=(VDummy1,VDummy2,);
elseif VDummytype==2
    VDummy1=Rep[U₁ × SU₂]((-2, 0)=>1);
    VDummy2=ProductSpace{GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}, 0}();
    VDummy_set=(VDummy1,VDummy2,);
end




##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;

data=load(optim_setting.init_statenm);
x=data["x"];
function init_x(x)
    if size(x)==(2,1)
        TA=deepcopy(x[1].T);
        TB=deepcopy(x[2].T);
        TC=deepcopy(x[1].T);
        TD=deepcopy(x[2].T);
    elseif size(x)==(2,2)
        TA=deepcopy(x[1][1].T);
        TB=deepcopy(x[2][1].T);
        TC=deepcopy(x[1][2].T);
        TD=deepcopy(x[2][2].T);
    end
    return TA,TB,TC,TD
end

TA,TB,TC,TD=init_x(x);


λ_A_L=unitary(space(TA,1)',space(TA,1)');
λ_A_D=unitary(space(TA,2)',space(TA,2)'); 
λ_A_R=unitary(space(TA,3)',space(TA,3)');
λ_A_U=unitary(space(TA,4)',space(TA,4)');
λ_D_L=unitary(space(TD,1)',space(TD,1)');
λ_D_D=unitary(space(TD,2)',space(TD,2)'); 
λ_D_R=unitary(space(TD,3)',space(TD,3)');
λ_D_U=unitary(space(TD,4)',space(TD,4)');


######################


println(space(TA))
println(space(TB))
println(space(TC))
println(space(TD))


##############
state_vec=Matrix{Square_iPEPS}(undef,2,2);
state_vec[1,1]=Square_iPEPS(TA);
state_vec[1,2]=Square_iPEPS(TC);
state_vec[2,1]=Square_iPEPS(TB);
state_vec[2,2]=Square_iPEPS(TD);

##############


global Lx,Ly,A_cell
A_cell=initial_tuple_cell(Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        global U_phy,A_cell
        A=state_vec[cx, cy].T;
        A_cell=fill_tuple(A_cell, A, cx,cy);
    end
end

# global chi, parameters, energy_setting, grad_ctm_setting
# init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

# CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);


# E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
# Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonians_spinful_U1_SU2();



tau=5;
dt=0.1;
TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U=itebd(parameters_evolve, TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U, tau, dt);

tau=1;
dt=0.05;
TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U=itebd(parameters_evolve, TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U, tau, dt);


tau=0.2;
dt=0.01;
TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U=itebd(parameters_evolve, TA, TB, TC, TD, λ_A_L, λ_A_D, λ_A_R, λ_A_U, λ_D_L, λ_D_D, λ_D_R, λ_D_U, tau, dt);


Lx=2;
Ly=2;
@tensor A_D[:]:=TD[1,2,3,4,-5]*λ_D_L[1,-1]*λ_D_D[2,-2]*λ_D_R[3,-3]*λ_D_U[4,-4];
@tensor A_A[:]:=TA[1,2,3,4,-5]*λ_A_L[1,-1]*λ_A_D[2,-2]*λ_A_R[3,-3]*λ_A_U[4,-4];
A_B=TB;
A_C=TC;
A_cell=initial_tuple_cell(2,2);
A_cell=fill_tuple(A_cell, A_A, 1,1);
A_cell=fill_tuple(A_cell, A_B, 2,1);
A_cell=fill_tuple(A_cell, A_C, 1,2);
A_cell=fill_tuple(A_cell, A_D, 2,2);
state=Matrix{Square_iPEPS}(undef,Lx,Ly);
state[1,1]=Square_iPEPS(A_A);
state[2,1]=Square_iPEPS(A_B);
state[1,2]=Square_iPEPS(A_C);
state[2,2]=Square_iPEPS(A_D);

init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
LS_ctm_setting.CTM_ite_nums=10;
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);
E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
println(E_total)
println(ex_set)
println(ey_set)
println(e_diagonala_set)
println(e0_set)
println(eU_set)
filenm="SU_csl_D"*string(D_max)*".jld2";
jldsave(filenm;x=state)
