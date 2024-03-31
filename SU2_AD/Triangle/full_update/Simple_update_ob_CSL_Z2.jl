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

D_max=6;
t1=1;
t2=1;
ϕ=pi/2;
μ=0;
U=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);
parameters_evolve=Dict([("t1", t1),("t2", t2*1), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);

println("parameters:")
println(parameters)
println("parameters_evolve:")
println(parameters_evolve)

global update_triangle1, update_triangle2
update_triangle1=true;
update_triangle2=false;
println("update_triangle1: "*string(update_triangle1));
println("update_triangle2: "*string(update_triangle2));



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
optim_setting.init_statenm="Optim_cell_LS_D_4_chi_40_2.36814.jld2";#"Optim_cell_LS_D_4_chi_40_2.140901.jld2";#"nothing";
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





##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################
global Lx,Ly
Lx=2;
Ly=2;

state=load(optim_setting.init_statenm);
state=state["x"];
if size(state)==(2,1)
    psi=Matrix{Tensor}(undef,2,2);
    psi[1,1]=state[1,1].T;
    psi[1,2]=state[1,1].T;
    psi[2,1]=state[2,1].T;
    psi[2,2]=state[2,1].T;
elseif size(state["x"])==(2,2)
    psi[1,1]=state[1,1].T;
    psi[1,2]=state[1,2].T;
    psi[2,1]=state[2,1].T;
    psi[2,2]=state[2,2].T;
end






##############


##############





tau=5;
dt=0.1;
# psi=full_update(parameters_evolve, psi, D_max tau, dt);




global Lx,Ly,A_cell
A_cell=initial_tuple_cell(Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        global U_phy,A_cell
        A=psi[cx, cy];
        A_cell=fill_tuple(A_cell, A, cx,cy);
    end
end



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
# filenm="FU_csl_D"*string(D_max)*".jld2";
# jldsave(filenm;x=state)


Hamiltonian_terms=Hamiltonians_spinful_Z2;
Hamiltonian_terms=Hamiltonians_spinless_U1;
Hamiltonian_terms=Hamiltonians_spinful_SU2;
Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;


Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
t1=parameters["t1"];
t2=parameters["t2"];
ϕ=parameters["ϕ"];
μ=parameters["μ"];
U=parameters["U"];

ex_set=zeros(Lx,Ly)*im;
ey_set=zeros(Lx,Ly)*im;
e_diagonala_set=zeros(Lx,Ly)*im;
e0_set=zeros(Lx,Ly)*im;
eU_set=zeros(Lx,Ly)*im;


E_total=0;

cx=1;cy=1;
ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
e0=ob_onsite(CTM_cell,N_occu_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
eU=ob_onsite(CTM_cell,n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
@ignore_derivatives ex_set[cx,cy]=ex;
@ignore_derivatives ey_set[cx,cy]=ey;
@ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
@ignore_derivatives e0_set[cx,cy]=e0;
@ignore_derivatives eU_set[cx,cy]=eU;
E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala')  +U*eU);

