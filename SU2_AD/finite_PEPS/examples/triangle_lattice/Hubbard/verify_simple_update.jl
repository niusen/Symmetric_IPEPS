using LinearAlgebra:diag,I,diagm 
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\fermion\\peps_double_layer_methods_fermion.jl")
include("..\\..\\..\\environment\\AD\\fermion\\fermi_CTM_observables.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\models\\Hubbard\\triangle_lattice\\Hofstadter_N2.jl")

include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_methods.jl")
include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_simple_update.jl")

D=4;


global use_AD;
use_AD=false;

t1=1;
t2=1;
ϕ=pi/2;
μ=0;
U=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("U",  U)]);
global parameters


svd_settings=Svd_settings();
svd_settings.svd_trun_method="chi";#chi" or "tol"
svd_settings.chi_max=500;
svd_settings.tol=1e-5;
dump(svd_settings);

backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);
global svd_settings, backward_settings



#Hamiltonian
# H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=4;
Ly=4;

data=load("FU_iPESS_LS_D_4_chi_40_2.17125.jld2");
# data=load("SU_iPESS_SU2_csl_D6_2.259.jld2")
T_virt_set=data["B_set"];
T_phy_set=data["T_set"];



psi=get_PESS_from_iPESS(T_phy_set,T_virt_set,Lx,Ly);

B_set,T_set=PESS_to_B_T_sets(psi);


psi=B_T_sets_to_PESS(B_set,T_set);



psi_PEPS=PESS_to_PEPS_matrix(psi);


# psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

# filenm="CSL_D3_Lx"*string(Lx)*"_Ly"*string(Ly)*".jld2";
# jldsave(filenm;psi);

psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi_PEPS,psi_PEPS,Lx,Ly);


multiplet_tol=1e-5;
chi=100;

global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;


E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_old(psi_PEPS,psi_double)
#-sum(imag.(Ex_set*2))-sum(abs.(real.(Ey_set)))*2-sum(abs.(real.(E_ld_ru_set)))*2+(sum(EU_set)*U)
println(E_total)




#step 1: 
# replace gate by Hamiltonian, verify energies of each triangle, but this is hard to do due to renormalization  of tensors during tebd


#step 2:
#set dt=0, check the state and energies don't change

#step 3:
#do evolution on each single terms to verify evolution code 



#Do step1:

lambdaset1,lambdaset2,lambdaset3=get_trivial_lambda(B_set);

Dmax=4;
tau=1;
dt=1;
trun_tol=1e-8;




verif2(parameters,Dmax, B_set, T_set, trun_tol)



#verif1(parameters,Dmax,psi,B_set, T_set, trun_tol)




Dmax=8;
tau=1;
dt=0.02;
B_set, T_set, lambdaset1, lambdaset2, lambdaset3=tebd_PESS(parameters, B_set, T_set, lambdaset1, lambdaset2, lambdaset3,  tau, dt, Dmax, trun_tol)
psi=B_T_sets_to_PESS(B_set,T_set);
psi_PEPS=PESS_to_PEPS_matrix(psi);
psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi_PEPS,psi_PEPS,Lx,Ly);
E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_old(psi_PEPS,psi_double);
println(E_total)