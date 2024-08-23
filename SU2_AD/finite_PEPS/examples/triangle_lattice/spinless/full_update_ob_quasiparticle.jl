using LinearAlgebra:diag,I,diagm 
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
cd(@__DIR__)

include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\..\\..\\src\\fermionic\\simple_update\\fermi_triangle_FullUpdate_iPESS.jl")
include("..\\..\\..\\..\\src\\fermionic\\simple_update\\Full_Update_lib.jl")

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\fermion\\peps_double_layer_methods_fermion.jl")
include("..\\..\\..\\environment\\AD\\fermion\\fermi_CTM_observables_spinless.jl")
include("..\\..\\..\\environment\\AD\\fermion\\fermi_contract.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\models\\Hubbard\\triangle_lattice\\Hofstadter_N2_spinless.jl")

include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_methods.jl")
include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_simple_update.jl")
include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_full_update.jl")

Dmax=6;
println("D="*string(Dmax));


####################
import LinearAlgebra.BLAS as BLAS
n_cpu=6;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);
Base.Sys.set_process_title("C"*string(n_cpu)*"_FU_D"*string(Dmax))
pid=getpid();
println("pid="*string(pid));;flush(stdout);
####################

global use_AD;
use_AD=false;

t1=1;
t2=1;
ϕ=pi/2;
μ=0;
V=0;
parameters=Dict([("t1", t1),("t2", t2), ("ϕ", ϕ), ("μ",  μ), ("V",  V)]);
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

data=load("FU_PESS_U1_D6_15.14566.jld2");
# data=load("SU_PESS_U1_D4.jld2");
B_set=data["B_set"];
T_set=data["T_set"];


psi=B_T_sets_to_PESS(B_set,T_set);
B_set,T_set=PESS_to_B_T_sets(psi);

psi_PEPS=PESS_to_PEPS_matrix(psi);


# psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

# filenm="CSL_D3_Lx"*string(Lx)*"_Ly"*string(Ly)*".jld2";
# jldsave(filenm;psi);

psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi_PEPS,Lx,Ly);


multiplet_tol=1e-5;
chi=100;

global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;


E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_old(psi_PEPS,psi_double)
#-sum(imag.(Ex_set*2))-sum(abs.(real.(Ey_set)))*2-sum(abs.(real.(E_ld_ru_set)))*2+(sum(EU_set)*U)
println(E_total);flush(stdout);




########################################
space_type=typeof(space(psi_PEPS[1,1],1));
if space_type==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
    Ident, N_occu, Cdag, C, _ =operators_spinless_Z2();
elseif space_type==GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}
    Ident, N_occu, Cdag, C, _ =operators_spinless_U1();
end


px=1;
py=1;
#Cdag:Dnew,d,d'
#T_set[1,1]: M,d',R,D
T1=T_set[px,py];
@tensor T1new[:]:=T1[-1,1,-4,-5]*Cdag[-2,-3,1];#M,Dnew,d,R,D
T1new=permute_neighbour_ind(T1new,2,3,5);#M,d,Dnew,R,D
T1new=permute_neighbour_ind(T1new,3,4,5);#M,d,R,Dnew,D
uni=unitary(fuse(space(T1new,4)*space(T1new,5)), space(T1new,4)*space(T1new,5));
@tensor T1new[:]:=T1new[-1,-2,-3,1,2]*uni[-4,1,2];#M,d,R,Dnew
T1new=permute(T1new,(1,),(2,3,4,));
T_set[px,py]=T1new;




psi=B_T_sets_to_PESS(B_set,T_set);
B_set,T_set=PESS_to_B_T_sets(psi);
psi_PEPS=PESS_to_PEPS_matrix(psi);
psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi_PEPS,Lx,Ly);

@time E_total,Ex_set,Ey_set,E_ld_ru_set, NNx_set,NNy_set,NN_ld_ru_set, occu_set=energy_disk_old(psi_PEPS,psi_double)
#-sum(imag.(Ex_set*2))-sum(abs.(real.(Ey_set)))*2-sum(abs.(real.(E_ld_ru_set)))*2+(sum(EU_set)*U)
println(E_total);flush(stdout);
############################################


global starting_time
starting_time=now();



trun_tol=1e-8;
ov_sweep=10;



####################################

tau=10;
dt=0.02;

trotter_order=2;
save_filenm="trotter"*string(trotter_order)*"_FU_PESS_U1_"*string(Lx)*"x"*string(Ly)*"_D"*string(Dmax)*"_dt"*string(dt);

B_set, T_set=Full_update_real_time_PESS_spinless(save_filenm, parameters, B_set, T_set,  tau, dt, Dmax, trun_tol, ov_sweep,trotter_order)
# B_set, T_set=Full_update_PESS_spinless(parameters, B_set, T_set,  tau, dt, Dmax, trun_tol, ov_sweep)











