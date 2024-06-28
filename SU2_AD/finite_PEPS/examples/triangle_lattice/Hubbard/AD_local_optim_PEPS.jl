using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
using LineSearches,OptimKit
cd(@__DIR__)

include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\..\\..\\src\\bosonic\\line_search_lib_cell.jl")

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\state\\FinitePEPS.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\fermion\\peps_double_layer_methods_fermion.jl")
include("..\\..\\..\\environment\\AD\\fermion\\fermi_CTM_observables.jl")
include("..\\..\\..\\environment\\AD\\fermion\\fermi_contract.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\models\\Hubbard\\triangle_lattice\\Hofstadter_N2.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\optimization\\LineSearches\\My_Backtracking.jl")

include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_methods.jl")
include("..\\..\\..\\environment\\simple_update\\fermionic\\triangle_PESS_simple_update.jl")

include("..\\..\\..\\environment\\simple_update\\fermionic\\gauge_fix.jl")


Random.seed!(666)

global D,chi,multiplet_tol

D=4;
chi=100;
multiplet_tol=1e-5;
init_noise=0;

filenm="SU_PESS_SU2_D4.jld2";
# filenm="sweep_4x4_D_4_chi_100.jld2";

println("D,chi="*string([D,chi]));
println("init_noise="*string(init_noise));



####################
import LinearAlgebra.BLAS as BLAS
n_cpu=6;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()));flush(stdout);
Base.Sys.set_process_title("C"*string(n_cpu)*"_sweep_D"*string(D))
pid=getpid();
println("pid="*string(pid));flush(stdout);
####################

global use_AD;
use_AD=true;



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


psi=initial_SU2_PESS(filenm,init_noise,true);

Lx,Ly=size(psi);
println("Lx,Ly="*string([Lx,Ly]))





global psi,psi_double

if isa(psi[1,1],Triangle_iPESS)
    psi=PESS_to_PEPS_matrix(psi);
end
psi=normalize_tensor_group(psi);


use_canonical_form=true;

global use_canonical_form
if use_canonical_form
    println("convert to canonical form")
    psi,_=fermiPEPS_gauge_fix_simple(psi,100);
    psi_double,_=construct_double_layer_swap_new(psi,Lx,Ly);
end
# psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap_new(psi,Lx,Ly);


global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;

global ppx,ppy


E=cost_fun_global(psi);
println("E= "*string(E));


n_mps_sweep=0;


save_opt_filenm="sweep_"*string(Lx)*"x"*string(Ly)*"_D_"*string(D)*"_chi_"*string(chi)*".jld2"
global save_opt_filenm

global starting_time
starting_time=now();



########################################


global E_history
E_history=[E];


#########################################
E_opt=E;

# ls = BackTracking(order=3)
ls = BackTracking(c_1=0.0001,ρ_hi=0.5,ρ_lo=0.1,iterations=7,order=3,maxstep=Inf);
println(ls)

optim_maxiter=5;
LS_maxiter=10;#number of gradient optimization for each site
grad_tol=1e-3;




# global ppx,ppy
# ppx=1;
# ppy=2;

# psi=normalize_tensor_group(psi);
# E_opt_new, T, iter_bt3 = gdoptimize(f, g!, fg!, psi[ppx,ppy]::TensorMap, ls,LS_maxiter, 1e-8, grad_tol);
# if E_opt_new<E_opt
#     psi[ppx,ppy]=T;
#     E_opt=E_opt_new;
# end


##########################################


for ite=1:optim_maxiter
    println("Optimization iteration: "*string(ite));
    for cx=1:Lx
        for cy=1:Ly
            global E_opt,psi,psi_double,ppx,ppy
            
            ppx=cx;
            ppy=cy;
            println("coordinate: "*string([ppx,ppy]));
            # try
                E_opt_new, T, iter_bt3 = gdoptimize(f, g!, fg!, psi[ppx,ppy]::TensorMap, ls,LS_maxiter, 1e-8, grad_tol);
            # catch e
            #     continue
            # end
            if E_opt_new<E_opt
                psi[ppx,ppy]=T;
                #psi=gauge_fix_global(psi,1,false);
                psi[ppx,ppy]=psi[ppx,ppy]/norm(psi[ppx,ppy]);
                psi_double=construct_double_layer_swap_position(psi,psi_double,ppx,ppy,Lx,Ly);
                E_opt=E_opt_new;
                println("Energy of updated state: "*string(E_opt));flush(stdout);
            else
                println("Energy not improved, change to next site")
            end
            global use_canonical_form
            if use_canonical_form
                println("convert to canonical form")
                psi,_=fermiPEPS_gauge_fix_simple(psi,100);
                psi_double,_=construct_double_layer_swap_new(psi,Lx,Ly);
            end

        end
    end
    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_global(psi::Matrix,psi_double::Matrix);
    println(Ex_set)
    println(Ey_set)
    println(E_ld_ru_set)
    println(occu_set)
    println(EU_set)
end


