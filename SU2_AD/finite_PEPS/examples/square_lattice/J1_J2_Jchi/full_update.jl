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

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\state\\FinitePEPS.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\extend_bond\\extend_bond.jl")
include("..\\..\\..\\environment\\extend_bond\\environment_2site.jl")
include("..\\..\\..\\environment\\full_update\\full_update_lib.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\triangle_terms.jl")
include("..\\..\\..\\optimization\\PEPS_methods.jl")

# let
Random.seed!(888)
global use_AD;
use_AD=false;

global chi,Dmax
chi=100;
Dmax=6;
filenm="optim_4x4_D_3_chi_100_13.89035.jld2";



J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
global parameters


svd_settings=Svd_settings();
svd_settings.svd_trun_method="tol";#chi" or "tol"
svd_settings.chi_max=500;
svd_settings.tol=1e-4;
dump(svd_settings);


global svd_settings




"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""




init_noise=0;
psi=initial_SU2_state(filenm,init_noise,true);
psi0=deepcopy(psi);

global Lx,Ly
Lx=size(psi,1);
Ly=size(psi,2);



global E_history
E_history=[10000];


save_opt_filenm="FU_"*string(Lx)*"x"*string(Ly)*"_Dmax_"*string(Dmax)*"_chi_"*string(chi)*".jld2"
global save_opt_filenm

global starting_time
starting_time=now();


multiplet_tol=1e-5;



global chi,multiplet_tol


global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;

E_opt=real(cost_fun_global(psi0));
println("initial energy: "*string(E_opt))
global E_opt





#########################################



dt=0.1;
n_lattice_sweep=1;
for ci=1:10
    psi_new,E_opt_new=full_update(parameters,dt,psi,Dmax,n_lattice_sweep,"forward");
    if E_opt_new<E_opt
        psi=psi_new;
        E_opt=E_opt_new;
        jldsave(save_opt_filenm; psi=psi);
    end
    psi_new,E_opt_new=full_update(parameters,dt,psi,Dmax,n_lattice_sweep,"backward");
    if E_opt_new<E_opt
        psi=psi_new;
        E_opt=E_opt_new;
        jldsave(save_opt_filenm; psi=psi);
    end
end


dt=0.02;
n_lattice_sweep=1;
for ci=1:10
    psi_new,E_opt_new=full_update(parameters,dt,psi,Dmax,n_lattice_sweep,"forward");
    if E_opt_new<E_opt
        psi=psi_new;
        E_opt=E_opt_new;
        jldsave(save_opt_filenm; psi=psi);
    end
    psi_new,E_opt_new=full_update(parameters,dt,psi,Dmax,n_lattice_sweep,"backward");
    if E_opt_new<E_opt
        psi=psi_new;
        E_opt=E_opt_new;
        jldsave(save_opt_filenm; psi=psi);
    end
end



println(E_new)

# end



