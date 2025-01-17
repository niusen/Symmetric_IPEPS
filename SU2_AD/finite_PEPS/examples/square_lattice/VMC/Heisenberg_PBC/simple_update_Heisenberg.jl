using LinearAlgebra: I,diag,diagm,svd
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
using LineSearches,OptimKit
cd(@__DIR__)


include("../../../setting/Settings.jl")
include("../../../setting/linearalgebra.jl")
include("../../../state/iPEPS_ansatz.jl")
include("../../../setting/tuple_methods.jl")
# include("../../../state/FinitePEPS.jl")
include("../../../symmetry/parity_funs.jl")
include("../../../environment/AD/convert_boundary_condition.jl")
include("../../../environment/AD/mps_methods.jl")
include("../../../environment/AD/mps_methods_new.jl")
include("../../../environment/AD/peps_double_layer_methods.jl")
include("../../../environment/AD/svd_AD_lib.jl")
include("../../../environment/AD/density_matrix.jl")
include("../../../environment/AD/density_matrix_new.jl")
include("../../../environment/extend_bond/extend_bond.jl")
include("../../../environment/extend_bond/environment_2site.jl")
include("../../../models/spin/square_lattice/spin_operator_dense.jl")
include("../../../models/spin/square_lattice/J1_J2_Jchi_disk.jl")
include("../../../optimization/line_search_lib.jl")
include("../../../optimization/PEPS_methods.jl")
# include("../../../optimization/LineSearches/My_Backtracking.jl")

include("../../../environment/simple_update/simple_update_lib.jl")

Random.seed!(888)
global use_AD;
use_AD=false;

global chi
chi=100;

D_max=2;


J1=1;
J2=0;
Jchi=0;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
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




"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

multiplet_tol=1e-5;
global chi,multiplet_tol

global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=0;



# data=load(filenm);
# psi=data["psi"];
# psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

global Lx,Ly
Lx=8;
Ly=8;
# Lx=size(psi,1);
# Ly=size(psi,2);



D_max=2;
d=2;
T_set,lambdax_set,lambday_set=initial_tensor(Lx,Ly,d,D_max);



tau=10;
dt=0.1;
T_set,lambdax_set,lambday_set=simple_update_Heisenberg_OBC(T_set,lambdax_set,lambday_set,tau,dt,D_max)
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(T_set));
E_total,E_set=energy_disk_(psi);
println(E_total)

tau=5;
dt=0.05;
T_set,lambdax_set,lambday_set=simple_update_Heisenberg_OBC(T_set,lambdax_set,lambday_set,tau,dt,D_max)
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(T_set));
E_total,E_set=energy_disk_(psi);
println(E_total)

tau=2;
dt=0.02;
T_set,lambdax_set,lambday_set=simple_update_Heisenberg_OBC(T_set,lambdax_set,lambday_set,tau,dt,D_max)
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(T_set));
E_total,E_set=energy_disk_(psi);
println(E_total)


tau=0.5;
dt=0.005;
T_set,lambdax_set,lambday_set=simple_update_Heisenberg_OBC(T_set,lambdax_set,lambday_set,tau,dt,D_max)
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(T_set));
E_total,E_set=energy_disk_(psi);
println(E_total)


filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D_max)*".jld2";
jldsave(filenm;psi,E=E_total)
####################################
D_max=3;

tau=10;
dt=0.1;
T_set,lambdax_set,lambday_set=simple_update_Heisenberg_OBC(T_set,lambdax_set,lambday_set,tau,dt,D_max)
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(T_set));
E_total,E_set=energy_disk_(psi);
println(E_total)

tau=5;
dt=0.05;
T_set,lambdax_set,lambday_set=simple_update_Heisenberg_OBC(T_set,lambdax_set,lambday_set,tau,dt,D_max)
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(T_set));
E_total,E_set=energy_disk_(psi);
println(E_total)

tau=2;
dt=0.02;
T_set,lambdax_set,lambday_set=simple_update_Heisenberg_OBC(T_set,lambdax_set,lambday_set,tau,dt,D_max)
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(T_set));
E_total,E_set=energy_disk_(psi);
println(E_total)


tau=0.5;
dt=0.005;
T_set,lambdax_set,lambday_set=simple_update_Heisenberg_OBC(T_set,lambdax_set,lambday_set,tau,dt,D_max)
psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(T_set));
E_total,E_set=energy_disk_(psi);
println(E_total)

filenm="Heisenberg_SU_"*string(Lx)*"x"*string(Ly)*"_D"*string(D_max)*".jld2";
jldsave(filenm;psi,E=E_total)