using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
cd(@__DIR__)

include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\state\\FinitePEPS.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")


global use_AD;
use_AD=true;

global chi,D
chi=80;
D=3;

J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);


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

Lx=4;
Ly=4;

filenm="CSL_D3_Lx4_Ly4.jld2";
filenm="test.jld2";
data=load(filenm);


psi=data["PEPS_init"];

psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));



psi_double=construct_double_layer(psi,psi);


global E_history
E_history=[10000];
save_filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
global save_filenm

global starting_time
starting_time=now();

global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="exact";#"simple_middle","canonical","exact"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=0;



multiplet_tol=1e-5;

global chi,multiplet_tol

global psi,psi_double,px,py
px=2;
py=2;
A0=psi[px,py];
E=cost_fun_local_test(A0);
println(E)

#f(A0)
# E_tem,∂E=get_grad(A0);
@time grad1=cost_fun_local_test'(A0);
@time _,grad2=FinteDiff_test(A0);

err1=norm(grad1-grad2)/min(norm(grad1),norm(grad2));
println(err1)
err2=dot(grad1,grad2)/sqrt(dot(grad1,grad1)*dot(grad2,grad2));
println(err2)


