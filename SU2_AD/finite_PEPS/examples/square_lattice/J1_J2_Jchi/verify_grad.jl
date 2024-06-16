using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
using Dates
cd(@__DIR__)

include("..\\..\\..\\setting\\Settings.jl")
include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\state\\FinitePEPS.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
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


backward_settings=Backward_settings();
backward_settings.grad_inverse_tol=1e-8
backward_settings.grad_regulation_epsilon=1e-12;
backward_settings.show_ite_grad_norm=false;
dump(backward_settings);
global backward_settings




"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=6;
Ly=6;

filenm="CSL_D"*string(D)*"_L"*string(Lx)*".jld2";
data=load(filenm);


psi=data["psi"];

psi_double=construct_double_layer(psi,psi);


global E_history
E_history=[10000];
save_filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
global save_filenm

global starting_time
starting_time=now();


multiplet_tol=1e-5;

global chi,multiplet_tol

global psi,psi_double,px,py
px=2;
py=2;
A0=psi[px,py];
E=cost_fun(A0);
println(E)

#f(A0)
# E_tem,∂E=get_grad(A0);
@time grad1=cost_fun'(A0);
_,grad2=FinteDiff(A0);

err=norm(grad1-grad2)/norm(grad2);
println(err)

