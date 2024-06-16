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
include("..\\..\\..\\environment\\AD\\svd_AD_lib.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")
include("..\\..\\..\\optimization\\line_search_lib.jl")
include("..\\..\\..\\optimization\\LineSearches\\My_Backtracking.jl")
include("..\\..\\..\\optimization\\PEPS_methods.jl")


global use_AD;
use_AD=false;

J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
global parameters


data=load("optim_4x4_D_3_chi_100_13.89035.jld2");
psi=data["psi"];

multiplet_tol=1e-5;
chi=100;


#Hamiltonian
H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

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



Lx,Ly=size(psi);

# psi=disk_to_torus(psi);
# psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

psi_double=construct_double_layer(psi,psi);




global chi,multiplet_tol

global psi,psi_double,px,py

global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"

global n_mps_sweep
n_mps_sweep=5;


px=1;py=1;
E_opt=real(cost_fun_local(psi[px,py]));




E_set=zeros(Lx-1,Ly-1)*im*1.0;
E,E_set=energy_disk_(psi);
println(E)


for ba=1.5:1:3.5
    for bb=1:4
        virtual_spin=1/2;
        bond_pos=[ba,bb];
        Noise=0;
        psi_new=extend_single_bond(virtual_spin, psi, bond_pos, Noise);

        E,E_set=energy_disk_(psi_new);
        println(E)
    end
end

for ba=1:4
    for bb=1.5:1:3.5
        virtual_spin=1/2;
        bond_pos=[ba,bb];
        Noise=0;
        psi_new=extend_single_bond(virtual_spin, psi, bond_pos, Noise);

        E,E_set=energy_disk_(psi_new);
        println(E)
    end
end


include("..\\..\\..\\optimization\\PEPS_methods.jl")
virtual_spin=1/2;
Noise=0;
psi_new=add_virtual_particle_global(virtual_spin, psi, Noise);
E,E_set=energy_disk_(psi_new);
println(E)


include("..\\..\\..\\optimization\\PEPS_methods.jl")
virtual_spin=1/2;
Noise=0;
psi_new=add_virtual_particle_boundary(virtual_spin, psi, Noise);
E,E_set=energy_disk_(psi_new);
println(E)




# jldsave("E_set_Lx"*string(Lx)*"_Ly"*string(Ly)*"_D_"*string(D)*".jld2"; E_set);
