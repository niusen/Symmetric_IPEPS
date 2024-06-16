using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\mps_methods.jl")
include("..\\..\\..\\environment\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\truncations.jl")
include("..\\..\\..\\environment\\density_matrix.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")




J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;


#Hamiltonian
H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""




filenm="Optim_LS_D_3_chi_80.jld2";
data=load(filenm);

Lx,Ly=size(psi);


psi=data["psi"];

psi_double=construct_double_layer(psi,psi);






multiplet_tol=1e-5;
chi=100;


E_set=zeros(Lx-1,Ly-1)*im*1.0;

for cx=1:Lx-1
    for cy=1:Ly-1
        x_range=[cx,cx+1];
        y_range=[cy,cy+1];
        iPEPS_2x2=psi[x_range,y_range];
        rho_plaquatte,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
        E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        E_set[cx,cy]=E;
    end
end



E=sum(E_set);



