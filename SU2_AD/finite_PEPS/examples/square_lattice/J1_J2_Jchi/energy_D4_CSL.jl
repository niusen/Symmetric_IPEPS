using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\setting\\tuple_methods.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\ansatz\\square_lattice\\square_lattice.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")

Random.seed!(777)

global use_AD;
use_AD=false;

J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;

D=4;

#Hamiltonian
H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=4;
Ly=4;

data=load("CSL_D3.jld2");
A=data["A"];


A_new=zeros(4,4,4,4,2)*im;
A_new[[1,3,4],[1,3,4],[1,3,4],[1,3,4],:]=convert(Array,A);
A=TensorMap(A_new,Rep[SU₂](0=>2, 1/2=>1) ⊗ Rep[SU₂](0=>2, 1/2=>1) ⊗ Rep[SU₂](0=>2, 1/2=>1)' ⊗ Rep[SU₂](0=>2, 1/2=>1)', Rep[SU₂](1/2=>1));

A_noise=TensorMap(randn,codomain(A),domain(A))+TensorMap(randn,codomain(A),domain(A))*im;
A=A+A_noise/norm(A_noise)*0.1;
A=permute(A,(1,2,3,4,5,));






P=zeros(1,dim(space(A,1)));P[1,1]=1;
P_L=TensorMap(P,Rep[SU₂](0=>1),space(A,1));
P_D=TensorMap(P,Rep[SU₂](0=>1),space(A,2));

psi=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
for cx=2:Lx-1
    for cy=2:Ly-1
        psi[cx,cy]=A;
    end
end

cx=1;
for cy=2:Ly-1
    @tensor T[:]:=A[1,-2,-3,-4,-5]*P_L[-1,1];
    psi[cx,cy]=T;
end

cx=Lx;
for cy=2:Ly-1
    @tensor T[:]:=A[-1,-2,1,-4,-5]*P_L'[1,-3];
    psi[cx,cy]=T;
end

cy=1;
for cx=2:Lx-1
    @tensor T[:]:=A[-1,1,-3,-4,-5]*P_D[-2,1];
    psi[cx,cy]=T;
end

cy=Ly;
for cx=2:Lx-1
    @tensor T[:]:=A[-1,-2,-3,1,-5]*P_D'[1,-4];
    psi[cx,cy]=T;
end

cx=1;
cy=1;
@tensor T[:]:=A[1,2,-3,-4,-5]*P_L[-1,1]*P_D[-2,2];
psi[cx,cy]=T;

cx=Lx;
cy=1;
@tensor T[:]:=A[-1,2,1,-4,-5]*P_L'[1,-3]*P_D[-2,2];
psi[cx,cy]=T;

cx=1;
cy=Ly;
@tensor T[:]:=A[1,-2,-3,2,-5]*P_L[-1,1]*P_D'[2,-4];
psi[cx,cy]=T;

cx=Lx;
cy=Ly;
@tensor T[:]:=A[-1,-2,1,2,-5]*P_L'[1,-3]*P_D'[2,-4];
psi[cx,cy]=T;





psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

filenm="CSL_D3_L"*string(Lx)*".jld2";
jldsave(filenm;psi);

psi_double=construct_double_layer(psi,psi);






multiplet_tol=1e-5;
chi=100;


E_set=zeros(Lx-1,Ly-1)*im*1.0;

for cx=1:Lx-1
    for cy=1:Ly-1
        x_range=[cx,cx+1];
        y_range=[cy,cy+1];
        iPEPS_2x2=psi[x_range,y_range];
        @time rho_plaquatte,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
        E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        E_set[cx,cy]=E;
    end
end



E=sum(E_set);


# for cx=7:7
#     for cy=1:1
#         x_range=[cx,cx+1];
#         y_range=[cy,cy+1];
#         iPEPS_2x2=psi[x_range,y_range];
#         rho_plaquatte,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
#         rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
#         E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
#         E_set[cx,cy]=E;
#     end
# end


jldsave("CSL_D4.jld2"; psi);
jldsave("E_set_L"*string(Lx)*"_D_"*string(D)*".jld2"; E_set);

