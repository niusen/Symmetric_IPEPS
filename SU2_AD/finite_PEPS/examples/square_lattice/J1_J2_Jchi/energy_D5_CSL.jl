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
include("..\\..\\..\\environment\\AD\\mps_methods_new.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\AD\\density_matrix_new.jl")
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
parameters=Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)]);
global parameters

D=5;

#Hamiltonian
H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=8;
Ly=8;

data=load("CSL_D3.jld2");
A=data["A"];

if D==5
    A_new=zeros(D,D,D,D,2)*im;
    A_new[[1,2,3],[1,2,3],[1,2,3],[1,2,3],:]=convert(Array,A);
    A=TensorMap(A_new,Rep[SU₂](0=>1, 1/2=>2) ⊗ Rep[SU₂](0=>1, 1/2=>2) ⊗ Rep[SU₂](0=>1, 1/2=>2)' ⊗ Rep[SU₂](0=>1, 1/2=>2)', Rep[SU₂](1/2=>1));
elseif D==8
    A_new=zeros(D,D,D,D,2)*im;
    A_new[[1,2,3],[1,2,3],[1,2,3],[1,2,3],:]=convert(Array,A);
    A=TensorMap(A_new,Rep[SU₂](0=>1, 1/2=>2,1=>1) ⊗ Rep[SU₂](0=>1, 1/2=>2,1=>1) ⊗ Rep[SU₂](0=>1, 1/2=>2,1=>1)' ⊗ Rep[SU₂](0=>1, 1/2=>2,1=>1)', Rep[SU₂](1/2=>1));
end
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

filenm="CSL_D5_Lx"*string(Lx)*"_Ly"*string(Ly)*".jld2";
jldsave(filenm;psi);




multiplet_tol=1e-5;
chi=100;

global mpo_mps_trun_method, left_right_env_method;
mpo_mps_trun_method="simple_middle";#"simple_middle","canonical"
left_right_env_method="trun";#"exact","trun"


global n_mps_sweep
n_mps_sweep=3;

E,E_set=energy_disk_(psi);
px=1;py=1;
psi_double=construct_double_layer(psi,psi);
# a=energy_disk(psi[px,py],psi,psi_double,px,py)

E









