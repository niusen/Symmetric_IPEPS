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




J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;


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

T_pair=@ignore_derivatives unitary(space(A,3)',space(A,3));
@tensor A_C4[:]:=A[-1,-2,1,2,-5]*T_pair[-3,1]*T_pair[-4,2];

P=zeros(1,3);P[1,1]=1;
P_L=TensorMap(P,Rep[SU₂](0=>1),space(A_C4,1));
P_D=TensorMap(P,Rep[SU₂](0=>1),space(A_C4,2));

psi=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
for cx=2:Lx-1
    for cy=2:Ly-1
        psi[cx,cy]=A_C4;
    end
end

cx=1;
for cy=2:Ly-1
    @tensor T[:]:=A_C4[1,-2,-3,-4,-5]*P_L[-1,1];
    psi[cx,cy]=T;
end

cx=Lx;
for cy=2:Ly-1
    @tensor T[:]:=A_C4[-1,-2,1,-4,-5]*P_L[-3,1];
    psi[cx,cy]=T;
end

cy=1;
for cx=2:Lx-1
    @tensor T[:]:=A_C4[-1,1,-3,-4,-5]*P_D[-2,1];
    psi[cx,cy]=T;
end

cy=Ly;
for cx=2:Lx-1
    @tensor T[:]:=A_C4[-1,-2,-3,1,-5]*P_D[-4,1];
    psi[cx,cy]=T;
end

cx=1;
cy=1;
@tensor T[:]:=A_C4[1,2,-3,-4,-5]*P_L[-1,1]*P_D[-2,2];
psi[cx,cy]=T;

cx=Lx;
cy=1;
@tensor T[:]:=A_C4[-1,2,1,-4,-5]*P_L[-3,1]*P_D[-2,2];
psi[cx,cy]=T;

cx=1;
cy=Ly;
@tensor T[:]:=A_C4[1,-2,-3,2,-5]*P_L[-1,1]*P_D[-4,2];
psi[cx,cy]=T;

cx=Lx;
cy=Ly;
@tensor T[:]:=A_C4[-1,-2,1,2,-5]*P_L[-3,1]*P_D[-4,2];
psi[cx,cy]=T;


psi_corner=iPEPS_C4(psi, Lx,Ly);
psi=iPEPS_from_c4_corner(psi_corner,Lx,Ly);

for cx=1:Lx
    for cy=1:Ly
        A_=psi[cx,cy];
        U3=@ignore_derivatives unitary(space(A_,3)',space(A_,3));
        U4=@ignore_derivatives unitary(space(A_,4)',space(A_,4));
        @tensor A_[:]:=A_[-1,-2,1,2,-5]*U3[-3,1]*U4[-4,2];
        psi[cx,cy]=A_;

    end
end




psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));

psi_double=construct_double_layer(psi,psi);






multiplet_tol=1e-5;
chi=100;


E_set=zeros(Int(Lx/2),Int(Ly/2)-1)*im*1.0;

for cx=1:Int(Lx/2)
    for cy=1:Int(Ly/2)-1
        x_range=[cx,cx+1];
        y_range=[cy,cy+1];
        iPEPS_2x2=psi[x_range,y_range];
        rho_plaquatte,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
        E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        E_set[cx,cy]=E;
    end
end

E_total=sum(E_set)*4;



cx=Int(Lx/2);
cy=Int(Ly/2);
x_range=[cx,cx+1];
y_range=[cy,cy+1];
iPEPS_2x2=psi[x_range,y_range];
rho_plaquatte,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);



E_total=E_total+E;







