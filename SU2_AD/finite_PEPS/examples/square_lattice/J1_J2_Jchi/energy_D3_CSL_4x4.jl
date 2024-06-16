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

P=zeros(1,3);P[1,1]=1;
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

psi_double=construct_double_layer(psi,psi);






multiplet_tol=1e-5;
chi=60;


function normalize_rho(rho,U_s_s)
    @tensor rho[:]:=rho[1,2,3,4]*U_s_s[-1,-5,1]*U_s_s[-2,-6,2]*U_s_s[-3,-7,3]*U_s_s[-4,-8,4];
    rho=permute(rho,(1,2,3,4,),(5,6,7,8,));
    Norm=@tensor rho[1,2,3,4,1,2,3,4];
    rho=rho/Norm;
    return rho
end


x_range=[2,3];
y_range=[2,3];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_bulk=normalize_rho(rho,U_s_s);

x_range=[1,2];
y_range=[2,3];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_left=normalize_rho(rho,U_s_s);

x_range=[Lx-1,Lx];
y_range=[2,3];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_right=normalize_rho(rho,U_s_s);

x_range=[2,3];
y_range=[Ly-1,Ly];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_top=normalize_rho(rho,U_s_s);

x_range=[2,3];
y_range=[1,2];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_bot=normalize_rho(rho,U_s_s);

x_range=[1,2];
y_range=[Ly-1,Ly];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_left_top=normalize_rho(rho,U_s_s);

x_range=[Lx-1,Lx];
y_range=[Ly-1,Ly];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_right_top=normalize_rho(rho,U_s_s);

x_range=[1,2];
y_range=[1,2];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_left_bot=normalize_rho(rho,U_s_s);

x_range=[Lx-1,Lx];
y_range=[1,2];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
rho_right_bot=normalize_rho(rho,U_s_s);



println(norm(rho_bulk-permute(rho_bulk,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_top-permute(rho_right,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_right-permute(rho_bot,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_bot-permute(rho_left,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_left-permute(rho_top,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_left_top-permute(rho_right_top,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_right_top-permute(rho_right_bot,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_right_bot-permute(rho_left_bot,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))

println(norm(rho_left_bot-permute(rho_left_top,(2,3,4,1,),(6,7,8,5,)))/norm(rho_bulk))


E_x=zeros(3,4)*im*1.0;

E_x[1,1]=@tensor rho_left_bot[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[1,2]=@tensor rho_left[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[1,3]=@tensor rho_left_top[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[1,4]=@tensor rho_left_top[1,2,5,6,3,4,5,6]*H_Heisenberg[1,2,3,4];

E_x[2,1]=@tensor rho_bot[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[2,2]=@tensor rho_bulk[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[2,3]=@tensor rho_top[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[2,4]=@tensor rho_top[1,2,5,6,3,4,5,6]*H_Heisenberg[1,2,3,4];

E_x[3,1]=@tensor rho_right_bot[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[3,2]=@tensor rho_right[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[3,3]=@tensor rho_right_top[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
E_x[3,4]=@tensor rho_right_top[1,2,5,6,3,4,5,6]*H_Heisenberg[1,2,3,4];


E_y=zeros(4,3)*im*1.0;

E_y[1,1]=@tensor rho_left_bot[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[2,1]=@tensor rho_bot[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[3,1]=@tensor rho_right_bot[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[4,1]=@tensor rho_right_bot[5,1,2,6,5,3,4,6]*H_Heisenberg[1,2,3,4];

E_y[1,2]=@tensor rho_left[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[2,2]=@tensor rho_bulk[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[3,2]=@tensor rho_right[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[4,2]=@tensor rho_right[5,1,2,6,5,3,4,6]*H_Heisenberg[1,2,3,4];

E_y[1,3]=@tensor rho_left_top[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[2,3]=@tensor rho_top[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[3,3]=@tensor rho_right_top[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
E_y[4,3]=@tensor rho_right_top[5,1,2,6,5,3,4,6]*H_Heisenberg[1,2,3,4];


E_right_top=zeros(3,3)*im*1.0;

E_right_top[1,1]=@tensor rho_left_bot[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[2,1]=@tensor rho_bot[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[3,1]=@tensor rho_right_bot[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];

E_right_top[1,2]=@tensor rho_left[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[2,2]=@tensor rho_bulk[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[3,2]=@tensor rho_right[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];

E_right_top[1,3]=@tensor rho_left_top[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[2,3]=@tensor rho_top[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
E_right_top[3,3]=@tensor rho_right_top[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];


E_right_bot=zeros(3,3)*im*1.0;

E_right_bot[1,1]=@tensor rho_left_bot[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[2,1]=@tensor rho_bot[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[3,1]=@tensor rho_right_bot[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];

E_right_bot[1,2]=@tensor rho_left[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[2,2]=@tensor rho_bulk[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[3,2]=@tensor rho_right[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];

E_right_bot[1,3]=@tensor rho_left_top[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[2,3]=@tensor rho_top[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
E_right_bot[3,3]=@tensor rho_right_top[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];


E_chiral_123=zeros(3,3)*im*1.0;

E_chiral_123[1,1]=@tensor rho_left_bot[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[2,1]=@tensor rho_bot[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[3,1]=@tensor rho_right_bot[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];

E_chiral_123[1,2]=@tensor rho_left[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[2,2]=@tensor rho_bulk[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[3,2]=@tensor rho_right[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];

E_chiral_123[1,3]=@tensor rho_left_top[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[2,3]=@tensor rho_top[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
E_chiral_123[3,3]=@tensor rho_right_top[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];


E_chiral_234=zeros(3,3)*im*1.0;

E_chiral_234[1,1]=@tensor rho_left_bot[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[2,1]=@tensor rho_bot[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[3,1]=@tensor rho_right_bot[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];

E_chiral_234[1,2]=@tensor rho_left[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[2,2]=@tensor rho_bulk[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[3,2]=@tensor rho_right[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];

E_chiral_234[1,3]=@tensor rho_left_top[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[2,3]=@tensor rho_top[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
E_chiral_234[3,3]=@tensor rho_right_top[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];


E_chiral_341=zeros(3,3)*im*1.0;

E_chiral_341[1,1]=@tensor rho_left_bot[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[2,1]=@tensor rho_bot[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[3,1]=@tensor rho_right_bot[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];

E_chiral_341[1,2]=@tensor rho_left[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[2,2]=@tensor rho_bulk[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[3,2]=@tensor rho_right[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];

E_chiral_341[1,3]=@tensor rho_left_top[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[2,3]=@tensor rho_top[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
E_chiral_341[3,3]=@tensor rho_right_top[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];


E_chiral_412=zeros(3,3)*im*1.0;

E_chiral_412[1,1]=@tensor rho_left_bot[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[2,1]=@tensor rho_bot[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[3,1]=@tensor rho_right_bot[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];

E_chiral_412[1,2]=@tensor rho_left[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[2,2]=@tensor rho_bulk[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[3,2]=@tensor rho_right[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];

E_chiral_412[1,3]=@tensor rho_left_top[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[2,3]=@tensor rho_top[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];
E_chiral_412[3,3]=@tensor rho_right_top[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];


E=J1*sum(E_x)+J1*sum(E_y)+J2*sum(E_right_bot)+J2*sum(E_right_top)+Jchi*sum(E_chiral_123+E_chiral_234+E_chiral_341+E_chiral_412)



