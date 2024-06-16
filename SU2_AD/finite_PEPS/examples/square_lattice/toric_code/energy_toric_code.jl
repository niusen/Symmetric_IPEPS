using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\symmetry\\parity_funs.jl")
include("convert_boundary_condition.jl")
include("mps_methods.jl")
include("peps_double_layer_methods.jl")
include("truncations.jl")
include("density_matrix.jl")
include("..\\models\\spin\\toric_code.jl")


#Hamiltonian
Ax,Az=toric_code_terms();



"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=8;
Ly=8;


data=load("Toric_code.jld2");
# T1=data["T1"];
# T2=data["T2"];

# psi=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
# for cx=1:Lx
#     for cy=1:Ly
#         if mod(cx+cy,2)==0
#             psi[cx,cy]=T1;
#         else
#             psi[cx,cy]=T2;
#         end
#     end
# end

A=data["A"];
U_phy=unitary(fuse(space(A,5)*space(A,6)), space(A,5)*space(A,6));
@tensor A[:]:=A[-1,-2,-3,-4,1,2]*U_phy[-5,1,2];


psi=Matrix{TensorMap}(undef,Lx,Ly);
for cx=1:Lx
    for cy=1:Ly
        psi[cx,cy]=A;
    end
end




psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));
psi_double=construct_double_layer(psi,psi);






multiplet_tol=1e-5;
chi=10;




x_range=[2,3];
y_range=[2,3];
iPEPS_2x2=psi[x_range,y_range];
rho,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);


@tensor U_p1p2[:]:=U_s_s[1,2,-5]*U_phy[1,-1,-2]*U_phy'[-3,-4,2];
@tensor U_p1[:]:=U_s_s[1,2,-3]*U_phy[1,-1,3]*U_phy'[-2,3,2];
@tensor U_p2[:]:=U_s_s[1,2,-3]*U_phy[1,3,-1]*U_phy'[3,-2,2];
@tensor U_0[:]:=U_s_s[1,2,-3]*U_phy[1,3,4]*U_phy'[3,4,2];


@tensor rho1[:]:=rho[1,2,3,4]*U_p1p2[-1,-2,-5,-6,1]*U_p1[-3,-7,2]*U_0[3]*U_p2[-4,-8,4];
rho1=permute(rho1,(1,2,3,4,),(5,6,7,8,));
Norm=@tensor rho1[1,2,3,4,1,2,3,4];
E1=@tensor Ax[5,6,7,8,1,2,3,4]*rho1[1,2,3,4,5,6,7,8];
E1=E1/Norm;

@tensor rho2[:]:=rho[1,2,3,4]*U_0[1]*U_p1[-1,-5,2]*U_p1p2[-3,-2,-7,-6,3]*U_p2[-4,-8,4];
rho2=permute(rho2,(1,2,3,4,),(5,6,7,8,));
Norm=@tensor rho2[1,2,3,4,1,2,3,4];
E2=@tensor Az[5,6,7,8,1,2,3,4]*rho2[1,2,3,4,5,6,7,8];
E2=E2/Norm;

println([E1,E2])

