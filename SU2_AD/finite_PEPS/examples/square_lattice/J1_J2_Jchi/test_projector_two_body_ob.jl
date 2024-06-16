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
include("..\\..\\..\\environment\\AD\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\AD\\mps_methods.jl")
include("..\\..\\..\\environment\\AD\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\AD\\truncations.jl")
include("..\\..\\..\\environment\\AD\\density_matrix.jl")
include("..\\..\\..\\environment\\Variational\\mps_methods_projector.jl")
include("..\\..\\..\\environment\\Variational\\check_ob.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\spin_operator.jl")
include("..\\..\\..\\models\\spin\\square_lattice\\J1_J2_Jchi_disk.jl")



global use_AD;
use_AD=false;

J1=2*cos(0.06*pi)*cos(0.14*pi);
J2=2*cos(0.06*pi)*sin(0.14*pi);
Jchi=2*sin(0.06*pi)*2;


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

global U_phy
U_phy=space(A,5);


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

filenm="CSL_D3_L"*string(Lx)*".jld2";
jldsave(filenm;psi);


chi=80;
global multiplet_tol;
multiplet_tol=1e-5;


psi_double=construct_double_layer(psi,psi);

x_range=[1,2];
y_range=[1,2];
x_range=[7,8];
y_range=[1];
psi_double_plaquatte=construct_double_layer_open_plaquatte(psi,x_range,y_range);



mps_set_down_move, trun_errs, norm_coe_down_move, UR_set_down_move, UL_set_down_move, unitarys_R_set_down_move, unitarys_L_set_down_move, projectors_R_set_down_move, projectors_L_set_down_move=get_projector_down_move(psi_double);
mps_set_up_move, trun_errs, norm_coe_up_move, UR_set_up_move, UL_set_up_move, unitarys_R_set_up_move, unitarys_L_set_up_move, projectors_R_set_up_move, projectors_L_set_up_move=get_projector_up_move(psi_double);


@time mps_set_up_move_new=reconstruct_boundary_mps_up_move(psi_double_plaquatte, norm_coe_up_move, unitarys_R_set_up_move,unitarys_L_set_up_move, UR_set_up_move,UL_set_up_move,projectors_R_set_up_move,projectors_L_set_up_move);

@time mps_set_down_move_new=reconstruct_boundary_mps_down_move(psi_double_plaquatte, norm_coe_down_move, unitarys_R_set_down_move,unitarys_L_set_down_move, UR_set_down_move,UL_set_down_move,projectors_R_set_down_move,projectors_L_set_down_move);



# for cc=1:Ly-1
#     ov=norm_1D(mps_set_up_move_new[:,cc],mps_set_up_move[:,cc])/sqrt(norm_1D(mps_set_up_move_new[:,cc],mps_set_up_move_new[:,cc])*norm_1D(mps_set_up_move[:,cc],mps_set_up_move[:,cc]));
#     println(ov)
# end
# for cc=Ly:-1:2
#     ov=norm_1D(mps_set_down_move_new[:,cc],mps_set_down_move[:,cc])/sqrt(norm_1D(mps_set_down_move_new[:,cc],mps_set_down_move_new[:,cc])*norm_1D(mps_set_down_move[:,cc],mps_set_down_move[:,cc]));
#     println(ov)
# end




mps_up_new=mps_set_down_move_new[:,Ly];
mps_down_new=mps_set_up_move_new[:,Ly-1];

mps_up=mps_set_down_move[:,Ly];
mps_down=mps_set_up_move[:,Ly-1];






e=contract_1D_with_plaquatte(mps_up_new,mps_down_new,[],[],x_range[1],x_range[2],h_plaquatte);
Norm=contract_1D_with_plaquatte(mps_up,mps_down,[],[],[],[],h_plaquatte);




# e/Norm


U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy);
@tensor e[:]:=e[1,2]*U_ss'[-1,-3,1]*U_ss'[-2,-4,2];
# Norm= @tensor e[1,2,1,2]
E=@tensor e[1,2,3,4]*H_Heisenberg[1,2,3,4]
E/Norm







