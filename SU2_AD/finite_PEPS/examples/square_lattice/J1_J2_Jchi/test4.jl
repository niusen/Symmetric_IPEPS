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


chi=50;
global multiplet_tol;
multiplet_tol=1e-5;


psi_double=construct_double_layer(psi,psi);


x_range=[2,3];
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



global use_AD;
use_AD=false;

mps_up_new=mps_set_down_move_new[:,3];
mps_down_new=mps_set_up_move_new[:,2];

mps_up=mps_set_down_move[:,3];
mps_down=mps_set_up_move[:,2];






e=contract_1D_with_plaquatte(mps_up_new,mps_down_new,[],[],2,3,h_plaquatte);
Norm=contract_1D_with_plaquatte(mps_up,mps_down,[],[],[],[],h_plaquatte);


U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy);
@tensor e1[:]:=e[1,2]*U_ss'[-1,-3,1]*U_ss'[-2,-4,2];
# Norm= @tensor e[1,2,1,2]
E=@tensor e1[1,2,3,4]*H_Heisenberg[1,2,3,4]
E/Norm


mps_down_new2=deepcopy(mps_down_new);
T=mps_down_new2[1];
@tensor T[:]:=T[1,-2,-3]*projectors_R_set_up_move[2][1][1,-1];
mps_down_new2[1]=T;

T=mps_down_new2[2];
@tensor T[:]:=T[1,-2,-3,-4]*projectors_L_set_up_move[2][2][-1,1];
mps_down_new2[2]=T;

e=contract_1D_with_plaquatte(mps_up_new,mps_down_new2,[],[],2,3,h_plaquatte);
Norm=contract_1D_with_plaquatte(mps_up,mps_down,[],[],[],[],h_plaquatte);

U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy);
@tensor e2[:]:=e[1,2]*U_ss'[-1,-3,1]*U_ss'[-2,-4,2];
# Norm= @tensor e[1,2,1,2]
E=@tensor e2[1,2,3,4]*H_Heisenberg[1,2,3,4]
E/Norm



pro=leftnull(projectors_R_set_up_move[2][1]);
mps_down_new3=deepcopy(mps_down_new);
T=mps_down_new3[1];
@tensor T[:]:=T[1,-2,-3]*pro[1,-1];
mps_down_new3[1]=T;

T=mps_down_new3[2];
@tensor T[:]:=T[1,-2,-3,-4]*pro'[-1,1];
mps_down_new3[2]=T;

e=contract_1D_with_plaquatte(mps_up_new,mps_down_new3,[],[],2,3,h_plaquatte);
Norm=contract_1D_with_plaquatte(mps_up,mps_down,[],[],[],[],h_plaquatte);

U_ss=unitary(fuse(U_phy*U_phy),U_phy'*U_phy);
@tensor e3[:]:=e[1,2]*U_ss'[-1,-3,1]*U_ss'[-2,-4,2];
# Norm= @tensor e[1,2,1,2]
E=@tensor e3[1,2,3,4]*H_Heisenberg[1,2,3,4]
E/Norm






# E_set=zeros(Lx-1,Ly-1)*im*1.0;

cx=1;
cy=1;
        x_range=[cx,cx+1];
        y_range=[cy,cy+1];
        iPEPS_2x2=psi[x_range,y_range];

        E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,e);
        E_set[cx,cy]=E;





  
for cx=1:1
    for cy=1:1
        x_range=[cx,cx+1];
        y_range=[cy,cy+1];
        iPEPS_2x2=psi[x_range,y_range];
        rho_plaquatte,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
        E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);println(E)
        u,s,v=tsvd(rho_plaquatte,(1,2,3,4,),(5,6,7,8,));println(s)
    end
end


function ov_special(mps1,mps2)
    @tensor env[:]:=mps1[1]'[-1,1,2]*mps2[1][-2,1,2];
    @tensor env[:]:=env[1,3]*mps1[2]'[1,-1,2,4]*mps2[2][3,-2,2,4];
    @tensor env[:]:=env[1,3]*mps1[3]'[1,-1,2]*mps2[3][3,-2,2];
     env=@tensor env[1,3]*mps1[4]'[1,2]*mps2[4][3,2];
     return env
end

ov_special(mps_down_compressed,mps_down_uncompressed)
ov_special(mps_down_uncompressed,mps_down_uncompressed)
ov_special(mps_down_compressed,mps_down_compressed)