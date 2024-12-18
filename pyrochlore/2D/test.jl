using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2
using Random

cd(@__DIR__)
include("pyrochlore_load_tensor.jl")
include("pyrochlore_IPESS.jl")
include("square_CTMRG_unitcell.jl")
include("spin_operator.jl")
include("pyrochlore_model_cell.jl")
include("build_tensor.jl")

include("funs_SU.jl")


Random.seed!(1234)

J1=1;
J2=1;
D_max=4;
symmetric_hosvd=false;
trun_tol=1e-6;
D=2;

println("D_max= "*string(D_max))

chi=40;

"Unit-cell format:
ABABAB
CDCDCD
ABABAB
CDCDCD


A11  A21
A12  A22


actual unit-cell:
ABAB
BABA
ABAB
BABA
"


virtual_type="square";
PEPS_tensor_A,A_fused_A,U_phy=build_PEPS(D,[],virtual_type);
PEPS_tensor_B,A_fused_B,_=build_PEPS(D,[],virtual_type);
######################

A_set,A1_set,A2_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D,virtual_type);
bond_tensor=A_set[1];
tetrahedral=A1_set[1];

@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[-4,2,-6]*tetrahedral[1,-2,-3,2];

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2]*U_phy[-5,1,2];


######################




T_u=deepcopy(tetrahedral);
T_d=deepcopy(tetrahedral);
B_a=deepcopy(bond_tensor);
B_b=deepcopy(bond_tensor);
B_c=deepcopy(bond_tensor);
B_d=deepcopy(bond_tensor);
lambda_u_a=unitary(space(bond_tensor,1),space(bond_tensor,1));
lambda_u_a=lambda_u_a'*lambda_u_a;
lambda_u_b=deepcopy(lambda_u_a);
lambda_u_c=deepcopy(lambda_u_a);
lambda_u_d=deepcopy(lambda_u_a);
lambda_d_a=deepcopy(lambda_u_a);
lambda_d_b=deepcopy(lambda_u_a);
lambda_d_c=deepcopy(lambda_u_a);
lambda_d_d=deepcopy(lambda_u_a);


U_d=space(bond_tensor,3);
U_phy_2=unitary(fuse(U_d ⊗ U_d), U_d ⊗ U_d);
H_plaquatte=plaquatte_Heisenberg(J1,J2);
#@tensor H_plaquatte[:]:=U_phy_2'[1,2,-1]*U_phy_2'[3,4,-2]*H_plaquatte[1,2,3,4,5,6,7,8]*U_phy_2[-3,5,6]*U_phy_2[-4,7,8];
#H_plaquatte=permute(H_plaquatte,(1,2,),(3,4,));

tau=0.2;
dt=0.1;
T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d=itebd(T_u,T_d,B_a,B_b,B_c,B_d,lambda_u_a,lambda_u_b,lambda_u_c,lambda_u_d,lambda_d_a,lambda_d_b,lambda_d_c,lambda_d_d, H_plaquatte,U_phy_2, trun_tol, tau, dt, D_max,symmetric_hosvd);


println(space(T_u))
println(space(T_d))



##############
@tensor PEPS_A[:]:=T_d[1,-2,-3,2]*B_a[1,-1,-5]*B_b[2,-4,-6]

@tensor PEPS_B[:]:=T_u[1,-2,-3,2]*B_c[-1,1,-5]*B_d[-4,2,-6]

U_phy=unitary(fuse(space(B_a, 3) ⊗ space(B_a, 3)), space(B_a, 3) ⊗ space(B_a, 3));
@tensor A_fused_A[:] :=PEPS_A[-1,-2,-3,-4,1,2]*U_phy[-5,1,2];
@tensor A_fused_B[:] :=PEPS_B[-1,-2,-3,-4,1,2]*U_phy[-5,1,2];
##############
Lx=2;Ly=2;
A_cell=Matrix(undef,Lx,Ly);
A_cell[1,1]=A_fused_A;
A_cell[2,1]=A_fused_B;
A_cell[1,2]=A_fused_B;
A_cell[2,2]=A_fused_A;
##############

A_unfused_cell=deepcopy(A_cell);
for cx=1:Lx
    for cy=1:Ly
        A_unfused=A_unfused_cell[cx,cy];
        @tensor A_unfused[:]:=A_unfused[-1,-2,-3,-4,1]*U_phy'[-5,-6,1];
        A_unfused_cell[cx,cy]=A_unfused;
    end
end


##############
CTM=[];
U_L=[];
U_D=[];
U_R=[];
U_U=[];

init=Dict([("CTM", []), ("init_type", "PBC")]);
conv_check="singular_value";
CTM_ite_info=true;
CTM_conv_info=true;
CTM_conv_tol=1e-6;
CTM_ite_nums=200;
CTM_trun_tol=1e-12;




println("chi= "*string(chi));flush(stdout);
@time CTM, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,_,_=CTMRG_cell(A_cell,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);

Sigma=plaquatte_Heisenberg(J1,J2);
AKLT=plaquatte_AKLT(Sigma);

####################
ca=1;
cb=1; #type 1 plaquatte
rho11=build_density_op_cell(U_phy, A_unfused_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell, CTM, ca,cb,Lx,Ly);#L',U',R',D',  L,U,R,D
Ea=plaquatte_ob(rho11,AKLT)
Eb=plaquatte_ob(rho11,Sigma)
println(Ea)
println(Eb)
####################
ca=1;
cb=2; #type 2 plaquatte
rho12=build_density_op_cell(U_phy, A_unfused_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell, CTM, ca,cb,Lx,Ly);#L',U',R',D',  L,U,R,D
Ea=plaquatte_ob(rho12,AKLT)
Eb=plaquatte_ob(rho12,Sigma)
println(Ea)
println(Eb)


