using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
cd(@__DIR__)
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("pyrochlore_load_tensor.jl")
include("pyrochlore_IPESS.jl")
include("square_CTMRG_unitcell.jl")
include("spin_operator.jl")
include("pyrochlore_model_cell.jl")
include("build_tensor.jl")


Random.seed!(1234)

J1=1;
J2=1;
D=2;
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

virtual_type="tetrahedral";#"tetrahedral",  "square"

if virtual_type=="tetrahedral"
    coe1=[1,0];
    coe2=[0,1];
    PEPS_tensor_A,A_fused_A,U_phy=build_PEPS(D,coe1,virtual_type);
    PEPS_tensor_B,A_fused_B,_=build_PEPS(D,coe2,virtual_type);
elseif virtual_type=="square"

    PEPS_tensor_A,A_fused_A,U_phy=build_PEPS(D,[],virtual_type);
    PEPS_tensor_B,A_fused_B,_=build_PEPS(D,[],virtual_type);
end




Lx=2;Ly=2;
A_cell=Matrix(undef,Lx,Ly);
A_cell[1,1]=A_fused_A;
A_cell[2,1]=A_fused_B;
A_cell[1,2]=A_fused_B;
A_cell[2,2]=A_fused_A;




#change virtual space to check the ctmrg code
A_fused_cell=change_virtual(A_cell);

A_unfused_cell=deepcopy(A_fused_cell);
for cx=1:Lx
    for cy=1:Ly
        A_unfused=A_unfused_cell[cx,cy];
        @tensor A_unfused[:]:=A_unfused[-1,-2,-3,-4,1]*U_phy'[-5,-6,1];
        A_unfused_cell[cx,cy]=A_unfused;
    end
end

init=Dict([("CTM", []), ("init_type", "PBC")]);
conv_check="singular_value";
CTM_ite_info=true;
CTM_conv_info=true;
CTM_conv_tol=1e-6;
CTM_ite_nums=100;
CTM_trun_tol=1e-12;

@time CTM, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,_,_=CTMRG_cell(A_fused_cell,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info,CTM_conv_info);


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
