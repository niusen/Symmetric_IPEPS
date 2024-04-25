using LinearAlgebra:diag,I,diagm 
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)



include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\mps_algorithms\\Projector_funs.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
let
y_anti_pbc=true;
filenm="stochastic_iPESS_LS_D_4_chi_40_2.2439.jld2";
data=load(filenm);
# A=data["x"][1].T;
# B=data["x"][2].T;
state=data["x"]


global Lx,Ly
Lx,Ly=size(state);
A_cell=initial_tuple_cell(Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        if isa(state[ca,cb],Square_iPEPS)
            A_cell=fill_tuple(A_cell, state[ca,cb].T, ca,cb);
        elseif isa(state[ca,cb],Triangle_iPESS)
            A0=iPESS_to_iPEPS(state[ca,cb]).T;
            A0=A0/norm(A0)*10;
            A_cell=fill_tuple(A_cell, A0, ca,cb);
        else
            error("unknown type ansatz")
        end
    end
end


#############################
A1=deepcopy(A_cell[1][1]);
A2=deepcopy(A_cell[1][2]);
A3=deepcopy(A_cell[1][1]);
A4=deepcopy(A_cell[1][2]);

A1p=A1';
A2p=A2';
A3p=A3';
A4p=A4';

B1=deepcopy(A_cell[2][1]);
B2=deepcopy(A_cell[2][2]);
B3=deepcopy(A_cell[2][1]);
B4=deepcopy(A_cell[2][2]);

B1p=B1';
B2p=B2';
B3p=B3';
B4p=B4';


use_translate=true;

if y_anti_pbc
    gauge_gate1=parity_gate(A1,4);
    @tensor A1[:]:=A1[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
    if use_translate
        @tensor A2t[:]:=A2[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
        A2p=A2t';
    else
        A1p=A1';
    end
end
if y_anti_pbc
    gauge_gate1=parity_gate(B1,4);
    @tensor B1[:]:=B1[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
    if use_translate
        @tensor B2t[:]:=B2[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
        B2p=B2t';
    else
        B1p=B1';
    end
end


#############################
if use_translate
    AA1, U_L,U_D,U_R,U_U=build_double_layer_swap(A2p,A1);
    AA2, U_L,U_D,U_R,U_U=build_double_layer_swap(A3p,A2);
    AA3, U_L,U_D,U_R,U_U=build_double_layer_swap(A4p,A3);
    AA4, U_L,U_D,U_R,U_U=build_double_layer_swap(A1p,A4);

    BB1, U_L,U_D,U_R,U_U=build_double_layer_swap(B2p,B1);
    BB2, U_L,U_D,U_R,U_U=build_double_layer_swap(B3p,B2);
    BB3, U_L,U_D,U_R,U_U=build_double_layer_swap(B4p,B3);
    BB4, U_L,U_D,U_R,U_U=build_double_layer_swap(B1p,B4);
else
    AA1, U_L,U_D,U_R,U_U=build_double_layer_swap(A1p,A1);
    AA2, U_L,U_D,U_R,U_U=build_double_layer_swap(A2p,A2);
    AA3, U_L,U_D,U_R,U_U=build_double_layer_swap(A3p,A3);
    AA4, U_L,U_D,U_R,U_U=build_double_layer_swap(A4p,A4);

    BB1, U_L,U_D,U_R,U_U=build_double_layer_swap(B1p,B1);
    BB2, U_L,U_D,U_R,U_U=build_double_layer_swap(B2p,B2);
    BB3, U_L,U_D,U_R,U_U=build_double_layer_swap(B3p,B3);
    BB4, U_L,U_D,U_R,U_U=build_double_layer_swap(B4p,B4);
end

#############################
#extra swap gate that was not included when construct double layer tensor
gate=swap_gate(U_L,2,3);
@tensor U_L_new[:]:=U_L'[1,2,-1]*gate[1,2,3,4]*U_L[-2,3,4];
gate=swap_gate(U_R,1,2);
@tensor U_R_new[:]:=U_R'[-1,1,2]*gate[1,2,3,4]*U_R[3,4,-2];

@tensor AA1[:]:=AA1[1,-2,2,-4]*U_L_new'[-1,1]*U_R_new'[-3,2]; 
@tensor AA2[:]:=AA2[1,-2,2,-4]*U_L_new'[-1,1]*U_R_new'[-3,2]; 
@tensor AA3[:]:=AA3[1,-2,2,-4]*U_L_new'[-1,1]*U_R_new'[-3,2]; 
@tensor AA4[:]:=AA4[1,-2,2,-4]*U_L_new'[-1,1]*U_R_new'[-3,2]; 

@tensor BB1[:]:=BB1[1,-2,2,-4]*U_L_new'[-1,1]*U_R_new'[-3,2]; 
@tensor BB2[:]:=BB2[1,-2,2,-4]*U_L_new'[-1,1]*U_R_new'[-3,2]; 
@tensor BB3[:]:=BB3[1,-2,2,-4]*U_L_new'[-1,1]*U_R_new'[-3,2]; 
@tensor BB4[:]:=BB4[1,-2,2,-4]*U_L_new'[-1,1]*U_R_new'[-3,2]; 
#############################
#extra parity gate from crossing
gate=parity_gate(AA1,4);
@tensor AA1[:]:=AA1[-1,-2,-3,1]*gate[-4,1];
gate=parity_gate(BB1,4);
@tensor BB1[:]:=BB1[-1,-2,-3,1]*gate[-4,1];

#############################


function M_vr(vr0,  AA1,AA2,AA3,AA4)
    @tensor vr[:]:=AA1[-1,2,1,8]*AA2[-2,4,3,2]*AA3[-3,6,5,4]*AA4[-4,8,7,6]*vr0[1,3,5,7,-5];
    return vr;
end

function vl_M(vl0,  AA1,AA2,AA3,AA4)
    @tensor vl[:]:=AA1[1,2,-2,8]*AA2[3,4,-3,2]*AA3[5,6,-4,4]*AA4[7,8,-5,6]*vl0[-1,1,3,5,7];
    return vl
end




v_init=TensorMap(randn, space(AA2,1)*space(AA2,1)*space(AA2,1)*space(AA2,1),Rep[SU₂]((0)=>1));
v_init=permute(v_init,(1,2,3,4,5,),());#L1,L2,L3,L4,dummy
contraction_fun_R(x)=M_vr(M_vr(x, BB1,BB2,BB3,BB4), AA1,AA2,AA3,AA4);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 1,:LM,Arnoldi(krylovdim=40));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,L4,dummy

println(eur)
end