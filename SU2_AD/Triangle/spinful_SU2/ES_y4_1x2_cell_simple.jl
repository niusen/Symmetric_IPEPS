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

y_anti_pbc=true;
filenm="Optim_cell_LS_D_4_chi_40_2.368055.jld2";
data=load(filenm);
A=data["x"][1].T;
B=data["x"][2].T;

#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D

A=A/norm(A);
B=B/norm(B);



#############################
# #convert to the order of PEPS code
A1=deepcopy(A);
A2=deepcopy(A);
A3=deepcopy(A);
A4=deepcopy(A);
if y_anti_pbc
    gauge_gate1=parity_gate(A1,4);
    @tensor A1[:]:=A1[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
end

B1=deepcopy(B);
B2=deepcopy(B);
B3=deepcopy(B);
B4=deepcopy(B);
if y_anti_pbc
    gauge_gate1=parity_gate(B1,4);
    @tensor B1[:]:=B1[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
end


#############################

AA1, U_L,U_D,U_R,U_U=build_double_layer_swap(A1',A1);
AA2, U_L,U_D,U_R,U_U=build_double_layer_swap(A2',A2);
AA3, U_L,U_D,U_R,U_U=build_double_layer_swap(A3',A3);
AA4, U_L,U_D,U_R,U_U=build_double_layer_swap(A4',A4);

BB1, U_L,U_D,U_R,U_U=build_double_layer_swap(B1',B1);
BB2, U_L,U_D,U_R,U_U=build_double_layer_swap(B2',B2);
BB3, U_L,U_D,U_R,U_U=build_double_layer_swap(B3',B3);
BB4, U_L,U_D,U_R,U_U=build_double_layer_swap(B4',B4);
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


v_init=TensorMap(randn, space(BB2,3)*space(BB2,3)*space(BB2,3)*space(BB2,3),Rep[SU₂]((0)=>1)');
v_init=permute(v_init,(5,1,2,3,4,),());#dummy,R1,R2,R3,R4
contraction_fun_L(x)=vl_M( vl_M(x, AA1,AA2,AA3,AA4), BB1,BB2,BB3,BB4);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 1,:LM,Arnoldi(krylovdim=40));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3,R4






@tensor VL[:]:=VL[-1,1,2,3,4]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7]*U_L[4,-8,-9];#dummy, R1',R1,R2',R2,R3',R3,R4',R4
VL=permute(VL,(1,3,2,5,4,7,6,9,8,));#dummy, R1,R1',R2,R2',R3,R3',R4,R4'
VL=permute_neighbour_ind(VL,3,4,9);#dummy, R1,R2,R1',R2',R3,R3',R4,R4'
VL=permute_neighbour_ind(VL,5,6,9);#dummy, R1,R2,R1',R3,R2',R3',R4,R4'
VL=permute_neighbour_ind(VL,4,5,9);#dummy, R1,R2,R3,R1',R2',R3',R4,R4'
VL=permute_neighbour_ind(VL,7,8,9);#dummy, R1,R2,R3,R1',R2',R4,R3',R4'
VL=permute_neighbour_ind(VL,6,7,9);#dummy, R1,R2,R3,R1',R4,R2',R3',R4'
VL=permute_neighbour_ind(VL,5,6,9);#dummy, R1,R2,R3,R4,R1',R2',R3',R4'


@tensor VR[:]:=VR[1,2,3,4,-9]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4];#L1',L1,L2',L2,L3',L3,L4',L4,dummy
VR=permute(VR,(7,8,5,6,3,4,1,2,9));#L4',L4,L3',L3,L2',L2,L1',L1,dummy
VR=permute_neighbour_ind(VR,6,7,9);#L4',L4,L3',L3,L2',L1',L2,L1,dummy
VR=permute_neighbour_ind(VR,4,5,9);#L4',L4,L3',L2',L3,L1',L2,L1,dummy
VR=permute_neighbour_ind(VR,5,6,9);#L4',L4,L3',L2',L1',L3,L2,L1,dummy
VR=permute_neighbour_ind(VR,2,3,9);#L4',L3',L4,L2',L1',L3,L2,L1,dummy
VR=permute_neighbour_ind(VR,3,4,9);#L4',L3',L2',L4,L1',L3,L2,L1,dummy
VR=permute_neighbour_ind(VR,4,5,9);#L4',L3',L2',L1',L4,L3,L2,L1,dummy



@tensor H[:]:=VR[-1,-2,-3,-4,5,4,3,2,1]*VL[1,2,3,4,5,-5,-6,-7,-8];#L4',L3',L2',L1',   R1',R2',R3',R4'
H=permute(H,(1,2,3,4,),(8,7,6,5,));#L4',L3',L2',L1',   R4',R3',R2',R1'


eu,ev=eig(H,(1,2,3,4,),(5,6,7,8,));
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))


if y_anti_pbc
    ev=permute(ev,(1,2,3,4,5,));#L1',L2',L3',L4',dummy
    op=parity_gate(ev,4);
    @tensor ev_translation[:]:=op[1,-1]*ev'[1,-2,-3,-4,-5];
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),1,2,5);#L2',L1',L3',L4',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,5);#L2',L3',L1',L4',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,5);#L2',L3',L4',L1',dummy
else
    ev=permute(ev,(1,2,3,4,5,));#L1',L2',L3',L4',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,5);#L2',L1',L3',L4',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,5);#L2',L3',L1',L4',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,5);#L2',L3',L4',L1',dummy
end


@tensor k_phase[:]:=ev_translation[1,2,3,4,-1]*ev[1,2,3,4,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;


order=sortperm(abs.(eu));
if length(order)>500
    order=order[end-500:end];
end
k_phase=diag(k_phase);
eu=eu[order];
k_phase=k_phase[order];
Spin=Spin[order]


##########################

if y_anti_pbc
    matnm="ES_unprojected_M1_Nv4_APBC"*".mat";
else
    matnm="ES_unprojected_M1_Nv4_PBC"*".mat";
end
matwrite(matnm, Dict(
    "k_phase" => k_phase,
    "eu" => eu,
    "Spin"=>Spin
); compress = false)



