using LinearAlgebra:diag,I,diagm 
using JLD2,ChainRulesCore,MAT, Zygote
using Zygote:@ignore_derivatives
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)



include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\mps_algorithms\\Projector_funs.jl")
include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")

y_anti_pbc=true;
filenm="Optim_cell_LS_D_4_chi_40_2.368055.jld2";
data=load(filenm);
# A=data["x"][1].T;
# B=data["x"][2].T;
state=data["x"]
#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
dump(algrithm_CTMRG_settings);
global algrithm_CTMRG_settings

LS_ctm_setting=LS_CTMRG_settings();
LS_ctm_setting.CTM_conv_tol=1e-6;
LS_ctm_setting.CTM_ite_nums=10;
LS_ctm_setting.CTM_trun_tol=1e-8;
LS_ctm_setting.svd_lanczos_tol=1e-8;
LS_ctm_setting.projector_strategy="4x4";#"4x4" or "4x2"
LS_ctm_setting.conv_check="singular_value";
LS_ctm_setting.CTM_ite_info=true;
LS_ctm_setting.CTM_conv_info=true;
LS_ctm_setting.CTM_trun_svd=false;
LS_ctm_setting.construct_double_layer=true;
LS_ctm_setting.grad_checkpoint=true;
dump(LS_ctm_setting);

global Lx,Ly
Lx,Ly=size(state);
A_cell=initial_tuple_cell(Lx,Ly);
for ca=1:Lx
    for cb=1:Ly
        A_cell=fill_tuple(A_cell, state[ca,cb].T, ca,cb);
    end
end

##################################
global chi,multiplet_tol,projector_trun_tol
multiplet_tol=1e-5;
projector_trun_tol=LS_ctm_setting.CTM_trun_tol
###################################


chi=40;
init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);
#############################
#get T tensors
ca=1;
cb=1;
TL=CTM_cell.Tset[ca][cb].T4;
TR=CTM_cell.Tset[mod1(ca+1,Lx)][cb].T2;
U_L=U_L_cell[ca][cb];
U_R=U_R_cell[ca+1][cb];

#############################
#extra swap gate that was not included when construct double layer tensor
gate=swap_gate(U_L,2,3);
@tensor U_L_new[:]:=U_L'[1,2,-1]*gate[1,2,3,4]*U_L[-2,3,4];
gate=swap_gate(U_R,1,2);
@tensor U_R_new[:]:=U_R'[-1,1,2]*gate[1,2,3,4]*U_R[3,4,-2];

@tensor TL[:]:=TL[-1,1,-3]*U_R_new'[-2,1]; 
@tensor TR[:]:=TR[-1,1,-3]*U_L_new'[-2,1]; 
#############################
TL1=deepcopy(TL);
TL2=deepcopy(TL);
TL3=deepcopy(TL);
TL4=deepcopy(TL);

TR1=deepcopy(TR);
TR2=deepcopy(TR);
TR3=deepcopy(TR);
TR4=deepcopy(TR);

#############################
#extra parity gate from crossing
gate=parity_gate(TL1,3);
@tensor TL1[:]:=TL1[-1,-2,1]*gate[-3,1];
gate=parity_gate(TR1,1);
@tensor TR1[:]:=TR1[1,-2,-3]*gate[-1,1];

#############################
if y_anti_pbc
    gate=parity_gate(TL1,3);
    @tensor TL1[:]:=TL1[-1,-2,1]*gate[-3,1];
    gate=parity_gate(TR1,1);
    @tensor TR1[:]:=TR1[1,-2,-3]*gate[-1,1];
end
#############################
@tensor ML[:]:=TL1[2,-1,1]*TL2[3,-2,2]*TL3[4,-3,3]*TL4[1,-4,4];
@tensor MR[:]:=TR1[1,-1,2]*TR2[2,-2,3]*TR3[3,-3,4]*TR4[4,-4,1];

#############################








@tensor ML[:]:=ML[1,2,3,4]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7]*U_L[4,-8,-9];#R1',R1,R2',R2,R3',R3,R4',R4
ML=permute(ML,(3-1,2-1,5-1,4-1,7-1,6-1,9-1,8-1,));#R1,R1',R2,R2',R3,R3',R4,R4'
ML=permute_neighbour_ind(ML,3-1,4-1,8);#R1,R2,R1',R2',R3,R3',R4,R4'
ML=permute_neighbour_ind(ML,5-1,6-1,8);#R1,R2,R1',R3,R2',R3',R4,R4'
ML=permute_neighbour_ind(ML,4-1,5-1,8);#R1,R2,R3,R1',R2',R3',R4,R4'
ML=permute_neighbour_ind(ML,7-1,8-1,8);#R1,R2,R3,R1',R2',R4,R3',R4'
ML=permute_neighbour_ind(ML,6-1,7-1,8);#R1,R2,R3,R1',R4,R2',R3',R4'
ML=permute_neighbour_ind(ML,5-1,6-1,8);#R1,R2,R3,R4,R1',R2',R3',R4'


@tensor MR[:]:=MR[1,2,3,4]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4];#L1',L1,L2',L2,L3',L3,L4',L4
MR=permute(MR,(7,8,5,6,3,4,1,2,));#L4',L4,L3',L3,L2',L2,L1',L1
MR=permute_neighbour_ind(MR,6,7,8);#L4',L4,L3',L3,L2',L1',L2,L1
MR=permute_neighbour_ind(MR,4,5,8);#L4',L4,L3',L2',L3,L1',L2,L1
MR=permute_neighbour_ind(MR,5,6,8);#L4',L4,L3',L2',L1',L3,L2,L1
MR=permute_neighbour_ind(MR,2,3,8);#L4',L3',L4,L2',L1',L3,L2,L1
MR=permute_neighbour_ind(MR,3,4,8);#L4',L3',L2',L4,L1',L3,L2,L1
MR=permute_neighbour_ind(MR,4,5,8);#L4',L3',L2',L1',L4,L3,L2,L1



@tensor H[:]:=MR[-1,-2,-3,-4,5,4,3,2]*ML[2,3,4,5,-5,-6,-7,-8];#L4',L3',L2',L1',   R1',R2',R3',R4'
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



