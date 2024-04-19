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

y_anti_pbc=false;
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

TL=TL*10;
TR=TR*10;
#############################
TL1=deepcopy(TL);
TL2=deepcopy(TL);
TL3=deepcopy(TL);
TL4=deepcopy(TL);
TL5=deepcopy(TL);
TL6=deepcopy(TL);

TR1=deepcopy(TR);
TR2=deepcopy(TR);
TR3=deepcopy(TR);
TR4=deepcopy(TR);
TR5=deepcopy(TR);
TR6=deepcopy(TR);

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
#expand T tensors
@tensor TL1[:]:=TL1[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL2[:]:=TL2[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL3[:]:=TL3[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL4[:]:=TL4[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL5[:]:=TL5[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U
@tensor TL6[:]:=TL6[-2,1,-4]*U_L[1,-3,-1]; #R,D,R',U

@tensor TR1[:]:=TR1[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR2[:]:=TR2[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR3[:]:=TR3[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR4[:]:=TR4[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR5[:]:=TR5[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR6[:]:=TR6[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U

#############################
#apply on-site swap gate on T tensors
TL_set=[TL1,TL2,TL3,TL4,TL5,TL6];
TR_set=[TR1,TR2,TR3,TR4,TR5,TR6];
for cc=1:length(TL_set)-1
    tl=TL_set[cc];
    gate=swap_gate(tl,2,3);
    @tensor tl[:]:=tl[-1,1,2,-4]*gate[-2,-3,1,2];
    TL_set[cc]=tl;

    tr=TR_set[cc];
    gate=swap_gate(tr,1,2);
    @tensor tr[:]:=tr[1,2,-3,-4]*gate[-1,-2,1,2];
    TR_set[cc]=tr;
end

TL1,TL2,TL3,TL4,TL5,TL6=TL_set;
TR1,TR2,TR3,TR4,TR5,TR6=TR_set;

#############################
P_odda,P_evena=projector_parity(space(TL1,4));
P_oddb,P_evenb=projector_parity(space(TR1,4));

P_seta=[P_odda,P_evena];
P_setb=[P_oddb,P_evenb];

gate_middle=parity_gate(TL1,3);

function vr_ML_MR(vr0,  TL_set,TR_set, P_seta,P_setb, gate_middle)
    println("apply Mr");flush(stdout);
    TL1,TL2,TL3,TL4,TL5,TL6=TL_set;
    TR1,TR2,TR3,TR4,TR5,TR6=TR_set;
    vr=deepcopy(vr0)*0;
    for ca=1:2#parity of index U in ML
        @tensor TL1_tem[:]:=TL1[-1,-2,-3,1]*P_seta[ca][-4,1];
        TL2_tem=deepcopy(TL2);
        TL3_tem=deepcopy(TL3);
        TL4_tem=deepcopy(TL4);
        TL5_tem=deepcopy(TL5);
        @tensor TL6_tem[:]:=TL6[-1,1,-3,-4]*P_seta[ca]'[1,-2];
        if mod(ca,2)==1
            @tensor TL1_tem[:]:=TL1_tem[-1,-2,1,-4]*gate_middle[-3,1];
            @tensor TL2_tem[:]:=TL2_tem[-1,-2,1,-4]*gate_middle[-3,1];
            @tensor TL3_tem[:]:=TL3_tem[-1,-2,1,-4]*gate_middle[-3,1];
            @tensor TL4_tem[:]:=TL4_tem[-1,-2,1,-4]*gate_middle[-3,1];
            @tensor TL5_tem[:]:=TL5_tem[-1,-2,1,-4]*gate_middle[-3,1];
        end
        for cb=1:2#parity of index U in MR
            @tensor TR1_tem[:]:=TR1[-1,-2,-3,1]*P_setb[cb][-4,1];
            TR2_tem=deepcopy(TR2);
            TR3_tem=deepcopy(TR3);
            TR4_tem=deepcopy(TR4);
            TR5_tem=deepcopy(TR5);
            @tensor TR6_tem[:]:=TR6[-1,1,-3,-4]*P_setb[cb]'[1,-2];
            if mod(cb,2)==1
                @tensor TR1_tem[:]:=TR1_tem[1,-2,-3,-4]*gate_middle'[1,-1];
                @tensor TR2_tem[:]:=TR2_tem[1,-2,-3,-4]*gate_middle'[1,-1];
                @tensor TR3_tem[:]:=TR3_tem[1,-2,-3,-4]*gate_middle'[1,-1];
                @tensor TR4_tem[:]:=TR4_tem[1,-2,-3,-4]*gate_middle'[1,-1];
                @tensor TR5_tem[:]:=TR5_tem[1,-2,-3,-4]*gate_middle'[1,-1];
            end

            
            @tensor vr_temp[:]:=TR1_tem[-1,2,1,12]*TR2_tem[-2,4,3,2]*TR3_tem[-3,6,5,4]*TR4_tem[-4,8,7,6]*TR5_tem[-5,10,9,8]*TR6_tem[-6,12,11,10]*vr0[1,3,5,7,9,11,-7];
            @tensor vr_temp[:]:=TL1_tem[-1,2,1,12]*TL2_tem[-2,4,3,2]*TL3_tem[-3,6,5,4]*TL4_tem[-4,8,7,6]*TL5_tem[-5,10,9,8]*TL6_tem[-6,12,11,10]*vr_temp[1,3,5,7,9,11,-7];
            vr=vr+vr_temp;
        end
    end
    return vr
end

function compute_k(ev,y_anti_pbc)
    ev0=deepcopy(ev);
    if y_anti_pbc
        ev=permute(ev,(1,2,3,4,5,6,7,));#L1',L2',L3',L4',dummy
        op=parity_gate(ev,6);
        @tensor ev_translation[:]:=op[1,-1]*ev'[1,-2,-3,-4,-5,-6,-7];
        ev_translation=permute_neighbour_ind(deepcopy(ev_translation),1,2,7);#L2',L1',L3',L4',dummy
        ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,7);#L2',L3',L1',L4',dummy
        ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,7);#L2',L3',L4',L1',dummy
        ev_translation=permute_neighbour_ind(deepcopy(ev_translation),4,5,7);#L2',L3',L4',L1',dummy
        ev_translation=permute_neighbour_ind(deepcopy(ev_translation),5,6,7);#L2',L3',L4',L1',dummy
    else
        ev=permute(ev,(1,2,3,4,5,6,7,));#L1',L2',L3',L4',dummy
        ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,7);#L2',L1',L3',L4',dummy
        ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,7);#L2',L3',L1',L4',dummy
        ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,7);#L2',L3',L4',L1',dummy
        ev_translation=permute_neighbour_ind(deepcopy(ev_translation),4,5,7);#L2',L3',L4',L1',dummy
        ev_translation=permute_neighbour_ind(deepcopy(ev_translation),5,6,7);#L2',L3',L4',L1',dummy
    end
    
    
    k_phase=@tensor ev_translation[1,2,3,4,5,6,7]*ev[1,2,3,4,5,6,7];
    Norm=@tensor ev0'[1,2,3,4,5,6,7]*ev0[1,2,3,4,5,6,7];
    k_phase=k_phase/Norm
    return k_phase'
end

Spin_set=[0,1/2,1,3/2,2,5/2];
n_Es=[100,80,40,30,20,20];
eu=Vector{ComplexF64}(undef,0);
k_phase=Vector{ComplexF64}(undef,0);
Spin=Vector{Float64}(undef,0);
for ss=1:length(Spin_set)
    v_init=TensorMap(randn, space(TL1,1)*space(TL2,1)*space(TL3,1)*space(TL4,1)*space(TL5,1)*space(TL6,1),Rep[SU₂](Spin_set[ss]=>1));
    v_init=permute(v_init,(1,2,3,4,5,6,7,),());#L1,L2,L3,L4,dummy
    if norm(v_init)==0
        continue;
    end
    contraction_fun_R(x)=vr_ML_MR(x, TL_set,TR_set, P_seta,P_setb, gate_middle);
    @time eu0,ev=eigsolve(contraction_fun_R, v_init, n_Es[ss],:LM,Arnoldi(krylovdim=n_Es[ss]+20));
    ks=Vector{ComplexF64}(undef,length(eu0));
    spins=Vector{ComplexF64}(undef,length(eu0));
    for cc=1:length(eu0)
        ks[cc]=compute_k(ev[cc],y_anti_pbc);
        spins[cc]=Spin_set[ss];
    end
    eu=vcat(eu,eu0);
    k_phase=vcat(k_phase,ks);
    Spin=vcat(Spin,spins);
end

##############################

eu=eu/sum(eu);
println(sort(abs.(eu)))

order=sortperm(abs.(eu));
if length(order)>500
    order=order[end-500:end];
end
eu=eu[order];
k_phase=k_phase[order];
Spin=Spin[order]


##########################

if y_anti_pbc
    matnm="ES_unprojected_M1_Nv6_APBC"*".mat";
else
    matnm="ES_unprojected_M1_Nv6_PBC"*".mat";
end
matwrite(matnm, Dict(
    "k_phase" => k_phase,
    "eu" => eu,
    "Spin"=>Spin
); compress = false)



