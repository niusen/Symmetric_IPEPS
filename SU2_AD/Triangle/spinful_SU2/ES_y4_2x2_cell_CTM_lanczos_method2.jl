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
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")

function get_CTM(y_translation)

    y_anti_pbc=true;
    filenm="stochastic_iPESS_LS_D_6_chi_40.jld2";
    data=load(filenm);
    # A=data["x"][1].T;
    # B=data["x"][2].T;
    state=data["x"]
    # state=data["T_set"]
    #convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


    algrithm_CTMRG_settings=Algrithm_CTMRG_settings()
    algrithm_CTMRG_settings.CTM_cell_ite_method= "continuous_update";#"continuous_update", "together_update"
    dump(algrithm_CTMRG_settings);
    global algrithm_CTMRG_settings

    LS_ctm_setting=LS_CTMRG_settings();
    LS_ctm_setting.CTM_conv_tol=1e-6;
    LS_ctm_setting.CTM_ite_nums=30;
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
            if isa(state[ca,cb],Square_iPEPS)
                A_cell=fill_tuple(A_cell, state[ca,cb].T, ca,cb);
            elseif isa(state[ca,cb],Triangle_iPESS)
                A0=iPESS_to_iPEPS(state[ca,cb])
                A_cell=fill_tuple(A_cell, A0.T, ca,cb);
            elseif isa(state[ca,cb],TensorMap)
                A_cell=fill_tuple(A_cell, state[ca,cb], ca,cb);
            else
                error("unknown type ansatz")
            end
        end
    end

    ##################################
    global chi,multiplet_tol,projector_trun_tol
    multiplet_tol=1e-5;
    projector_trun_tol=LS_ctm_setting.CTM_trun_tol
    ###################################


    chi=40;
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    if y_translation
        CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell_y_translate(A_cell,chi,init, init_CTM,LS_ctm_setting);
    else
        CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,LS_ctm_setting);
    end
    return CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell
end
#############################

y_translation=false;
CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell=get_CTM(y_translation);
y_translation=true;
CTM_cell_tran, AA_cell_tran, U_L_cell_tran,U_D_cell_tran,U_R_cell_tran,U_U_cell_tran=get_CTM(y_translation);

############################
#get T tensors
ca=1;
cb=1;
TL01=CTM_cell.Tset[ca][cb].T4;
TR01=CTM_cell_tran.Tset[mod1(ca+1,Lx)][cb].T2;
U_L=U_L_cell[ca][cb];
U_R=U_R_cell_tran[ca+1][cb];

cb=2;
TL02=CTM_cell.Tset[ca][cb].T4;
TR02=CTM_cell_tran.Tset[mod1(ca+1,Lx)][cb].T2;
U_L=U_L_cell[ca][cb];
U_R=U_R_cell_tran[ca+1][cb];

#############################
#extra swap gate that was not included when construct double layer tensor
gate=swap_gate(U_L,2,3);
@tensor U_L_new[:]:=U_L'[1,2,-1]*gate[1,2,3,4]*U_L[-2,3,4];
gate=swap_gate(U_R,1,2);
@tensor U_R_new[:]:=U_R'[-1,1,2]*gate[1,2,3,4]*U_R[3,4,-2];

@tensor TL01[:]:=TL01[-1,1,-3]*U_R_new'[-2,1]; 
@tensor TR01[:]:=TR01[-1,1,-3]*U_L_new'[-2,1]; 
@tensor TL02[:]:=TL02[-1,1,-3]*U_R_new'[-2,1]; 
@tensor TR02[:]:=TR02[-1,1,-3]*U_L_new'[-2,1]; 

TL01=TL01*10;
TR01=TR01*10;
TL02=TL02*10;
TR02=TR02*10;
#############################
TL1=deepcopy(TL01);
TL2=deepcopy(TL02);
TL3=deepcopy(TL01);
TL4=deepcopy(TL02);

TR1=deepcopy(TR01);
TR2=deepcopy(TR02);
TR3=deepcopy(TR01);
TR4=deepcopy(TR02);

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

@tensor TR1[:]:=TR1[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR2[:]:=TR2[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR3[:]:=TR3[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U
@tensor TR4[:]:=TR4[-4,1,-2]*U_L'[-1,-3,1]; #L',D,L',U

#############################
#apply on-site swap gate on T tensors
TL_set=[TL1,TL2,TL3,TL4];
TR_set=[TR1,TR2,TR3,TR4];
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

TL1,TL2,TL3,TL4=TL_set;
TR1,TR2,TR3,TR4=TR_set;

#############################

P_odda,P_evena=projector_parity(space(TL1,4));
P_oddb,P_evenb=projector_parity(space(TR1,4));
P_odda=P_odda'*P_odda;
P_evena=P_evena'*P_evena;
P_oddb=P_oddb'*P_oddb;
P_evenb=P_evenb'*P_evenb;

gate=parity_gate(TL1,3);

@tensor ML_even[:]:=P_evena[1,2]*TL1[-1,3,-5,2]*TL2[-2,4,-6,3]*TL3[-3,5,-7,4]*TL4[-4,1,-8,5];
@tensor ML_odd[:]:=P_odda[1,2]*TL1[-1,3,-5,2]*TL2[-2,4,-6,3]*TL3[-3,5,-7,4]*TL4[-4,1,-8,5];
@tensor ML_odd[:]:=ML_odd[-1,-2,-3,-4,1,2,3,-8]*gate[-5,1]*gate[-6,2]*gate[-7,3];

@tensor MR_even[:]:=P_evenb[1,2]*TR1[-1,3,-5,2]*TR2[-2,4,-6,3]*TR3[-3,5,-7,4]*TR4[-4,1,-8,5];
@tensor MR_odd[:]:=P_oddb[1,2]*TR1[-1,3,-5,2]*TR2[-2,4,-6,3]*TR3[-3,5,-7,4]*TR4[-4,1,-8,5];
@tensor MR_odd[:]:=MR_odd[1,2,3,-4,-5,-6,-7,-8]*gate[1,-1]*gate[2,-2]*gate[3,-3];

ML_even=permute(ML_even,(1,2,3,4,),(5,6,7,8,));
ML_odd=permute(ML_odd,(1,2,3,4,),(5,6,7,8,));
MR_even=permute(MR_even,(1,2,3,4,),(5,6,7,8,));
MR_odd=permute(MR_odd,(1,2,3,4,),(5,6,7,8,));

H=ML_even*MR_even+ML_even*MR_odd+ML_odd*MR_even+ML_odd*MR_odd;

#######################

if y_anti_pbc
    op=parity_gate(H,8);
    @tensor H[:]:=op[-5,1]*H[-1,-2,-3,-4,1,-6,-7,-8];
end
H=permute_neighbour_ind(H,5,6,8);
H=permute_neighbour_ind(H,6,7,8);
H=permute_neighbour_ind(H,7,8,8);


# if y_anti_pbc
#     op=parity_gate(H,8);
#     @tensor H[:]:=op[-8,1]*H[-1,-2,-3,-4,-5,-6,-7,1];
# end
# H=permute_neighbour_ind(H,7,8,8);
# H=permute_neighbour_ind(H,6,7,8);
# H=permute_neighbour_ind(H,5,6,8);
#######################


H=permute(H,(1,2,3,4,),(5,6,7,8,));

eu,ev=eig(H);
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);

eu=diag(convert(Array,eu));




println(sort(abs.(eu)))

pos_max=findall(x -> x == maximum(abs.(eu)), abs.(eu));
eu=eu/eu[pos_max[1]];


k_phase=deepcopy(eu);
for cc=1:length(eu)
    k_phase[cc]=(eu[cc])'/abs(eu[cc]);
    eu[cc]=abs(eu[cc]);
end



order=sortperm(abs.(eu));
if length(order)>500
    order=order[end-500:end];
end

eu=eu[order];
k_phase=k_phase[order];
Spin=Spin[order]


##########################
D=dim(space(A_cell[1][1],1));

if y_anti_pbc
    matnm="ES_CTM_D"*string(D)*"_Nv4_APBC"*".mat";
else
    matnm="ES_CTM_D"*string(D)*"_Nv4_PBC"*".mat";
end
matwrite(matnm, Dict(
    "k_phase" => k_phase,
    "eu" => eu,
    "Spin"=>Spin
); compress = false)