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

y_anti_pbc=false;
filenm="stochastic_iPESS_LS_D_6_chi_40_3.98238.jld2";
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
if Ly==2
    A1=deepcopy(A_cell[1][1]);
    A2=deepcopy(A_cell[1][2]);
elseif Ly==1
    A1=deepcopy(A_cell[1][1]);
    A2=deepcopy(A_cell[1][1]);
end
if y_anti_pbc
    gauge_gate1=parity_gate(A1,4);
    @tensor A1[:]:=A1[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
end

if Ly==2
    B1=deepcopy(A_cell[2][1]);
    B2=deepcopy(A_cell[2][2]);
elseif Ly==1
    B1=deepcopy(A_cell[2][1]);
    B2=deepcopy(A_cell[2][1]);
end

if y_anti_pbc
    gauge_gate1=parity_gate(B1,4);
    @tensor B1[:]:=B1[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
end

A1=A1/norm(A1);
A2=A2/norm(A2);


B1=B1/norm(B1);
B2=B2/norm(B2);



#############################
#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D
A1=permute(A1,(1,4,5,3,2,));#LUPRD
A2=permute(A2,(1,4,5,3,2,));#LUPRD

B1=permute(B1,(1,4,5,3,2,));#LUPRD
B2=permute(B2,(1,4,5,3,2,));#LUPRD

#############################
function combine_mps_2row(mps1,mps2)
    mps1=deepcopy(mps1);#L1 U1 P1 R1 D1
    mps2=deepcopy(mps2);#L2 U2 P2 R2 D2
    
    mps1=permute_neighbour_ind(mps1,3,4,5);#L1 U1 R1 P1 D1

    mps2=permute_neighbour_ind(mps2,1,2,5);#U2 L2 P2 R2 D2
    mps2=permute_neighbour_ind(mps2,2,3,5);#U2 P2 L2 R2 D2

    Up=unitary(fuse(space(mps1,4)*space(mps2,2)), space(mps1,4)*space(mps2,2));
    @tensor mps12[:]:=mps1[-1,-2,-3,1,2]*mps2[2,3,-5,-6,-7]*Up[-4,1,3];#L1 U1 R1 P L2 R2 D2
    mps12=permute_neighbour_ind(mps12,4,5,7);#L1 U1 R1 L2 P R2 D2
    mps12=permute_neighbour_ind(mps12,3,4,7);#L1 U1 L2 R1 P R2 D2
    mps12=permute_neighbour_ind(mps12,2,3,7);#L1 L2 U1 R1 P R2 D2
    mps12=permute_neighbour_ind(mps12,4,5,7);#L1 L2 U1 P R1 R2 D2
    gate=swap_gate(mps12,5,6);
    @tensor mps12[:]:=mps12[-1,-2,-3,-4,1,2,-7]*gate[-5,-6,1,2];#L1 L2 U1 P R2 R1 D2 #note that only apply swap gate, but no permutation
    UL=unitary(fuse(space(mps12,1)*space(mps12,2)),space(mps12,1)*space(mps12,2));
    UR=unitary(space(mps12,5)'*space(mps12,6)',fuse(space(mps12,5)*space(mps12,6)));
    @tensor mps12[:]:=mps12[1,2,-2,-3,3,4,-5]*UL[-1,1,2]*UR[3,4,-4];#L U1 P R D2 
    return mps12,UL,UR,Up
end

####################
A12,UL12a,UR12a,Up12a=combine_mps_2row(A1,A2);
A12=permute_neighbour_ind(A12,2,3,5);#L P U R D 
A12=permute_neighbour_ind(A12,3,4,5);#L P R U D 
A12=permute_neighbour_ind(A12,4,5,5);#L P R D U 
@tensor A12[:]:=A12[-1,-2,-3,1,1];#L P R

B12,UL12b,UR12b,Up12b=combine_mps_2row(B1,B2);
B12=permute_neighbour_ind(B12,2,3,5);#L P U R D 
B12=permute_neighbour_ind(B12,3,4,5);#L P R U D 
B12=permute_neighbour_ind(B12,4,5,5);#L P R D U 
@tensor B12[:]:=B12[-1,-2,-3,1,1];#L P R
#############################




function M_vr(vr0, AAAAp,BBBBp, AAAA,BBBB)
    @tensor vr[:]:=AAAAp[-1,5,4]*AAAA[-2,5,6]*BBBBp[4,2,1]*BBBB[6,2,3]*vr0[1,3];
    return vr;
end

function vl_M(vl0, AAAAp,BBBBp, AAAA,BBBB)
    @tensor vl[:]:=AAAAp[1,2,4]*AAAA[3,2,6]*BBBBp[4,5,-1]*BBBB[6,5,-2]*vl0[1,3];
    return vl
end




v_init=TensorMap(randn, space(A12,1)',space(A12,1)');
v_init=permute(v_init,(1,2,),());#L1,L2,L3,L4,dummy
contraction_fun_R(x)=M_vr(x,A12',B12',A12,B12);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 1,:LM,Arnoldi(krylovdim=40));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,L4,dummy


v_init=TensorMap(randn, space(B12,3)',space(B12,3)');
v_init=permute(v_init,(1,2,),());#dummy,R1,R2,R3,R4
contraction_fun_L(x)=vl_M(x, A12',B12',A12,B12);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 1,:LM,Arnoldi(krylovdim=40));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3,R4



############################
function translate_AAAA(M12,Up12,y_anti_pbc)
    M12_translated=deepcopy(M12);
    @tensor M12_translated[:]:=M12_translated[-1,1,-4]*Up12'[-2,-3,1];#L,P1,P2,R
    if y_anti_pbc
        op=parity_gate(M12_translated,2);
        @tensor M12_translated[:]:=op[-2,1]*M12_translated[-1,1,-3,-4];
    end
        
    M12_translated=permute_neighbour_ind(M12_translated,2,3,4);#L,P2,P1,R
    @tensor M12_translated[:]:=M12_translated[-1,1,2,-3]*Up12[-2,1,2];
    return M12_translated
end

A12_translated=translate_AAAA(A12,Up12a,y_anti_pbc);
B12_translated=translate_AAAA(B12,Up12b,y_anti_pbc);


v_init=TensorMap(randn, space(A12,1)',space(A12,1)');
v_init=permute(v_init,(1,2,),());#L1,L2,L3,L4,dummy
contraction_fun_R(x)=M_vr(x,A12_translated',B12_translated',A12,B12);
@time eur_tran,evr_tran=eigsolve(contraction_fun_R, v_init, 1,:LM,Arnoldi(krylovdim=40));
VR_tran=evr_tran[findmax(abs.(eur_tran))[2]];#L1,L2,L3,L4,dummy


v_init=TensorMap(randn, space(B12,3)',space(B12,3)');
v_init=permute(v_init,(1,2,),());#dummy,R1,R2,R3,R4
contraction_fun_L(x)=vl_M(x, A12_translated',B12_translated',A12,B12);
@time eul_tran,evl_tran=eigsolve(contraction_fun_L, v_init, 1,:LM,Arnoldi(krylovdim=40));
VL_tran=evl_tran[findmax(abs.(eul_tran))[2]];#dummy,R1,R2,R3,R4



@tensor H[:]:=VR[1,-1]*VL[1,-2];
eu,ev=eig(H,(1,),(2,));

@tensor H_tran[:]:=VR_tran[1,-1]*VL[1,-2];
eu_tran,ev_tran=eig(H_tran,(1,),(2,));



Spin=get_Vspace_Spin(space(eu_tran,1));Spin=Float64.(Spin);


eu=diag(convert(Array,eu));
eu=eu/norm(eu)
eu_tran=diag(convert(Array,eu_tran));
eu_tran=eu_tran/norm(eu_tran)

pos_max=findall(x -> x == maximum(abs.(eu_tran)), abs.(eu_tran));
eu_tran=eu_tran/eu_tran[pos_max[1]];


k_phase=deepcopy(eu_tran);
for cc=1:length(eu_tran)
    k_phase[cc]=(eu_tran[cc])'/abs(eu_tran[cc]);
    eu_tran[cc]=abs(eu_tran[cc]);
end

order=sortperm(abs.(eu));
if length(order)>500
    order=order[end-500:end];
end

eu=eu_tran[order];
k_phase=k_phase[order];
Spin=Spin[order]


##########################

if y_anti_pbc
    matnm="ES_unprojected_M1_Nv2_APBC"*".mat";
else
    matnm="ES_unprojected_M1_Nv2_PBC"*".mat";
end
matwrite(matnm, Dict(
    "k_phase" => k_phase,
    "eu" => eu,
    "Spin"=>Spin
); compress = false)



