using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model_string.jl")
include("kagome_model.jl")
include("kagome_IPESS.jl")



D=6;
chi=20;
N=20;


J1=0.80902;
J2=0;
J3=0;
Jchi=0;
Jtrip=0.5878;

parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);


CTM_conv_tol=1e-6;
CTM_ite_nums=50;
CTM_trun_tol=1e-12;


A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

filenm="LS_D_"*string(D)*"_chi_40.json"
json_dict=read_json_state(filenm);

bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

PEPS_tensor=bond_tensor;
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

CTM=[];
U_L=[];
U_D=[];
U_R=[];
U_U=[];

init=Dict([("CTM", []), ("init_type", "PBC")]);


conv_check="singular_value";
CTM_ite_info=false;
CTM, AA_fused, U_L,U_D,U_R,U_U,AAR__,AARZ_,AAR_Z,AARZZ,T1__,T1Z_,T1_Z,T1ZZ,T3__,T3Z_,T3_Z,T3ZZ=CTMRG_string(A_fused,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol,CTM_ite_info);

@assert norm(T1__-CTM["Tset"][1])/norm(CTM["Tset"][1])<1e-10
@assert norm(T3__-CTM["Tset"][3])/norm(CTM["Tset"][3])<1e-10

E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
energy=(E_up+E_down)/3;
println("energy:"*string(real(energy)))
  
#check if unitary (the operation of fusing legs) commutes with the gauge transformation
# Z1=gauge_operator(space(U_L,1));
# Z2=gauge_operator(space(U_L,2));
# Z3=gauge_operator(space(U_L,3));
# @tensor U_La[:]:=Z1[-1,1]*U_L[1,-2,-3];
# @tensor U_Lb[:]:=U_L[-1,1,2]*Z2[-2,1]*Z3[-3,2];
# println(norm(U_La-U_Lb)/norm(U_La))
# println(norm(U_La)/norm(U_L))
# println(Z2)


H_triangle,_,_,_=Hamiltonians(U_phy,parameters["J1"],parameters["J2"],parameters["J3"],parameters["Jchi"],parameters["Jtrip"])

AA,_,_,_,_=build_double_layer(A_fused,[]);
AZA=build_double_layer_string(A_fused,[],"D",nothing);
AAZ=build_double_layer_string(A_fused,[],nothing,"D");
AZAZ=build_double_layer_string(A_fused,[],"D","D");

AHA,_,_,_,_=build_double_layer(A_fused,H_triangle);
AZHA=build_double_layer_string(A_fused,H_triangle,"D",nothing);
AHAZ=build_double_layer_string(A_fused,H_triangle,nothing,"D");
AZHAZ=build_double_layer_string(A_fused,H_triangle,"D","D");


Tup=CTM["Tset"][1];
Tdown=CTM["Tset"][3];

@tensor TAAT[:]:=Tup[-1,1,-4]*AA[-2,2,-5,1]*Tdown[-6,2,-3];
TAAT=permute(TAAT,(1,2,3,),(4,5,6,));
@tensor TAHAT[:]:=Tup[-1,1,-4]*AHA[-2,2,-5,1]*Tdown[-6,2,-3];
TAHAT=permute(TAHAT,(1,2,3,),(4,5,6,));

@tensor TAZAT[:]:=Tup[-1,1,-4]*AZA[-2,2,-5,1]*Tdown[-6,2,-3];
TAZAT=permute(TAZAT,(1,2,3,),(4,5,6,));
@tensor TAZHAT[:]:=Tup[-1,1,-4]*AZHA[-2,2,-5,1]*Tdown[-6,2,-3];
TAZHAT=permute(TAZHAT,(1,2,3,),(4,5,6,));

@tensor TAAZT[:]:=Tup[-1,1,-4]*AAZ[-2,2,-5,1]*Tdown[-6,2,-3];
TAAZT=permute(TAAZT,(1,2,3,),(4,5,6,));
@tensor TAHAZT[:]:=Tup[-1,1,-4]*AHAZ[-2,2,-5,1]*Tdown[-6,2,-3];
TAHAZT=permute(TAHAZT,(1,2,3,),(4,5,6,));

@tensor TAZAZT[:]:=Tup[-1,1,-4]*AZAZ[-2,2,-5,1]*Tdown[-6,2,-3];
TAZAZT=permute(TAZAZT,(1,2,3,),(4,5,6,));
@tensor TAZHAZT[:]:=Tup[-1,1,-4]*AZHAZ[-2,2,-5,1]*Tdown[-6,2,-3];
TAZHAZT=permute(TAZHAZT,(1,2,3,),(4,5,6,));




bulk=TAAT;
bulkZ=TAAZT;
Zbulk=TAZAT;
ZbulkZ=TAZAZT;
for cc=2:N-1
    bulk=bulk*TAAT;
    bulkZ=bulkZ*TAAZT;
    Zbulk=Zbulk*TAZAT;
    ZbulkZ=ZbulkZ*TAZAZT;
end

@tensor E[:]:=bulk[1,2,3,4,5,6]*TAHAT[4,5,6,1,2,3];
@tensor Norm[:]:=bulk[1,2,3,4,5,6]*TAAT[4,5,6,1,2,3];
E=blocks(E)[Irrep[SU₂](0)][1];
Norm=blocks(Norm)[Irrep[SU₂](0)][1];

@tensor EZ[:]:=bulkZ[1,2,3,4,5,6]*TAHAZT[4,5,6,1,2,3];
@tensor NormZ[:]:=bulkZ[1,2,3,4,5,6]*TAAZT[4,5,6,1,2,3];
EZ=blocks(EZ)[Irrep[SU₂](0)][1];
NormZ=blocks(NormZ)[Irrep[SU₂](0)][1];

@tensor ZE[:]:=Zbulk[1,2,3,4,5,6]*TAZHAT[4,5,6,1,2,3];
@tensor ZNorm[:]:=Zbulk[1,2,3,4,5,6]*TAZAT[4,5,6,1,2,3];
ZE=blocks(ZE)[Irrep[SU₂](0)][1];
ZNorm=blocks(ZNorm)[Irrep[SU₂](0)][1];

@tensor ZEZ[:]:=ZbulkZ[1,2,3,4,5,6]*TAZHAZT[4,5,6,1,2,3];
@tensor ZNormZ[:]:=ZbulkZ[1,2,3,4,5,6]*TAZAZT[4,5,6,1,2,3];
ZEZ=blocks(ZEZ)[Irrep[SU₂](0)][1];
ZNormZ=blocks(ZNormZ)[Irrep[SU₂](0)][1];

EE=E/Norm*2/3;
println(EE)

E_even=(E+EZ+ZE+ZEZ)/4;
Norm_even=(Norm+NormZ+ZNorm+ZNormZ)/4;
EE_even=E_even/Norm_even*2/3;
println(EE_even)

E_odd=(E-EZ-ZE+ZEZ)/4;
Norm_odd=(Norm-NormZ-ZNorm+ZNormZ)/4;
EE_odd=E_odd/Norm_odd*2/3;
println(EE_odd)





################
# C1=CTM["Cset"][1];
# C2=CTM["Cset"][2];
# C3=CTM["Cset"][3];
# C4=CTM["Cset"][4];
# Tleft=CTM["Tset"][4];
# Tright=CTM["Tset"][2];
# @tensor Vl[:]:=C1[1,-1]*Tleft[2,-2,1]*C4[-3,2];
# @tensor Vr[:]:=C2[-1,1]*Tright[1,-2,2]*C3[2,-3];
# @tensor E[:]:=Vl[4,5,6]*TAHAT[4,5,6,1,2,3]*Vr[1,2,3];
# @tensor Norm[:]:=Vl[4,5,6]*TAAT[4,5,6,1,2,3]*Vr[1,2,3];
# E=E/Norm;
# E=blocks(E)[Irrep[SU₂](0)][1]*2/3;
# print(E)
#varinfo()
###################