using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
cd("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
#push!(LOAD_PATH, "D:\\My Documents\\Code\\Julia_codes\\Tensor network\\IPEPS_TensorKit\\kagome\\SU2_PG")
include("kagome_load_tensor.jl")
include("kagome_CTMRG.jl")
include("kagome_model.jl")
include("kagome_IPESS.jl")
include("mps_algorithms\\ITEBD_algorithms.jl")
include("mps_algorithms\\TransfOp_decomposition.jl")
include("mps_algorithms\\PUMPS_algorithms.jl")
include("mps_algorithms\\ES_preliminary.jl")



D=8;
chi=20;
W=20;
N=20;
tol=1e-6;
EH_n=3;#number of entanglement spectrum per k point

multi_threads=true;
Dtrun_init=200;
Dtrun_max=200;
trun_tol=1e-8;
group_size=Int(round((10^8)/(chi*chi*W*W*D)));

mpo_type="OO";#"O_O" or "OO", in my test "OO" is faster for large bond dimension

pow=Int((N-2)/2);

J1=1;
J2=0;
J3=0;
Jchi=0;
Jtrip=0;





parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);

A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

filenm="LS_D_"*string(D)*"_chi_40.json"
json_dict=read_json_state(filenm);

bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

PEPS_tensor=bond_tensor;
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];



CTM,U_L,U_D,U_R,U_U=try_CTM(D,chi,parameters, U_phy, A_unfused, A_fused);


Ag,O1,O2=try_ITEBD(D,chi,W,CTM);




space_AOA=fuse(space(Ag,1)'⊗space(O2,1)'⊗space(O1,1)⊗ space(Ag,1));
space_AA=fuse(space(Ag,1)'⊗ space(Ag,1));

AOA_sec=collect(sectors(space_AOA))
AA_sec=collect(sectors(space_AA))

@tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
@tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];
display(space(OO))



#normalize the MPO
euL_set,_,_,_,_=FLR_eig(Ag,OO,20,space_AOA,AOA_sec);
norm_coe=maximum(abs.(group_numbers(euL_set)));
OO=OO/norm_coe;
O1=O1/norm_coe;





euR_set,evL_set,evR_set,SPIN_eig_set=TransfOp_decom(Ag,OO,space_AOA,AOA_sec,pow,Dtrun_init,Dtrun_max,trun_tol,"eigenvalue_FLR");
# display(euR_set)

eur_set,evl_set,evr_set,spin_eig_set=TransfOp_decom(Ag,OO,space_AA,AA_sec,pow,Dtrun_init,Dtrun_max,trun_tol,"eigenvalue_GLR");
# display(eur_set)

S_set,U_set,Vh_set,SPIN_svd_set=TransfOp_decom(Ag,OO,space_AOA,AOA_sec,pow,Dtrun_init,Dtrun_max,trun_tol,"svd_FLR");
# display(S_set)

s_set,u_set,vh_set,spin_svd_set=TransfOp_decom(Ag,OO,space_AA,AA_sec,pow,Dtrun_init,Dtrun_max,trun_tol,"svd_GLR");
# display(s_set)


check_truncated_decomp_error=false;

if mpo_type=="O_O"
    OO_transform=true;
elseif mpo_type=="OO"
    OO_transform=false;
end

euR_set_combined,evL_set_combined,evR_set_combined,SPIN_eig_set_combined=combine_singlespin_sector(euR_set,evL_set,evR_set,SPIN_eig_set,true);
euR_set_grouped,evL_set_grouped,evR_set_grouped,SPIN_eig_set_grouped,DTrun_FLR_eig=group_singlespin_sector(group_size,euR_set_combined,evL_set_combined,evR_set_combined,SPIN_eig_set_combined,OO_transform,U_fuse_chichi)
display("group information:")
display(DTrun_FLR_eig)

eur_set_combined,evl_set_combined,evr_set_combined,spin_eig_set_combined=combine_singlespin_sector(eur_set,evl_set,evr_set,spin_eig_set,true)
eur_set_grouped,evl_set_grouped,evr_set_grouped,spin_eig_set_grouped,Dtrun_GLR_eig=group_singlespin_sector(group_size,eur_set_combined,evl_set_combined,evr_set_combined,spin_eig_set_combined,false,[])
display("group information:")
display(Dtrun_GLR_eig)


S_set_combined,Vh_set_combined,U_set_combined,SPIN_svd_set_combined=combine_singlespin_sector(S_set,Vh_set,U_set,SPIN_svd_set,false)
S_set_grouped,Vh_set_grouped,U_set_grouped,SPIN_svd_set_grouped,DTrun_FLR_svd=group_singlespin_sector(group_size,S_set_combined,Vh_set_combined,U_set_combined,SPIN_svd_set_combined,OO_transform,U_fuse_chichi)
display("group information:")
display(DTrun_FLR_svd)

s_set_combined,vh_set_combined,u_set_combined,spin_svd_set_combined=combine_singlespin_sector(s_set,vh_set,u_set,spin_svd_set,false)
s_set_grouped,vh_set_grouped,u_set_grouped,spin_svd_set_grouped,Dtrun_GLR_svd=group_singlespin_sector(group_size,s_set_combined,vh_set_combined,u_set_combined,spin_svd_set_combined,false,[])
display("group information:")
display(Dtrun_GLR_svd)



Dtrun_method="svds";
ES_sectors=[0,1/2,1,3/2,2,5/2];
kset=0:N-1;
#kset=0:0
Eset=[];
Trun_err=0;
DTrun=0;
print("calculate ES for N="*string(N));
display("kset="*string(kset));
pow=round((N-2)/2);
k=0;



norm_eff=excitation_TrunTransOp_iterative_norm_eff(Ag,pow,N,k) # put it on cpu because this matrix maybe large
norm_eff=permute(norm_eff,(1,2,3,),(4,5,6,))
uu,ss,vvt=tsvd(norm_eff, trunc=truncerr(0.0000001));
norm_eff=[]#clear this big matrix
input_transform=vvt';
output_transform=uu';
output_transform=output_transform;
output_transform=pinv(ss)*output_transform;

sector_ind=1;
SPIN=ES_sectors[sector_ind];
sectr=Irrep[SU₂](SPIN);

v_init=TensorMap(randn,domain(input_transform), SU₂Space(SPIN=>1));




mpo_type="OO"
@time excitation_TrunTransOp_iterative_H_eff(v_init,input_transform,output_transform,O1,O2,OO,Ag,pow,U_set_grouped,S_set_grouped,Vh_set_grouped,SPIN_svd_set_grouped,N,k,DTrun_FLR_svd,mpo_type,multi_threads)
@time excitation_TrunTransOp_iterative_H_eff(v_init,input_transform,output_transform,O1,O2,OO,Ag,pow,U_set_grouped,S_set_grouped,Vh_set_grouped,SPIN_svd_set_grouped,N,k,DTrun_FLR_svd,mpo_type,multi_threads)
@time excitation_TrunTransOp_iterative_H_eff(v_init,input_transform,output_transform,O1,O2,OO,Ag,pow,U_set_grouped,S_set_grouped,Vh_set_grouped,SPIN_svd_set_grouped,N,k,DTrun_FLR_svd,mpo_type,multi_threads)


