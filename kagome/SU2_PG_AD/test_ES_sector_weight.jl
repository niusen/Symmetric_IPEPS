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
include("kagome_model_string.jl")
include("mps_algorithms/ITEBD_algorithms.jl")
include("mps_algorithms/TransfOp_decomposition.jl")
include("mps_algorithms/PUMPS_algorithms.jl")
include("mps_algorithms/ES_preliminary.jl")




D=8;
chi=20;
N=8
EH_n=10;#number of entanglement spectrum
filenm="julia_LS_D_6_chi_20.json"
#filenm="LS_D_3_chi_40.json"

J1=0.80902;
J2=0;
J3=0;
Jchi=0;
Jtrip=0.5878;
parameters=Dict([("J1", J1), ("J2", J2), ("J3", J3), ("Jchi", Jchi), ("Jtrip", Jtrip)]);

CTM_conv_tol=1e-6;
CTM_ite_nums=200;
CTM_trun_tol=1e-12;
group_index=true;

println("D="*string(D));
println("chi="*string(chi));



mpo_type="OO";#"O_O" or "OO", in my test "OO" is faster for large bond dimension


A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);

#filenm="LS_D_"*string(D)*"_chi_40.json"
json_dict=read_json_state(filenm);

bond_tensor,triangle_tensor=construct_su2_PG_IPESS(json_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

PEPS_tensor=bond_tensor;
@tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
A_unfused=PEPS_tensor;

U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
@tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];


CTM,U_L,U_D,U_R,U_U=try_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,U_phy, A_unfused, A_fused);


Tleft=CTM["Tset"][4];
Tright=CTM["Tset"][2];
@tensor O1[:]:=Tleft[-3,1,-1]*U_L[1,-2,-4];
@tensor O2[:]:=Tright[-1,1,-3]*U_R[-4,-2,1];

@tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
@tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];

U_fuse_DD=unitary(fuse(space(O1,2)⊗ space(O1,2)),space(O1,2)'⊗ space(O1,2)');
if group_index
   @tensor O1_O1[:]:=O1[-1,1,2,4]*O1[2,3,-3,5]*U_fuse_DD'[1,3,-2]*U_fuse_DD[-4,4,5];
   @tensor O2_O2[:]:=O2[-1,1,2,4]*O2[2,3,-3,5]*U_fuse_DD'[1,3,-2]*U_fuse_DD[-4,4,5];
   O1_O1=O1_O1/norm(O1_O1);
   O2_O2=O2_O2/norm(O2_O2);
   if N==8
        U_fuse_DD_D=unitary(fuse(space(O1_O1,2)⊗ space(O1,2)),space(O1_O1,2)'⊗ space(O1,2)');
        @tensor O1_O1_O1[:]:=O1_O1[-1,1,2,4]*O1[2,3,-3,5]*U_fuse_DD_D'[1,3,-2]*U_fuse_DD_D[-4,4,5];
        @tensor O2_O2_O2[:]:=O2_O2[-1,1,2,4]*O2[2,3,-3,5]*U_fuse_DD_D'[1,3,-2]*U_fuse_DD_D[-4,4,5];
        O1_O1_O1=O1_O1_O1/norm(O1_O1_O1);
        O2_O2_O2=O2_O2_O2/norm(O2_O2_O2);
        @tensor a_bcd_To_abc_d[:]:=U_fuse_DD_D[-1,3,4]*U_fuse_DD[3,-3,2]*U_fuse_DD'[2,4,1]*U_fuse_DD_D'[1,-2,-4];
   else
        U_fuse_DD_D=nothing;
        O1_O1_O1=nothing;
        O2_O2_O2=nothing;
        a_bcd_To_abc_d=nothing;
   end
end

println("2-site MPO")
@tensor OO[:]:=OO[-1,1,2,4]*OO[2,3,-3,5]*U_fuse_DD'[1,3,-2]*U_fuse_DD[-4,4,5];

Z1=gauge_operator(space(OO,1));
Z2=gauge_operator(space(OO,2));
Z3=gauge_operator(space(OO,3));
Z4=gauge_operator(space(OO,4));

P1even=(Z1+Z1*Z1)/2;
P1odd=(-Z1+Z1*Z1)/2;
P2even=(Z2+Z2*Z2)/2;
P2odd=(-Z2+Z2*Z2)/2;
P3even=(Z3+Z3*Z3)/2;
P3odd=(-Z3+Z3*Z3)/2;
P4even=(Z4+Z4*Z4)/2;
P4odd=(-Z4+Z4*Z4)/2;

println("fix virtual indices parity")
@tensor t[:]:=OO[1,-2,2,-4]*P1even[-1,1]*P3even[-3,2];
println(norm(t))
@tensor t[:]:=OO[1,-2,2,-4]*P1odd[-1,1]*P3odd[-3,2];
println(norm(t))


@tensor t[:]:=OO[1,-2,2,-4]*P1even[-1,1]*P3odd[-3,2];
println(norm(t))
@tensor t[:]:=OO[1,-2,2,-4]*P1odd[-1,1]*P3even[-3,2];
println(norm(t))


println("fix parity of both physical and virtual indices")
@tensor t[:]:=P2even[-2,2]*P4even[-4,4]*OO[1,2,3,4]*P1even[-1,1]*P3even[-3,3];
println(norm(t))
@tensor t[:]:=P2odd[-2,2]*P4odd[-4,4]*OO[1,2,3,4]*P1even[-1,1]*P3even[-3,3];
println(norm(t))

@tensor t[:]:=P2even[-2,2]*P4even[-4,4]*OO[1,2,3,4]*P1odd[-1,1]*P3odd[-3,3];
println(norm(t))
@tensor t[:]:=P2odd[-2,2]*P4odd[-4,4]*OO[1,2,3,4]*P1odd[-1,1]*P3odd[-3,3];
println(norm(t))


println("4-site Hamiltonian, with only 2 physical indices")
@tensor H4[:]:=OO[2,-1,1,-3]*OO[1,-2,2,-4];
U_fuse_DD_DD=unitary(fuse(space(H4,1)⊗ space(H4,2)),space(H4,1)'⊗ space(H4,2)');
@tensor H4_fused[:]:=H4[1,2,3,4]*U_fuse_DD_DD'[1,2,-1]*U_fuse_DD_DD[-2,3,4];
Z1=gauge_operator(space(H4_fused,1));
Z2=gauge_operator(space(H4_fused,2));


P1even=(Z1+Z1*Z1)/2;
P1odd=(-Z1+Z1*Z1)/2;
P2even=(Z2+Z2*Z2)/2;
P2odd=(-Z2+Z2*Z2)/2;

@tensor t[:]:=H4_fused[1,2]*P1even[-1,1]*P2even[-2,2];
println(norm(t))
@tensor t[:]:=H4_fused[1,2]*P1odd[-1,1]*P2odd[-2,2];
println(norm(t))


println("4-site Hamiltonian, 4 physical indices")
@tensor H4[:]:=OO[2,-1,1,-3]*OO[1,-2,2,-4];
Z1=gauge_operator(space(H4,1));
Z2=gauge_operator(space(H4,2));
Z3=gauge_operator(space(H4,3));
Z4=gauge_operator(space(H4,4));

P1even=(Z1+Z1*Z1)/2;
P1odd=(-Z1+Z1*Z1)/2;
P2even=(Z2+Z2*Z2)/2;
P2odd=(-Z2+Z2*Z2)/2;
P3even=(Z3+Z3*Z3)/2;
P3odd=(-Z3+Z3*Z3)/2;
P4even=(Z4+Z4*Z4)/2;
P4odd=(-Z4+Z4*Z4)/2;

@tensor t[:]:=H4[1,2,3,4]*P1even[-1,1]*P2even[-2,2]*P3even[-3,3]*P4even[-4,4];
println(norm(t))
@tensor t[:]:=H4[1,2,3,4]*P1even[-1,1]*P2even[-2,2]*P3odd[-3,3]*P4odd[-4,4];
println(norm(t))
@tensor t[:]:=H4[1,2,3,4]*P1odd[-1,1]*P2odd[-2,2]*P3even[-3,3]*P4even[-4,4];
println(norm(t))
@tensor t[:]:=H4[1,2,3,4]*P1odd[-1,1]*P2odd[-2,2]*P3odd[-3,3]*P4odd[-4,4];
println(norm(t))
@tensor t[:]:=H4[1,2,3,4]*P1even[-1,1]*P2odd[-2,2]*P3even[-3,3]*P4odd[-4,4];
println(norm(t))
@tensor t[:]:=H4[1,2,3,4]*P1even[-1,1]*P2odd[-2,2]*P3odd[-3,3]*P4even[-4,4];
println(norm(t))
@tensor t[:]:=H4[1,2,3,4]*P1odd[-1,1]*P2even[-2,2]*P3even[-3,3]*P4odd[-4,4];
println(norm(t))
@tensor t[:]:=H4[1,2,3,4]*P1odd[-1,1]*P2even[-2,2]*P3odd[-3,3]*P4even[-4,4];
println(norm(t))