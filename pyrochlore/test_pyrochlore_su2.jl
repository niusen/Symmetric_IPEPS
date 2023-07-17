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



Random.seed!(1234)


D=2;





Bond_irrep="A";
Tetrahedral_irrep="E";

init_statenm=nothing;
init_noise=0;
json_state_dict, Bond_A_coe, Tetrahedral_E_coe=initial_state(Bond_irrep,Tetrahedral_irrep,D,init_statenm,init_noise)
Tetrahedral_E_coe[1]=1;
Tetrahedral_E_coe[2]=0;
json_state_dict=set_vector(json_state_dict, [1,1,0]);

A_set,E_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
bond_tensor,tetrahedral_tensor=construct_su2_PG_IPESS(json_state_dict,A_set,E_set, S_label, Sz_label, virtual_particle, Va, Vb);


Id=I(2);
sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
@tensor H12[:]:=sx[-1,-4]*sx[-2,-5]*Id[-3,-6]+sy[-1,-4]*sy[-2,-5]*Id[-3,-6]+sz[-1,-4]*sz[-2,-5]*Id[-3,-6];
@tensor H31[:]:=sx[-1,-4]*Id[-2,-5]*sx[-3,-6]+sy[-1,-4]*Id[-2,-5]*sy[-3,-6]+sz[-1,-4]*Id[-2,-5]*sz[-3,-6];
@tensor H23[:]:=Id[-1,-4]*sx[-2,-5]*sx[-3,-6]+Id[-1,-4]*sy[-2,-5]*sy[-3,-6]+Id[-1,-4]*sz[-2,-5]*sz[-3,-6];
@tensor H123chiral[:]:=sx[-1,-4]*sy[-2,-5]*sz[-3,-6]-sx[-1,-4]*sz[-2,-5]*sy[-3,-6]+sy[-1,-4]*sz[-2,-5]*sx[-3,-6]-sy[-1,-4]*sx[-2,-5]*sz[-3,-6]+sz[-1,-4]*sx[-2,-5]*sy[-3,-6]-sz[-1,-4]*sy[-2,-5]*sx[-3,-6];
H123chiral_TensorKit=TensorMap(H123chiral, Va ⊗Va ⊗Va ← Va ⊗Va ⊗Va);

@tensor tetrahedral_ov[:] := tetrahedral_tensor'[1,2,3,4]*tetrahedral_tensor[1,2,3,4];

@tensor chirality123[:] := tetrahedral_tensor'[1,2,3,7]*H123chiral_TensorKit[4,5,6,1,2,3]*tetrahedral_tensor[4,5,6,7];
@tensor chirality243[:] := tetrahedral_tensor'[7,1,3,2]*H123chiral_TensorKit[4,5,6,1,2,3]*tetrahedral_tensor[7,4,6,5];
@tensor chirality341[:] := tetrahedral_tensor'[3,7,1,2]*H123chiral_TensorKit[4,5,6,1,2,3]*tetrahedral_tensor[6,7,4,5,];
@tensor chirality421[:] := tetrahedral_tensor'[3,2,7,1]*H123chiral_TensorKit[4,5,6,1,2,3]*tetrahedral_tensor[6,5,7,4];

chirality123=chirality123/tetrahedral_ov;
chirality243=chirality243/tetrahedral_ov;
chirality341=chirality341/tetrahedral_ov;
chirality421=chirality421/tetrahedral_ov;

@tensor PEPS_part1[w,s,d,m,p1,p2,p3,p4] := bond_tensor[w,ww,p1]*bond_tensor[d,dd,p2]*bond_tensor[s,ss,p3]*bond_tensor[m,mm,p4]*tetrahedral_tensor[ww,dd,ss,mm];

U_phy=unitary(fuse(space(PEPS_part1, 5) ⊗ space(PEPS_part1, 6) ⊗ space(PEPS_part1, 7)⊗ space(PEPS_part1, 8)), space(PEPS_part1, 5) ⊗ space(PEPS_part1, 6) ⊗ space(PEPS_part1, 7)⊗ space(PEPS_part1, 8));
@tensor PEPS_part1[w,s,d,m,p]:=PEPS_part1[w,s,d,m,p1,p2,p3,p4]*U_phy[p,p1,p2,p3,p4];
@tensor PEPS_tensor[w,s,e,n,u,d,p]:=PEPS_part1[w,s,d,m,p]*tetrahedral_tensor[u,e,n,m];

U_W=unitary(fuse(space(tetrahedral_tensor, 1) ⊗ space(tetrahedral_tensor, 1)'), space(tetrahedral_tensor, 1) ⊗ space(tetrahedral_tensor, 1)');
U_E=inv(U_W);
U_S=U_W;
U_N=inv(U_S);
U_D=U_W;
U_U=inv(U_D);
U_M=U_W;
@tensor double_PEPS_part1[w,s,d,m]:=PEPS_part1'[w1,s1,d1,m1,p]*PEPS_part1[w2,s2,d2,m2,p]*U_W[w,w1,w2]*U_S[s,s1,s2]*U_D[d,d1,d2]*U_M[m,m1,m2];
@tensor double_PEPS_part2[u,e,n,m]:=tetrahedral_tensor'[u1,e1,n1,m1]*tetrahedral_tensor[u2,e2,n2,m2]*U_U[u1,u2,u]*U_E[e1,e2,e]*U_N[n1,n2,n]*U_M'[m1,m2,m];
@tensor double_PEPS[w,s,e,n,u,d]:=double_PEPS_part1[w,s,d,m]*double_PEPS_part2[u,e,n,m];


# 2X2X2 cluster
@tensor layer_2D[u1,u2,u3,u4,d1,d2,d3,d4]:=double_PEPS[2,5,1,6,u1,d1]*double_PEPS[1,7,2,8,u2,d2]*double_PEPS[4,6,3,5,u3,d3]*double_PEPS[3,8,4,7,u4,d4];
println(varinfo(r"layer_2D"))
@tensor Norm_3D[:]:=layer_2D[5,6,7,8,1,2,3,4]*layer_2D[1,2,3,4,5,6,7,8];






