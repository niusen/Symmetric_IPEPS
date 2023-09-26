using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
using MAT
cd(@__DIR__)


# Va=SU2Space(0=>length(findall(x->x==0, S_label))/1, 1/2=>length(findall(x->x==1/2, S_label))/2,1=>length(findall(x->x==1, S_label))/3,3/2=>length(findall(x->x==3/2, S_label))/4,2=>length(findall(x->x==2, S_label))/5,5/2=>length(findall(x->x==5/2, S_label))/6)
# Vb=SU2Space(1=>1)
# t2 = TensorMap(T, Va ⊗ Va ← Vb);

filenm="tensor"*".mat"
vars = matread(filenm)
tensor=vars["tensor"];

Va=SU2Space(1=>1);
Vb=SU2Space(3=>1);
tensor = TensorMap(tensor, Vb ← Va*Va*Va*Va*Va*Va );


U_phy=unitary(fuse(Va*Va)',Va'*Va');
@tensor tensor_fused[:]:=tensor[-1,1,2,3,4,5,6]*U_phy[-2,1,2]*U_phy[-3,3,4]*U_phy[-4,5,6];
tensor_fused=permute(tensor_fused,(1,),(2,3,4,))
