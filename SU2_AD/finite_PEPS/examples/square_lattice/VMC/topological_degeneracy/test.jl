using LinearAlgebra:I,diagm,diag
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

V=Rep[U₁](0=>200, 1=>200, 2=>200);
T=TensorMap(randn,V,fuse(V*V));

@time u,s,v=tsvd(T);

@time a,b=rightorth(T);

@time begin
    TT=T*T';
    u1,s1,v1=tsvd(TT);
    1+1
end


a,b=eigh(TT);
norm(b*a*b'-TT)/norm(TT)