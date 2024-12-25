using Revise, TensorKit, Zygote
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives
using Dates
using SUNRepresentations
#https://github.com/QuantumKitHub/SUNRepresentations.jl
cd(@__DIR__)


Irrep[SU{3}]((2, 1, 0))

V0=Irrep[SU{4}]((1,0, 0,0))

#SU(4) P_{ij} operator
P_ij=zeros(4,4,4,4);#d1',d2',d1,d2

for ca=1:4
    for cb=1:4
        P_ij[ca,cb,cb,ca]=1;
    end
end

# for ca=1:4
#     for cb=1:4
#         P_ij[ca,cb,ca,cb]=P_ij[ca,cb,ca,cb]-1;
#     end
# end




V1=Rep[SU₂](0=>2, 1/2=>1);
V2=Rep[U₁ × SU₂]((0,0)=>1,(2,0)=>1,(1, 1/2)=>1);
V3=Rep[SU₂ × SU₂]((0,0)=>1,(2,0)=>1,(1, 1/2)=>1);

# Vp=Rep[SU₂ × SU₂]((1/2,1/2)=>1);
Vp=Rep[SU{4}]((1,0, 0,0)=>1)


T2=TensorMap(P_ij,Vp*Vp,Vp*Vp);
u2,s2,v2=tsvd(permute(T2,(1,3,),(2,4,)));


P_ijk=zeros(4,4,4,4,4,4);#d1',d2',d3',d1,d2,d3

for ca=1:4
    for cb=1:4
        for cc=1:4
            P_ijk[ca,cb,cc,cb,cc,ca]=1;
        end
    end
end

T3=TensorMap(P_ijk,Vp*Vp*Vp,Vp*Vp*Vp);
u3,s3,v3=tsvd(permute(T3,(1,4,),(2,3,5,6,)));