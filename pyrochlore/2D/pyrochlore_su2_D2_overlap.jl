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
include("square_CTMRG.jl")
include("spin_operator.jl")
include("pyrochlore_model.jl")
include("build_tensor.jl")

Random.seed!(1234)

J1=1;
J2=1;
lambda=1;




D=2;

Size="2x3B";#"2x2" or "2x3A" or "2x3B"


coe=[1,0];
virtual_type="square";
Irrep="A1+iB1";#"A1", "A1+iB1"
PEPS_tensor1,A_fused1,U_phy=build_PEPS(D,coe,virtual_type,Irrep);

coe=[0,im];
virtual_type="square";
Irrep="A1+iB1";#"A1", "A1+iB1"
PEPS_tensor2,A_fused2,U_phy=build_PEPS(D,coe,virtual_type,Irrep);

A_set=Vector(undef,2);
A_set[1]=A_fused1;
A_set[2]=A_fused2;

U_virtual=unitary(fuse(space(A_fused1,1)*space(A_fused1,1)),space(A_fused1,1)*space(A_fused1,1));

Sigma=plaquatte_Heisenberg(J1,J2);
AKLT=plaquatte_AKLT(Sigma);
H=AKLT*(lambda)+(1-lambda)*Sigma;
@tensor H_eff[:]:=H[3,4,6,10,1,2,5,9]*U_phy[-1,1,2]*U_phy[-2,5,7]*U_phy[-3,8,9]*U_phy'[3,4,-4]*U_phy'[6,7,-5]*U_phy'[8,10,-6];

global Step
if Size=="2x2"
    states=Vector(undef,2^4);
    Step=1;
    global Step
    for c1=1:2
        for c2=1:2
            for c3=1:2
                for c4=1:2
                    global Step
                    @tensor T1[:]:=A_set[c1][-1,1,-3,2,-5]*A_set[c2][-2,2,-4,1,-6];
                    @tensor T2[:]:=A_set[c3][-1,1,-3,2,-5]*A_set[c4][-2,2,-4,1,-6];
                    @tensor state[:]:=T1[1,2,3,4,-1,-2]*T2[3,4,1,2,-3,-4];
                    states[Step]=state;
                    Step=Step+1;
                end
            end
        end
    end
elseif Size=="2x3B"
    states=Vector(undef,2^6);
    Step=1;
    global Step
    for c1=1:2
        for c2=1:2
            for c3=1:2
                for c4=1:2
                    for c5=1:2
                        for c6=1:2
                            global Step
                            @tensor T1[:]:=A_set[c1][3,1,5,2,-3]*A_set[c2][4,2,6,1,-4]*U_virtual[-1,3,4]*U_virtual'[5,6,-2];
                            @tensor T2[:]:=A_set[c3][3,1,5,2,-3]*A_set[c4][4,2,6,1,-4]*U_virtual[-1,3,4]*U_virtual'[5,6,-2];
                            @tensor T3[:]:=A_set[c5][3,1,5,2,-3]*A_set[c6][4,2,6,1,-4]*U_virtual[-1,3,4]*U_virtual'[5,6,-2];
                            @tensor state[:]:=T1[3,1,-1,-2]*T2[1,2,-3,-4]*T3[2,3,-5,-6];
                            states[Step]=state;
                            Step=Step+1;
                        end
                    end
                end
            end
        end
    end
elseif Size=="2x3A" #twisted boundary connection used in ED
    states=Vector(undef,2^6);
    Step=1;
    U_phy_phy=unitary(fuse(space(A_set[1],5)*space(A_set[1],5)), space(A_set[1],5)*space(A_set[1],5));
    global Step
    for c1=1:2
        for c2=1:2
            for c3=1:2
                for c4=1:2
                    for c5=1:2
                        for c6=1:2
                            global Step
                            @tensor T1[:]:=A_set[c1][1,2,4,-4,6]*A_set[c2][3,-2,5,2,7]*U_virtual[-1,1,3]*U_virtual'[4,5,-3]*U_phy_phy[-5,6,7];
                            @tensor T2[:]:=A_set[c3][1,2,4,-4,6]*A_set[c4][3,-2,5,2,7]*U_virtual[-1,1,3]*U_virtual'[4,5,-3]*U_phy_phy[-5,6,7];
                            @tensor T3[:]:=A_set[c5][1,2,4,-4,6]*A_set[c6][3,-2,5,2,7]*U_virtual[-1,1,3]*U_virtual'[4,5,-3]*U_phy_phy[-5,6,7];
                            @tensor state[:]:=T1[5,6,1,2,-1]*T2[1,2,3,4,-2]*T3[3,4,5,6,-3];
                            @tensor state[:]:=state[1,2,3]*U_phy_phy'[-1,-2,1]*U_phy_phy'[-3,-4,2]*U_phy_phy'[-5,-6,3];
                            states[Step]=state;
                            Step=Step+1;
                        end
                    end
                end
            end
        end
    end

end

Ns=length(states);
ov=zeros(Ns,Ns)*im;
for c1=1:Ns
    for c2=1:Ns
        ov[c1,c2]=dot(states[c1],states[c2]);
    end
end

Hm=zeros(Ns,Ns)*im;
for c2=1:Ns
    state2=states[c2];
    for c1=1:Ns
        state1=states[c1];

        Hmm=0;
        if Size=="2x2"
            @tensor rho1[:]:=state1[-1,-3,-2,1]*state2'[-4,-6,-5,1];
            @tensor rho2[:]:=state1[-2,1,-1,-3]*state2'[-5,1,-4,-6];
            @tensor rho3[:]:=state1[-3,-1,1,-2]*state2'[-6,-4,1,-5];
            @tensor rho4[:]:=state1[1,-2,-3,-1]*state2'[1,-5,-6,-4];
            Hmm=Hmm+dot(H_eff,rho1)+dot(H_eff,rho2)+dot(H_eff,rho3)+dot(H_eff,rho4);
            Hm[c1,c2]=Hmm;
        elseif Size=="2x3B"
            println([c1,c2])
            @tensor rho1[:]:=state1[-1,-3,-2,1,2,3]*state2'[-4,-6,-5,1,2,3];
            @tensor rho2[:]:=state1[2,3,-1,-3,-2,1]*state2'[2,3,-4,-6,-5,1];
            @tensor rho3[:]:=state1[-2,3,1,2,-1,-3]*state2'[-5,3,1,2,-4,-6];
            @tensor rho4[:]:=state1[-3,-1,1,-2,2,3]*state2'[-6,-4,1,-5,2,3];
            @tensor rho5[:]:=state1[2,3,-3,-1,1,-2]*state2'[2,3,-6,-4,1,-5];
            @tensor rho6[:]:=state1[3,-2,1,2,-3,-1]*state2'[3,-5,1,2,-6,-4];
            Hmm=Hmm+dot(H_eff,rho1)+dot(H_eff,rho2)+dot(H_eff,rho3)+dot(H_eff,rho4)+dot(H_eff,rho5)+dot(H_eff,rho6);
            Hm[c1,c2]=Hmm;
        elseif Size=="2x3A"
            println([c1,c2])
            @tensor rho1[:]:=state1[-1,-3,-2,1,2,3]*state2'[-4,-6,-5,1,2,3];
            @tensor rho2[:]:=state1[2,3,-1,-3,-2,1]*state2'[2,3,-4,-6,-5,1];
            @tensor rho3[:]:=state1[-2,3,1,2,-1,-3]*state2'[-5,3,1,2,-4,-6];
            @tensor rho4[:]:=state1[1,-1,2,-2,-3,3]*state2'[1,-4,2,-5,-6,3];
            @tensor rho5[:]:=state1[-3,3,1,-1,2,-2]*state2'[-6,3,1,-4,2,-5];
            @tensor rho6[:]:=state1[1,-2,-3,2,3,-1]*state2'[1,-5,-6,2,3,-4];
            Hmm=Hmm+dot(H_eff,rho1)+dot(H_eff,rho2)+dot(H_eff,rho3)+dot(H_eff,rho4)+dot(H_eff,rho5)+dot(H_eff,rho6);
            Hm[c1,c2]=Hmm;
        end

        
        
    end
end






matwrite("D2_subspace_lambda"*string(lambda)*"_"*Size*".mat", Dict(
    "ov" => ov,
    "Hm" => Hm
); compress = false)