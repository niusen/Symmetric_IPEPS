using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD
using Random
using Combinatorics
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


plaquettes=[1 2 3 4;3 4 5 6;1 2 5 6];

D=2;




Bond_irrep="A";
Square_irrep="A1";
init_statenm="nothing";
init_noise=0;
A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
json_state_dict, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=initial_state(Bond_irrep,Square_irrep,D,init_statenm,init_noise);
bond_tensor,square_tensor_A1=construct_su2_PG_IPESS(json_state_dict,A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb);
square_tensor_A1=square_tensor_A1/norm(square_tensor_A1);

Bond_irrep="A";
Square_irrep="B1";
init_statenm="nothing";
init_noise=0;
A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
json_state_dict, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=initial_state(Bond_irrep,Square_irrep,D,init_statenm,init_noise);
bond_tensor,square_tensor_B1=construct_su2_PG_IPESS(json_state_dict,A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb);
square_tensor_B1=square_tensor_B1/norm(square_tensor_B1);

PEPS_tensor,A_fused1,U_phy=build_PEPS(bond_tensor,square_tensor_A1);
PEPS_tensor,A_fused2,U_phy=build_PEPS(bond_tensor,square_tensor_B1);


A_set=Vector(undef,2);
A_set[1]=A_fused1;
A_set[2]=A_fused2;


Sigma=plaquatte_Heisenberg(J1,J2);
AKLT=plaquatte_AKLT(Sigma);
H=AKLT*(lambda)+(1-lambda)*Sigma;
@tensor H_eff[:]:=H[3,4,6,10,1,2,5,9]*U_phy[-1,1,2]*U_phy[-2,5,7]*U_phy[-3,8,9]*U_phy'[3,4,-4]*U_phy'[6,7,-5]*U_phy'[8,10,-6];


states=Vector(undef,2^3);
Step=1;
global Step
for c1 in [1,2]
    for c2=1:2
        for c3=1:2
                global Step
                @tensor state[:]:=A_set[c1][3,1,2,5,-1]*A_set[c2][2,6,4,1,-2]*A_set[c3][4,3,5,6,-3];

                states[Step]=state;
                Step=Step+1;
        end
    end
end







b = permutations([1,2,3,4,5,6], 6)
Combs=Vector(undef,6*5*4*3*2*1);
step=1;
for item in b

    group1=sort(item[1:2]);
    group2=sort(item[3:4]);
    group3=sort(item[5:6]);
    group=Vector(undef,3);
    group[1]=group1;
    group[2]=group2;
    group[3]=group3;

    inds=sortperm([group1[1],group2[1],group3[1]]);
    group_new=vcat(group[inds[1]],group[inds[2]],group[inds[3]]);
    Combs[step]=group_new;
    step=step+1;
end

Num=Vector(undef,size(Combs,1));
for c1=1:size(Combs,1)
    group=Combs[c1];
    Num[c1]=group[1]*10^5+group[2]*10^4+group[3]*10^3+group[4]*10^2+group[5]*10^1+group[6];
end

Num=unique(Num);
for c1=1:length(Num)
    group=digits(Num[c1]);
    Num[c1]=group[end:-1:1];
end

pos=[];
for c1=1:length(Num)
    group=Num[c1];
    if ((group[1] in plaquettes[1,:])&(group[2] in plaquettes[1,:]))|((group[3] in plaquettes[1,:])&(group[4] in plaquettes[1,:]))|((group[5] in plaquettes[1,:])&(group[6] in plaquettes[1,:]))
        if ((group[1] in plaquettes[2,:])&(group[2] in plaquettes[2,:]))|((group[3] in plaquettes[2,:])&(group[4] in plaquettes[2,:]))|((group[5] in plaquettes[2,:])&(group[6] in plaquettes[2,:]))
            if ((group[1] in plaquettes[3,:])&(group[2] in plaquettes[3,:]))|((group[3] in plaquettes[3,:])&(group[4] in plaquettes[3,:]))|((group[5] in plaquettes[3,:])&(group[6] in plaquettes[3,:]))
                pos=vcat(pos,c1);
            end
        end
    end
end


#construct spin-1 dimer state
Va=SU2Space(1=>1);
Vb=SU2Space(0=>1);
dimer = TensorMap(randn, Vb ← Va*Va );
dimer=dimer/norm(dimer);
dimer=permute(dimer,(1,2,3,))
U=unitary(fuse(space(dimer,1)*space(dimer,2))', space(dimer,1)*space(dimer,2));
@tensor dimer[:]:=U[-1,1,2]*dimer[1,2,-2];
@tensor Dimer[:]:=dimer[-1,-2]*dimer[-3,-4]*dimer[-5,-6];


states2=Vector(undef,length(Num));
for c1=1:length(Num)
    group=Num[c1];
    state=permute(Dimer,(group[1],group[2],group[3],group[4],group[5],group[6]));
    @tensor state[:]:=state[1,2,3,4,5,6]*U_phy[-1,1,2]*U_phy[-2,3,4]*U_phy[-3,5,6];
    states2[c1]=state;
end

Ns=length(states);
ov1=zeros(Ns,Ns)*im;
for c1=1:Ns
    for c2=1:Ns
        ov1[c1,c2]=dot(states[c1],states[c2]);
    end
end

Ns=length(states2);
ov2=zeros(Ns,Ns)*im;
for c1=1:Ns
    for c2=1:Ns
        ov2[c1,c2]=dot(states2[c1],states2[c2]);
    end
end

states_total=vcat(states,states2);

Ns=length(states_total);
ov=zeros(Ns,Ns)*im;
for c1=1:Ns
    for c2=1:Ns
        ov[c1,c2]=dot(states_total[c1],states_total[c2]);
    end
end

# matwrite("D2_subspace_lambda"*string(lambda)*"_"*Size*".mat", Dict(
#     "ov" => ov,
#     "Hm" => Hm
# ); compress = false)