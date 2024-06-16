using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)

include("..\\..\\..\\symmetry\\parity_funs.jl")
include("..\\..\\..\\environment\\convert_boundary_condition.jl")
include("..\\..\\..\\environment\\mps_methods.jl")
include("..\\..\\..\\environment\\peps_double_layer_methods.jl")
include("..\\..\\..\\environment\\truncations.jl")

"""coordinate
    (1,2),(2,2)
    (1,1),(2,1)
"""

Lx=6;
Ly=6;


data=load("CSL_D3.jld2");
A=data["A"];

gate_up=parity_gate(A,4);
gate_left=parity_gate(A,1);

# Vv=U₁Space(0=>1,1=>1,-1=>1);
# Vp=U₁Space(1=>1,-1=>1);
Vv=ℤ₂Space(0=>1,1=>2);
Vp=ℤ₂Space(1=>2);
A=TensorMap(convert(Array,A),Vv*Vv,Vv*Vv*Vp);

gate_up=TensorMap(convert(Array,gate_up),space(A,4),space(A,4));
gate_left=TensorMap(convert(Array,gate_left),space(A,1),space(A,1));

psi_0_0=Matrix{TensorMap}(undef,Lx,Ly);#PBC-PBC
for cx=1:Lx
    for cy=1:Ly
        psi_0_0[cx,cy]=A;
    end
end

psi_0_pi=deepcopy(psi_0_0);
psi_pi_0=deepcopy(psi_0_0);
psi_pi_pi=deepcopy(psi_0_0);


for cx=1:Lx
    @tensor psi_0_pi[cx,Ly][:]:=psi_0_pi[cx,Ly][-1,-2,-3,1,-5]*gate_up[-4,1];
    @tensor psi_pi_pi[cx,Ly][:]:=psi_pi_pi[cx,Ly][-1,-2,-3,1,-5]*gate_up[-4,1];
end


for cy=1:Ly
    @tensor psi_pi_0[1,cy][:]:=psi_pi_0[1,cy][1,-2,-3,-4,-5]*gate_left[-1,1];
    @tensor psi_pi_pi[1,cy][:]:=psi_pi_pi[1,cy][1,-2,-3,-4,-5]*gate_left[-1,1];
end


function PEPS_sample(psi,Config)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert Lx==size(Config,1)
    @assert Ly==size(Config,2)

    Vp=ℤ₂Space(1=>2);
    Vpp=ℤ₂Space(1=>1);
    Vpa=(TensorMap([1 0],Vpp,Vp'), TensorMap([0 1],Vpp,Vp'),);#for cx= odd
    Vpb=(TensorMap([1 0],Vpp',Vp'), TensorMap([0 1],Vpp',Vp'),);#for cx= even

    psi_sample=deepcopy(psi);
    for cx=1:Lx
        for cy=1:Ly
            if mod(cx+cy,2)==1
                @tensor T[:]:=psi[cx,cy][-1,-2,-3,-4,1]*Vpa[Config[cx,cy]][-5,1];
                psi_sample[cx,cy]=T;
            else
                @tensor T[:]:=psi[cx,cy][-1,-2,-3,-4,1]*Vpb[Config[cx,cy]][-5,1];
                psi_sample[cx,cy]=T;
            end
        end
    end
    return psi_sample
end


function contraction_6x6(psi)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert Lx==6;
    @assert Ly==6;

    T=psi[1,1];
    U3=unitary(fuse(space(A,1)*space(A,1)*space(A,1)), space(A,1)*space(A,1)*space(A,1));
    U2=unitary(fuse(space(A,2)*space(A,4)), space(A,2)*space(A,4));

    cx=1;
    @tensor A1_bot[:]:=psi[cx,3][6,5,7,11,-4]*psi[cx,2][4,2,8,5,1]*psi[cx,1][3,10,9,2,1]*U3[-1,6,4,3]*U3'[7,8,9,-2]*U2[-3,10,11];
    @tensor A1_top[:]:=psi[cx,6][6,5,7,11,-4]*psi[cx,5][4,2,8,5,1]*psi[cx,4][3,10,9,2,1]*U3[-1,6,4,3]*U3'[7,8,9,-2]*U2'[11,10,-3];

    @tensor Transop[:]:=A1_top[-1,-3,1,2]*A1_bot[-2,-4,1,2];
    for cx=2:Lx-1
        @tensor A1_bot[:]:=psi[cx,3][6,5,7,11,-4]*psi[cx,2][4,2,8,5,1]*psi[cx,1][3,10,9,2,1]*U3[-1,6,4,3]*U3'[7,8,9,-2]*U2[-3,10,11];
        @tensor A1_top[:]:=psi[cx,6][6,5,7,11,-4]*psi[cx,5][4,2,8,5,1]*psi[cx,4][3,10,9,2,1]*U3[-1,6,4,3]*U3'[7,8,9,-2]*U2'[11,10,-3];
        @tensor Transop[:]:=Transop[-1,-2,1,2]*A1_top[1,-3,3,4]*A1_bot[2,-4,3,4];
    end

    cx=Lx;
    @tensor A1_bot[:]:=psi[cx,3][6,5,7,11,-4]*psi[cx,2][4,2,8,5,1]*psi[cx,1][3,10,9,2,1]*U3[-1,6,4,3]*U3'[7,8,9,-2]*U2[-3,10,11];
    @tensor A1_top[:]:=psi[cx,6][6,5,7,11,-4]*psi[cx,5][4,2,8,5,1]*psi[cx,4][3,10,9,2,1]*U3[-1,6,4,3]*U3'[7,8,9,-2]*U2'[11,10,-3];

    Transop=@tensor Transop[2,4,1,3]*A1_top[1,2,5,6]*A1_bot[3,4,5,6];

    return Transop
end

Config=Int.(round.(rand(Lx,Ly)).+1);

psi_sample=PEPS_sample(psi_0_0,Config);

@time ov_=contraction_6x6(psi_sample)



