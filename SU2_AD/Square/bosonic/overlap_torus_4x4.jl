using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)



include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\bosonic\\square\\square_AD_2site.jl")
include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")

include("..\\..\\src\\mps_algorithms\\ES_algorithms.jl")
include("..\\..\\src\\mps_algorithms\\parity_funs.jl")


Random.seed!(555)


D=3;
Nv=8;
y_anti_pbc=false;
sector="odd";#"even","odd"



optim_setting=Optim_settings();
optim_setting.init_statenm="SU_D_3.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);


# data=load(optim_setting.init_statenm);
# #A=data["A"];
# A=data["x"];

data=load("didier.jld2");
A=data["A"];

#connect two tensors in y direction
U_y_2=unitary(fuse(space(A,1)*space(A,1)), space(A,1)*space(A,1));
U_y_4=unitary(fuse(space(U_y_2,1)*space(U_y_2,1)), space(U_y_2,1)*space(U_y_2,1));
U_d_2=unitary(fuse(space(A,5)*space(A,5)), space(A,5)*space(A,5));
U_d_4=unitary(fuse(space(U_d_2,1)*space(U_d_2,1)), space(U_d_2,1)*space(U_d_2,1));

@tensor AA_y[:]:=A[1,3,4,-4,6]*A[2,-2,5,3,7]*U_y_2[-1,1,2]*U_y_2'[4,5,-3]*U_d_2[-5,6,7];

gate_y=parity_gate(A,4);

@tensor AAAA__0[:]:=AA_y[1,3,4,8,6]*AA_y[2,8,5,3,7]*U_y_4[-1,1,2]*U_y_4'[4,5,-3]*U_d_4[-5,6,7];
@tensor AAAA__pi[:]:=gate_y[9,8]*AA_y[1,3,4,8,6]*AA_y[2,9,5,3,7]*U_y_4[-1,1,2]*U_y_4'[4,5,-3]*U_d_4[-5,6,7];

gate_x=parity_gate(AAAA__0,1);
@tensor psi_0_0[:]:=AAAA__0[1,2,-1]*AAAA__0[2,3,-2]*AAAA__0[3,4,-3]*AAAA__0[4,1,-4];
@tensor psi_0_pi[:]:=AAAA__pi[1,2,-1]*AAAA__pi[2,3,-2]*AAAA__pi[3,4,-3]*AAAA__pi[4,1,-4];
@tensor psi_pi_0[:]:=gate_x[5,1]*AAAA__0[1,2,-1]*AAAA__0[2,3,-2]*AAAA__0[3,4,-3]*AAAA__0[4,5,-4];
@tensor psi_pi_pi[:]:=gate_x[5,1]*AAAA__pi[1,2,-1]*AAAA__pi[2,3,-2]*AAAA__pi[3,4,-3]*AAAA__pi[4,5,-4];


psi_0_0=psi_0_0/norm(psi_0_0);
psi_0_pi=psi_0_pi/norm(psi_0_pi);
psi_pi_0=psi_pi_0/norm(psi_pi_0);
psi_pi_pi=psi_pi_pi/norm(psi_pi_pi);
psis=(psi_0_0,psi_0_pi,psi_pi_0,psi_pi_pi,);

O=Matrix{ComplexF64}(undef,4,4);
for c1=1:4
    for c2=1:4
        O[c1,c2]=dot(psis[c1],psis[c2]);
    end
end


#AA, U_L,U_D,U_R,U_U=build_double_layer(A,[]);
