
using LinearAlgebra:I,diagm,diag
using TensorKit
using Random
using Printf
using DelimitedFiles
using CSV
using DataFrames
using JLD2
using Statistics
using MAT

cd(@__DIR__)


include("../../../../state/iPEPS_ansatz.jl")
include("../../../../setting/Settings.jl")
include("../../../../setting/linearalgebra.jl")
include("../../../../setting/tuple_methods.jl")
include("../../../../environment/MC/finite_clusters.jl")

include("../../../../environment/MC/contract_torus.jl")
include("../../../../environment/MC/sampling.jl")
include("../../../../environment/MC/sampling_eliminate_physical_leg.jl")
include("../../../../environment/MC/build_degenerate_states.jl")



D=3;
 
filenm="SimpleUpdate_D_"*string(D);
@show to_dense=false;#convert to dense
psi0,Vp,Vv=load_fPEPS_from_kagome_iPESS(Lx,Ly,filenm,to_dense);
A=psi0[1,1];
A=A/norm(A);
Vp=space(A,5);


sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
@tensor H_Heisenberg[:]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4];
H_Heisenberg=TensorMap(H_Heisenberg,Vp*Vp,  Vp*Vp);



@tensor AA[:]:=A[2,-3,1,-1,-5,-6,-7]*A[1,-4,2,-2,-8,-9,-10];
@tensor AAAA[:]:=AA[1,2,3,4,-1,-2,-3,-4,-5,-6]*AA[3,4,1,2,-7,-8,-9,-10,-11,-12];


function ob_NN(rho,op)
    ob=@tensor rho[1,2,3,4]*op[1,2,3,4];
    Norm=@tensor rho[1,2,1,2];
    return ob/Norm
end

@tensor rho12[:]:=AAAA'[-1,-2,1,2,3,4,5,6,7,8,9,10]*AAAA[-3,-4,1,2,3,4,5,6,7,8,9,10];
@tensor rho13[:]:=AAAA'[-1,1,-2,2,3,4,5,6,7,8,9,10]*AAAA[-3,1,-4,2,3,4,5,6,7,8,9,10];
@tensor rho23[:]:=AAAA'[1,-1,-2,2,3,4,5,6,7,8,9,10]*AAAA[1,-3,-4,2,3,4,5,6,7,8,9,10];
@tensor rho14[:]:=AAAA'[-1,1,2,-2,3,4,5,6,7,8,9,10]*AAAA[-3,1,2,-4,3,4,5,6,7,8,9,10];
@tensor rho34[:]:=AAAA'[1,2,-1,-2,3,4,5,6,7,8,9,10]*AAAA[1,2,-3,-4,3,4,5,6,7,8,9,10];
@tensor rho38[:]:=AAAA'[1,2,-1,3,4,5,6,-2,7,8,9,10]*AAAA[1,2,-3,3,4,5,6,-4,7,8,9,10];
@tensor rho48[:]:=AAAA'[1,2,3,-1,4,5,6,-2,7,8,9,10]*AAAA[1,2,3,-3,4,5,6,-4,7,8,9,10];

@show ob_NN(rho12,H_Heisenberg)
@show ob_NN(rho13,H_Heisenberg)
@show ob_NN(rho23,H_Heisenberg)
@show ob_NN(rho14,H_Heisenberg)
@show ob_NN(rho34,H_Heisenberg)
@show ob_NN(rho38,H_Heisenberg)
@show ob_NN(rho48,H_Heisenberg)


AAAA_dense=convert(Array,AAAA);
AAAA_dense[1,2,2,1,2,1,1,2,1,2,2,1]
AAAA_dense[1,2,1,2,1,2,2,1,2,2,1,1]


A_dense=convert(Array,A);
A_dense[:,:,:,:,1,1,1]

dense1b=A_dense[:,:,:,:,1,2,1]
dense2b=A_dense[:,:,:,:,2,1,2]
dense3b=A_dense[:,:,:,:,2,1,2]
dense4b=A_dense[:,:,:,:,1,2,1]


@tensor b1234[:]:= dense1b[2,7,1,5]*dense2b[1,8,2,6]*dense3b[4,5,3,7]*dense4b[3,6,4,8];