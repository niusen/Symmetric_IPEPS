using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)



include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\..\\..\\src\\bosonic\\CTMRG_unitcell.jl")
include("..\\..\\..\\..\\src\\bosonic\\square\\square_AD_SU2_cell.jl")
include("..\\..\\..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\..\\..\\src\\bosonic\\optimkit_lib.jl")

include("..\\..\\..\\..\\src\\mps_algorithms\\ES_algorithms.jl")


Random.seed!(555)


D=4;
Nv=4;
y_anti_pbc=false;
filenm="Optim_cell_LS_D_4_chi_100.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";

data=load(filenm);
x=data["x"];





global Lx,Ly
Lx=2;
Ly=1;
@assert size(x)==(Lx,Ly)





A1=x[1].T;
A2=x[2].T;



AA1, U_L,U_D,U_R,U_U=build_double_layer(A1,[]);
AA2, U_L,U_D,U_R,U_U=build_double_layer(A2,[]);

AA1=AA1/norm(AA1);
AA2=AA2/norm(AA2);




function M_vr(AA1,AA2,vr0)
    vr=deepcopy(vr0);
    @tensor vr[:]:=AA2[-1,2,1,8]*AA2[-2,4,3,2]*AA2[-3,6,5,4]*AA2[-4,8,7,6]*vr[1,3,5,7,-5];
    @tensor vr[:]:=AA1[-1,2,1,8]*AA1[-2,4,3,2]*AA1[-3,6,5,4]*AA1[-4,8,7,6]*vr[1,3,5,7,-5];
    return vr;
end

function vl_M(AA1,AA2,vl0)
    vl=deepcopy(vl0);
    @tensor vl[:]:=AA1[1,2,-2,8]*AA1[3,4,-3,2]*AA1[5,6,-4,4]*AA1[7,8,-5,6]*vl[-1,1,3,5,7];
    @tensor vl[:]:=AA2[1,2,-2,8]*AA2[3,4,-3,2]*AA2[5,6,-4,4]*AA2[7,8,-5,6]*vl[-1,1,3,5,7];
    return vl;
end

v_init=TensorMap(randn, space(AA1,1)*space(AA1,1)*space(AA1,1)*space(AA1,1),Rep[SU₂](0=>1));
v_init=permute(v_init,(1,2,3,4,5,),());#L1,L2,L3,L4,dummy
contraction_fun_R(x)=M_vr(AA1,AA2,x);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 4,:LM,Arnoldi(krylovdim=20));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,L4,dummy

println(eur)


v_init=TensorMap(randn, space(AA1,3)*space(AA1,3)*space(AA1,3)*space(AA1,3),Rep[SU₂](0=>1)');
v_init=permute(v_init,(5,1,2,3,4,),());#dummy,R1,R2,R3,R4
contraction_fun_L(x)=vl_M(AA1,AA2,x);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 4,:LM,Arnoldi(krylovdim=20));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3,R4

println(eul)



fixpoint_ind=4;

VR=evr[fixpoint_ind];#L1,L2,L3,L4,dummy
VL=evl[fixpoint_ind];#dummy,R1,R2,R3,R4

@tensor VL[:]:=VL[-1,1,2,3,4]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7]*U_L[4,-8,-9];#dummy, R1',R1,R2',R2,R3',R3,R4',R4
VL=permute(VL,(1,2,4,6,8,3,5,7,9,));#dummy, R1',R2',R3',R4', R1,R2,R3,R4


@tensor VR[:]:=VR[1,2,3,4,-9]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4];#L1',L1,L2',L2,L3',L3,L4',L4,dummy
VR=permute(VR,(2,4,6,8,1,3,5,7,9,));#L1,L2,L3,L4,L1',L2',L3',L4',dummy


@tensor H[:]:=VL[1,-1,-2,-3,-4,2,3,4,5]*VR[2,3,4,5,-5,-6,-7,-8,1];#R1',R2',R3',R4' ,L1',L2',L3',L4'


Es,ev=eig(H,(1,2,3,4,),(5,6,7,8,))


ev=permute(ev,(1,2,3,4,5,));#L1',L2',L3',L4',dummy
ev_translation=permute(ev,(2,3,4,1,5,));#L2',L3',L4',L1',dummy

@tensor K_phase[:]:=ev_translation'[1,2,3,4,-1]*ev[1,2,3,4,-2];
K_phase=permute(K_phase,(1,),(2,));


Es=Es/norm(Es);


Spin=Vector{Float64}(undef,0);
k_phase=Vector{ComplexF64}(undef,0);
eu=Vector{ComplexF64}(undef,0);

V=space(Es,1);
for cc in eachindex(V.dims.keys)
    eu=vcat(eu,diag(Es.data.values[cc]));
    k_phase=vcat(k_phase,diag(K_phase.data.values[cc]));
    Spin=vcat(Spin,ones(length(diag(Es.data.values[cc])))*(V.dims.keys[cc].j));
end


order=sortperm(abs.(eu));


eu=eu[order];
Spin=Spin[order];
k_phase=k_phase[order];

if y_anti_pbc
    matwrite("ES_Nv4_APBC"*".mat", Dict(
        "k_phase" => k_phase,
        "Spin" => Spin,
        "eu" => eu,
    ); compress = false)
else
    matwrite("ES_Nv4_PBC"*".mat", Dict(
        "k_phase" => k_phase,
        "Spin" => Spin,
        "eu" => eu,
    ); compress = false)
end
