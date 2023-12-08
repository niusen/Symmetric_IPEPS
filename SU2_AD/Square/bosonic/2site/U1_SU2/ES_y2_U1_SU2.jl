using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using ChainRulesCore,Zygote
using HDF5, JLD2, MAT
using Random
cd(@__DIR__)



include("..\\..\\..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\..\\..\\src\\bosonic\\square\\square_AD_2site.jl")
include("..\\..\\..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\..\\..\\src\\bosonic\\optimkit_lib.jl")

include("..\\..\\..\\..\\src\\mps_algorithms\\ES_algorithms.jl")
include("..\\..\\..\\..\\src\\mps_algorithms\\parity_funs.jl")


Random.seed!(555)


D=16;
Nv=2;
y_anti_pbc=false;


optim_setting=Optim_settings();
optim_setting.init_statenm="parton_M2.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);


data=load(optim_setting.init_statenm);
A=data["A"];







U_AA=unitary(fuse(space(A,1)*space(A,1)), space(A,1)*space(A,1));

@tensor A_A[:]:=U_AA[-1,1,2]*A[1,3,4,6,-3]*A[2,6,5,3,-4]*U_AA'[4,5,-2];

gate=parity_gate(A,4);
@tensor A_A_gate[:]:=U_AA[-1,1,2]*A[1,3,4,6,-3]*A[2,7,5,3,-4]*U_AA'[4,5,-2]*gate[7,6];





function M_vr(A_A,A_A_gate,vr0,y_anti_pbc)
    vr=deepcopy(vr0);
    if y_anti_pbc
        
        @tensor vr[:]:=A_A_gate'[-1,1,3,4]*A_A_gate[-2,2,3,4]*vr[1,2,-3];
    else
        @tensor vr[:]:=A_A'[-1,1,3,4]*A_A[-2,2,3,4]*vr[1,2,-3];
    end
    return vr;
end

function vl_M(A_A,A_A_gate,vl0,y_anti_pbc)
    vl=deepcopy(vl0);
    if y_anti_pbc
        gate=parity_gate(A_A,4);
        @tensor vl[:]:=A_A_gate'[1,-2,3,4]*A_A_gate[2,-3,3,4]*vl[-1,1,2];
    else
        @tensor vl[:]:=A_A'[1,-2,3,4]*A_A[2,-3,3,4]*vl[-1,1,2];
    end
    return vl;
end

v_init=TensorMap(randn, space(A_A,1)'*space(A_A,1),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1));
v_init=permute(v_init,(1,2,3,),());#L1,L2,L3,L4,dummy
contraction_fun_R(x)=M_vr(A_A,A_A_gate,x,y_anti_pbc);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 6,:LM,Arnoldi(krylovdim=20));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,L4,dummy

println(eur)


v_init=TensorMap(randn, space(A_A,2)'*space(A_A,2),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1)');
v_init=permute(v_init,(3,1,2,),());#dummy,R1,R2,R3,R4
contraction_fun_L(x)=vl_M(A_A,A_A_gate,x,y_anti_pbc);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 6,:LM,Arnoldi(krylovdim=20));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3,R4

println(eul)



fixpoint_ind=1;

VR=evr[fixpoint_ind];#L1,L2,L3,L4,dummy
VL=evl[fixpoint_ind];#dummy,R1,R2,R3,R4
# VR=evr[2];#L1,L2,L3,L4,dummy
# VL=evl[2];#dummy,R1,R2,R3,R4

@tensor VL[:]:=VL[-1,1,2]*U_AA'[-2,-3,1]*U_AA[2,-4,-5];


@tensor VR[:]:=VR[1,2,-5]*U_AA[1,-1,-2]*U_AA'[-3,-4,2];

@tensor H[:]:=VL[1,-1,-2,2,3]*VR[-3,-4,2,3,1];


Es,ev=eig(H,(1,2,),(3,4,))


ev=permute(ev,(1,2,3,));
ev_translation=permute(ev,(2,1,3,));#L2',L3',L4',L1',dummy
if y_anti_pbc
    gate=parity_gate(ev,1);
    @tensor K_phase[:]:=gate[3,2]*ev_translation'[1,3,-1]*ev[1,2,-2];
else
    @tensor K_phase[:]:=ev_translation'[1,2,-1]*ev[1,2,-2];
end
K_phase=permute(K_phase,(1,),(2,));


Es=Es/norm(Es);


Spin=Vector{Float64}(undef,0);
k_phase=Vector{ComplexF64}(undef,0);
eu=Vector{ComplexF64}(undef,0);
Qn=Vector{Float64}(undef,0);

V=space(Es,1);
for cc in eachindex(V.dims.keys)
    eu=vcat(eu,diag(Es.data.values[cc]));
    k_phase=vcat(k_phase,diag(K_phase.data.values[cc]));
    Spin=vcat(Spin,ones(length(diag(Es.data.values[cc])))*(V.dims.keys[cc].sectors[2].j));
    Qn=vcat(Qn,ones(length(diag(Es.data.values[cc])))*(V.dims.keys[cc].sectors[1].charge));
end


order=sortperm(abs.(eu));


eu=eu[order];
Spin=Spin[order];
k_phase=k_phase[order];
Qn=Qn[order];

if y_anti_pbc
    matwrite("ES_Nv2_APBC"*".mat", Dict(
        "k_phase" => k_phase,
        "Spin" => Spin,
        "eu" => eu,
        "Qn" => Qn
    ); compress = false)
else
    matwrite("ES_Nv2_PBC"*".mat", Dict(
        "k_phase" => k_phase,
        "Spin" => Spin,
        "eu" => eu,
        "Qn" => Qn
    ); compress = false)
end
