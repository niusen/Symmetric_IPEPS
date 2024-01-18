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
Nv=4;
y_anti_pbc=false;



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







AA, U_L,U_D,U_R,U_U=build_double_layer(A,[]);

AA=AA/norm(AA);







function M_vr(AA,vr0,y_anti_pbc)
    vr=deepcopy(vr0);
    if y_anti_pbc
        gate=parity_gate(AA,4);
        @tensor vr[:]:=AA[-1,2,1,9]*AA[-2,4,3,2]*AA[-3,6,5,4]*AA[-4,8,7,6]*gate[8,9]*vr[1,3,5,7,-5];
    else
        @tensor vr[:]:=AA[-1,2,1,8]*AA[-2,4,3,2]*AA[-3,6,5,4]*AA[-4,8,7,6]*vr[1,3,5,7,-5];
    end
    return vr;
end

function vl_M(AA,vl0,y_anti_pbc)
    vl=deepcopy(vl0);
    if y_anti_pbc
        gate=parity_gate(AA,4);
        @tensor vl[:]:=AA[1,2,-2,9]*AA[3,4,-3,2]*AA[5,6,-4,4]*AA[7,8,-5,6]*gate[8,9]*vl[-1,1,3,5,7];
    else
        @tensor vl[:]:=AA[1,2,-2,8]*AA[3,4,-3,2]*AA[5,6,-4,4]*AA[7,8,-5,6]*vl[-1,1,3,5,7];
    end
    return vl;
end

v_init=TensorMap(randn, space(AA,1)*space(AA,1)*space(AA,1)*space(AA,1),Rep[SU₂](0=>1));
v_init=permute(v_init,(1,2,3,4,5,),());#L1,L2,L3,L4,dummy
contraction_fun_R(x)=M_vr(AA,x,y_anti_pbc);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 6,:LM,Arnoldi(krylovdim=20));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,L4,dummy

println(eur)



v_init=TensorMap(randn, space(AA,3)*space(AA,3)*space(AA,3)*space(AA,3),Rep[SU₂](0=>1)');
v_init=permute(v_init,(5,1,2,3,4,),());#dummy,R1,R2,R3,R4
contraction_fun_L(x)=vl_M(AA,x,y_anti_pbc);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 6,:LM,Arnoldi(krylovdim=20));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3,R4

println(eul)



fixpoint_ind=7;

VR=evr[fixpoint_ind];#L1,L2,L3,L4,dummy
VL=evl[fixpoint_ind];#dummy,R1,R2,R3,R4
# VR=evr[2];#L1,L2,L3,L4,dummy
# VL=evl[2];#dummy,R1,R2,R3,R4



@tensor VR[:]:=VR[1,2,3,4,-9]*U_L'[-1,-5,1]*U_L'[-2,-6,2]*U_L'[-3,-7,3]*U_L'[-4,-8,4];
@tensor VL[:]:=VL[-1,1,2,3,4]*U_L[1,-2,-6]*U_L[2,-3,-7]*U_L[3,-4,-8]*U_L[4,-5,-9];




@tensor H[:]:=VL[1,2,3,4,5,-1,-2,-3,-4]*VR[2,3,4,5,-5,-6,-7,-8,1];


Es,ev=eig(H,(1,2,3,4,),(5,6,7,8,));


ev=permute(ev,(1,2,3,4,5,));
# if sector=="odd"
#     ev_translation=permute(ev,(3,4,5,6,7,8,1,2,9,));
# elseif sector=="even"
    ev_translation=permute(ev,(2,3,4,1,5,));
# end
if y_anti_pbc
    gate=parity_gate(ev,1);
    @tensor K_phase[:]:=gate[5,4]*ev_translation'[1,2,3,5,-1]*ev[1,2,3,4,-2];
else
    @tensor K_phase[:]:=ev_translation'[1,2,3,4,-1]*ev[1,2,3,4,-2];
end
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
k_phase=conj(k_phase[order]);


if y_anti_pbc
    matwrite("ES_Nv4_APBC_"*".mat", Dict(
        "k_phase" => k_phase,
        "Spin" => Spin,
        "eu" => eu,
        "Qn" => Qn
    ); compress = false)
else
    matwrite("ES_Nv4_PBC_"*".mat", Dict(
        "k_phase" => k_phase,
        "Spin" => Spin,
        "eu" => eu,
        "Qn" => Qn
    ); compress = false)
end
