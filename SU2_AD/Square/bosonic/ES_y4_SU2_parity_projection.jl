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







AA, U_L,U_D,U_R,U_U=build_double_layer(A,[]);

AA=AA/norm(AA);

U_2site=unitary(fuse(space(AA,1)*space(AA,1)), space(AA,1)*space(AA,1));

@tensor AAAA[:]:=U_2site[-1,1,2]*AA[1,3,4,-4]*AA[2,-2,5,3]*U_2site'[4,5,-3];



U_A_A=unitary(fuse(space(A,1)*space(A,1)), space(A,1)*space(A,1));

@tensor U_total[:]:=U_A_A[-2,4,6]*U_A_A'[3,5,-1]*U_L'[3,4,1]*U_L'[5,6,2]*U_2site'[1,2,-3];
U_total=permute(U_total,(1,2,),(3,));

gate_upper=parity_gate(U_total,1);
@tensor gate_upper[:]:=U_total'[-1,3,2]*gate_upper[3,1]*U_total[1,2,-2]

gate_lower=parity_gate(U_total,2);
@tensor gate_lower[:]:=U_total'[-1,1,3]*gate_lower[3,2]*U_total[1,2,-2]

function parity_projection_vr(v,sector)
    n0=norm(v);
    @tensor v_lower[:]:=gate_lower[-1,1]*gate_lower[-2,2]*v[1,2,-3];
    @tensor v_upper[:]:=gate_upper[-1,1]*gate_upper[-2,2]*v[1,2,-3];
    @tensor v_upper_lower[:]:=gate_upper[-1,1]*gate_upper[-2,2]*v_lower[1,2,-3];
    if sector=="even"
        v_new=v+v_lower+v_upper+v_upper_lower;
    elseif sector=="odd"
        v_new=v-v_lower-v_upper+v_upper_lower;
    end
    v_new_norm=norm(v_new);
    v_new=v_new*n0/v_new_norm;
    return v_new
end

function parity_projection_vl(v,sector)
    n0=norm(v);
    @tensor v_lower[:]:=gate_lower[1,-2]*gate_lower[2,-3]*v[-1,1,2];
    @tensor v_upper[:]:=gate_upper[1,-2]*gate_upper[2,-3]*v[-1,1,2];
    @tensor v_upper_lower[:]:=gate_upper[1,-2]*gate_upper[2,-3]*v_lower[-1,1,2];
    if sector=="even"
        v_new=v+v_lower+v_upper+v_upper_lower;
    elseif sector=="odd"
        v_new=v-v_lower-v_upper+v_upper_lower;
    end
    v_new_norm=norm(v_new);
    v_new=v_new*n0/v_new_norm;
    return v_new
end


function M_vr(AAAA,vr0,y_anti_pbc)
    vr=deepcopy(vr0);
    if y_anti_pbc
        gate=parity_gate(AAAA,4);
        @tensor vr[:]:=AAAA[-1,2,1,5]*AAAA[-2,4,3,2]*gate[4,5]*vr[1,3,-5];
    else
        @tensor vr[:]:=AAAA[-1,2,1,4]*AAAA[-2,4,3,2]*vr[1,3,-5];
    end
    vr=parity_projection_vr(vr,sector);
    return vr;
end

function vl_M(AAAA,vl0,y_anti_pbc)
    vl=deepcopy(vl0);
    if y_anti_pbc
        gate=parity_gate(AAAA,4);
        @tensor vl[:]:=AAAA[1,2,-2,5]*AAAA[3,4,-3,2]*gate[4,5]*vl[-1,1,3];
    else
        @tensor vl[:]:=AAAA[1,2,-2,4]*AAAA[3,4,-3,2]*vl[-1,1,3];
    end
    vl=parity_projection_vl(vl,sector);
    return vl;
end

v_init=TensorMap(randn, space(AAAA,1)*space(AAAA,1),Rep[SU₂](0=>1));
v_init=permute(v_init,(1,2,3,),());#L1,L2,L3,L4,dummy
v_init=parity_projection_vr(v_init,sector);
contraction_fun_R(x)=M_vr(AAAA,x,y_anti_pbc);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 6,:LM,Arnoldi(krylovdim=20));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,L4,dummy

println(eur)



v_init=TensorMap(randn, space(AAAA,3)*space(AAAA,3),Rep[SU₂](0=>1)');
v_init=permute(v_init,(3,1,2,),());#dummy,R1,R2,R3,R4
v_init=parity_projection_vl(v_init,sector);
contraction_fun_L(x)=vl_M(AAAA,x,y_anti_pbc);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 6,:LM,Arnoldi(krylovdim=20));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3,R4

println(eul)



fixpoint_ind=1;

VR=evr[fixpoint_ind];#L1,L2,L3,L4,dummy
VL=evl[fixpoint_ind];#dummy,R1,R2,R3,R4
# VR=evr[2];#L1,L2,L3,L4,dummy
# VL=evl[2];#dummy,R1,R2,R3,R4



@tensor VR[:]:=VR[1,2,-5]*U_total[-1,-3,1]*U_total[-2,-4,2];
@tensor VL[:]:=VL[-1,1,2]*U_total'[1,-2,-4]*U_total'[2,-3,-5];




@tensor H[:]:=VL[1,2,3,-1,-2]*VR[2,3,-5,-6,1];


Es,ev=eig(H,(1,2,),(3,4,));


ev=permute(ev,(1,2,3,));
@tensor ev[:]:=ev[1,2,-5]*U_A_A[1,-1,-2]*U_A_A[2,-3,-4];
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
    matwrite("ES_Nv4_APBC_"*sector*".mat", Dict(
        "k_phase" => k_phase,
        "Spin" => Spin,
        "eu" => eu,
        "Qn" => Qn
    ); compress = false)
else
    matwrite("ES_Nv4_PBC_"*sector*".mat", Dict(
        "k_phase" => k_phase,
        "Spin" => Spin,
        "eu" => eu,
        "Qn" => Qn
    ); compress = false)
end
