using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)



include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("Projector_funs.jl")
include("double_layer_funs.jl")

optim_setting=Optim_settings();
optim_setting.init_statenm="Optim_cell_LS_D_2_chi_20_1.081059.jld2";#"SimpleUpdate_D_6.jld2";#"nothing";
optim_setting.init_noise=0;
optim_setting.linesearch_CTM_method="from_converged_CTM"; # "restart" or "from_converged_CTM"
dump(optim_setting);


# data=load(optim_setting.init_statenm);
# #A=data["A"];
# A=data["x"];

data=load(optim_setting.init_statenm);
x=data["x"];
A_a=x[1].T;
A_b=x[2].T;

#######################
#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D
println("need to include swap gate between A_a and A_b if you want to combine them")
#######################

#original order:  L1,U1,P1,R1,D1, L2,U2,P2,R2,D2,
U_AB=unitary(fuse(space(A_a,4)*space(A_b,4)), space(A_a,4)*space(A_b,4));
U_p=unitary(fuse(space(A_a,5)*space(A_b,5)), space(A_a,5)*space(A_b,5));
A_a=permute(A_a,(1,4,5,3,2,));
A_b=permute(A_b,(1,4,5,3,2,));



A_a=permute_neighbour_ind(A_a,4,5,5);# L1,U1,P1,D1,R1, L2,U2,P2,R2,D2,
@tensor A[:]:=A_a[-1,-2,-3,-4,1]*A_b[1,-5,-6,-7,-8]; # L1,U1,P1,D1,U2,P2,R2,D2,
A=permute_neighbour_ind(A,4,5,8);# L1,U1,P1,U2,D1,P2,R2,D2,
A=permute_neighbour_ind(A,3,4,8);# L1,U1,U2,P1,D1,P2,R2,D2,
A=permute_neighbour_ind(A,7,8,8);# L1,U1,U2,P1,D1,P2,D2,R2,
A=permute_neighbour_ind(A,6,7,8);# L1,U1,U2,P1,D1,D2,P2,R2,
A=permute_neighbour_ind(A,5,6,8);# L1,U1,U2,P1,D2,D1,P2,R2,
#be careful about order: U1,U2, D2, D1
@tensor A[:]:=A[-1,1,2,-3,3,4,-5,-6]*U_AB[-2,1,2]*U_AB'[4,3,-4];# L1,U,P1,D,P2,R2,
A=permute_neighbour_ind(A,5,6,6);# L1,U,P1,D,R2,P2
A=permute_neighbour_ind(A,3,4,6);# L1,U,D,P1,R2,P2
A=permute_neighbour_ind(A,4,5,6);# L1,U,D,R2,P1,P2
@tensor A[:]:=A[-1,-2,-3,-4,1,2]*U_p[-5,1,2];# L1,U,D,R2,P
A=permute_neighbour_ind(A,4,5,5);# L1,U,D,P,R2
A=permute_neighbour_ind(A,3,4,5);# L1,U,P,D,R2
A=permute_neighbour_ind(A,4,5,5);# L1,U,P,R2,D1

A=permute(A,(1,5,4,2,3,));









A_origin=deepcopy(A);




y_anti_pbc=false;
boundary_phase_y=0.5;

# if y_anti_pbc
#     gauge_gate1=gauge_gate(A,2,2*pi/4*boundary_phase_y);
#     @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
# end

#############################
# #convert to the order of PEPS code
A1=deepcopy(A);
A2=deepcopy(A);
A3=deepcopy(A);
A4=deepcopy(A);
#############################

println(space(A1,4))
V_odd,V_even=projector_virtual(space(A1,4))
#physical state only has even parity


l_Vodd=length(V_odd);
l_Veven=length(V_even);

A1=A1/norm(A1);
A2=A2/norm(A2);
A3=A3/norm(A3);
A4=A4/norm(A4);

A1_Vodd=Vector(undef,l_Vodd);
A1_Veven=Vector(undef,l_Veven);
A4_Vodd=Vector(undef,l_Vodd);
A4_Veven=Vector(undef,l_Veven);




for cc1=1:l_Vodd
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc1][-4,1];
        A1_Vodd[cc1]=A_temp;

        @tensor A_temp[:]:=A4[-1,1,-3,-4,-5]*V_odd[cc1]'[1,-2];
        A4_Vodd[cc1]=A_temp;
end

for cc1=1:l_Veven
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc1][-4,1];
        A1_Veven[cc1]=A_temp;

        @tensor A_temp[:]:=A4[-1,1,-3,-4,-5]*V_even[cc1]'[1,-2];
        A4_Veven[cc1]=A_temp;
end



gate_L=parity_gate(A2,1); 

U_DD=unitary(fuse(space(A2,1)⊗space(A2,1)), space(A2,1)⊗space(A2,1));
U_PP=unitary(fuse(space(A2,5)⊗space(A2,5)), space(A2,5)⊗space(A2,5));
U_PPPP=unitary(fuse(space(U_PP,1)⊗space(U_PP,1)), space(U_PP,1)⊗space(U_PP,1));

PPPP_projector=projector_general(space(U_PPPP,1));
l_P=length(PPPP_projector);
AAAA_set=Vector(undef,l_P);

for cp=1:l_P
    println("cp="*string(cp));flush(stdout);
    P_projector=PPPP_projector[cp];
    @tensor P_projector[:]:=P_projector[-1,1]*U_PPPP[1,-2,-3];
    W=TensorMap(randn,space(U_DD,1)⊗space(U_DD,1)⊗space(U_DD,1)'⊗space(U_DD,1)', space(P_projector,1)');
    W=permute(W,(1,2,5,3,4,));
    W=W*0*im;
    for cv=1:length(A1_Vodd)
        A1_temp=deepcopy(A1_Vodd[cv]);
        A1_temp=-A1_temp;#parity gate of U1
        A2_temp=deepcopy(A2);
        A3_temp=deepcopy(A3);
        A4_temp=deepcopy(A4_Vodd[cv]);
        @tensor A1_temp[:]:=A1_temp[1,-2,-3,-4,-5]*gate_L[-1,1];
        @tensor A2_temp[:]:=A2_temp[1,-2,-3,-4,-5]*gate_L[-1,1];
        @tensor A3_temp[:]:=A3_temp[1,-2,-3,-4,-5]*gate_L[-1,1];
        @tensor A4_temp[:]:=A4_temp[1,-2,-3,-4,-5]*gate_L[-1,1];

        @tensor A1A2[:]:=A1_temp[1,3,4,-4,6]*A2_temp[2,-2,5,3,7]*U_DD[-1,1,2]*U_DD'[4,5,-3]*U_PP[-5,6,7];
        @tensor A3A4[:]:=A3_temp[1,3,4,-4,6]*A4_temp[2,-2,5,3,7]*U_DD[-1,1,2]*U_DD'[4,5,-3]*U_PP[-5,6,7];
        @tensor A1A2A3A4[:]:=A1A2[-1,2,-4,3,1]*A3A4[-2,3,-5,2,4]*P_projector[-3,1,4];
        W=W+A1A2A3A4;



    end

    for cv=1:length(A1_Veven)
        A1_temp=deepcopy(A1_Veven[cv]);
        A2_temp=deepcopy(A2);
        A3_temp=deepcopy(A3);
        A4_temp=deepcopy(A4_Veven[cv]);

        @tensor A1A2[:]:=A1_temp[1,3,4,-4,6]*A2_temp[2,-2,5,3,7]*U_DD[-1,1,2]*U_DD'[4,5,-3]*U_PP[-5,6,7];
        @tensor A3A4[:]:=A3_temp[1,3,4,-4,6]*A4_temp[2,-2,5,3,7]*U_DD[-1,1,2]*U_DD'[4,5,-3]*U_PP[-5,6,7];
        @tensor A1A2A3A4[:]:=A1A2[-1,2,-4,3,1]*A3A4[-2,3,-5,2,4]*P_projector[-3,1,4];
        W=W+A1A2A3A4;
    end
    AAAA_set[cp]=W;

end

global AAAA_set, l_P;


function M_vr(vr0)
    vr=deepcopy(vr0)*0;#L1'L2',L3'L4',L1L2,L3L4
    for cp=1:l_P
        @tensor vr_temp[:]:=vr0[3,4,1,2]*AAAA_set[cp]'[-1,-2,5,3,4]*AAAA_set[cp][-3,-4,5,1,2];
        vr=vr+vr_temp;
    end
    println("finished one Mv operation");flush(stdout);
    return vr;
end

function vl_M(vl0)
    vl=deepcopy(vl0)*0;#R1'R2',R3'R4',R1R2,R3R4
    for cp=1:l_P
        @tensor vl_temp[:]:=vl0[3,4,1,2]*AAAA_set[cp]'[3,4,5,-1,-2]*AAAA_set[cp][1,2,5,-3,-4];
        vl=vl+vl_temp;
    end
    println("finished one Mv operation");flush(stdout);
    return vl;
end
v_init=TensorMap(randn, space(U_DD,1)'*space(U_DD,1)'*space(U_DD,1),space(U_DD,1)');
v_init=permute(v_init,(1,2,3,4,),());#L1'L2',L3'L4',L1L2,L3L4
contraction_fun_R(x)=M_vr(x);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 3,:LM,Arnoldi(krylovdim=20));
VR=evr[findmax(abs.(eur))[2]];#L1'L2',L3'L4',L1L2,L3L4

println(eur)


v_init=TensorMap(randn, space(U_DD,1)*space(U_DD,1)*space(U_DD,1)',space(U_DD,1));
v_init=permute(v_init,(1,2,3,4,),());#R1'R2',R3'R4',R1R2,R3R4
contraction_fun_L(x)=vl_M(x);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 3,:LM,Arnoldi(krylovdim=20));
VL=evl[findmax(abs.(eul))[2]];#R1'R2',R3'R4',R1R2,R3R4

println(eul)



##################

@tensor H[:]:=VL[-1,-2,1,2]*VR[-3,-4,1,2];#R1'R2',R3'R4' ,L1'L2',L3'L4'



eu,ev=eig(H,(1,2,),(3,4,));

Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))
@tensor ev[:]:=U_DD'[-1,-2,1]*U_DD'[-3,-4,2]*ev[1,2,-5];
ev=permute(ev,(1,2,3,4,5,));#L1',L2',L3',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,5);#L2',L1',L3',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,5);#L2',L3',L1',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,5);#L2',L3',L4',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,4,-1]*ev[1,2,3,4,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;


order=sortperm(abs.(eu));
if length(order)>500
    order=order[end-500:end];
end
k_phase=diag(k_phase);
eu_set1=eu[order];
k_phase_set1=k_phase[order];
Qn_set1=Qn[order];



####################################
VR=evr[2];
VL=evl[2];

@tensor H[:]:=VL[-1,-2,1,2]*VR[-3,-4,1,2];#R1'R2',R3'R4' ,L1'L2',L3'L4'



eu,ev=eig(H,(1,2,),(3,4,));
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))
@tensor ev[:]:=U_DD'[-1,-2,1]*U_DD'[-3,-4,2]*ev[1,2,-5];
ev=permute(ev,(1,2,3,4,5,));#L1',L2',L3',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,5);#L2',L1',L3',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,5);#L2',L3',L1',L4',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,5);#L2',L3',L4',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,4,-1]*ev[1,2,3,4,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;


order=sortperm(abs.(eu));
if length(order)>500
    order=order[end-500:end];
end
k_phase=diag(k_phase);
eu_set2=eu[order];
k_phase_set2=k_phase[order];
Qn_set2=Qn[order];
Spin_set2=Spin[order]


##########################
eu=vcat(eu_set1,eu_set2);
k_phase=vcat(k_phase_set1,k_phase_set2);
Qn=vcat(Qn_set1,Qn_set2);
Spin=vcat(Spin_set1,Spin_set2);

matwrite("ES_Gutzwiller_M1_Nv4_MPS"*".mat", Dict(
    "k_phase" => k_phase,
    "eu" => eu,
    "Qn"=>Qn,
    "Spin"=>Spin
); compress = false)



