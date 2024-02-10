using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)



include("swap_funs.jl")
include("fermi_permute.jl")

include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\Projector_funs.jl")




M=1;
Guztwiller=true;#add projector


data=load("swap_gate_Tensor_M"*string(M)*".jld2");
P_G=data["P_G"];

psi_G=data["psi_G"];   #P1,P2,L,R,D,U
M1=psi_G[1];
M2=psi_G[2];
M3=psi_G[3];
M4=psi_G[4];
M5=psi_G[5];
M6=psi_G[6];

if Guztwiller
    @tensor M1[:]:=M1[-1,-2,1]*P_G[-3,1];
    @tensor M2[:]:=M2[-1,-2,1]*P_G[-3,1];
    SS_op=data["SS_op_S"];
else
    SS_op=data["SS_op_F"];
end


U_phy1=unitary(fuse(space(M1,1)⊗space(M1,3)⊗space(M2,3)), space(M1,1)⊗space(M1,3)⊗space(M2,3));

@tensor A[:]:=M1[4,1,2]*M2[1,-2,3]*U_phy1[-1,4,2,3];
@tensor A[:]:=A[-1,1]*M3[1,-3,-2];
@tensor A[:]:=A[-1,-2,1]*M4[1,-4,-3];
@tensor A[:]:=A[-1,-2,-3,1]*M5[1,-5,-4];
@tensor A[:]:=A[-1,-2,-3,-4,1]*M6[1,-6,-5];

U_phy2=unitary(fuse(space(A,1)⊗space(A,6)), space(A,1)⊗space(A,6));
@tensor A[:]:=A[1,-2,-3,-4,-5,2]*U_phy2[-1,1,2];
# P,L,R,D,U


bond=data["bond_gate"];#dummy, D1, D2 

#Add bond:both parity gate and bond operator
@tensor A[:]:=A[-1,-2,1,2,-5]*bond[-6,-3,1]*bond[-7,-4,2];
U_phy2=unitary(fuse(space(A,1)⊗space(A,6)⊗space(A,7)), space(A,1)⊗space(A,6)⊗space(A,7));
@tensor A[:]:=A[1,-2,-3,-4,-5,2,3]*U_phy2[-1,1,2,3];
#P,L,R,D,U





#swap between spin up and spin down modes, since |L,U,P><D,R|====L,U,P|><|R,D
special_gate=special_parity_gate(A,3);
@tensor A[:]:=A[-1,-2,1,-4,-5]*special_gate[-3,1];
special_gate=special_parity_gate(A,4);
@tensor A[:]:=A[-1,-2,-3,1,-5]*special_gate[-4,1];



gate=swap_gate(A,4,5); @tensor A[:]:=A[-1,-2,-3,1,2]*gate[-4,-5,1,2];           
A=permute(A,(1,2,3,5,4,));#P,L,R,U,D

gate=swap_gate(A,3,4); @tensor A[:]:=A[-1,-2,1,2,-5]*gate[-3,-4,1,2]; 
A=permute(A,(1,2,4,3,5,));#P,L,U,R,D

gate=swap_gate(A,1,2); @tensor A[:]:=A[1,2,-3,-4,-5]*gate[-1,-2,1,2]; 
A=permute(A,(2,1,3,4,5,));#L,P,U,R,D

gate=swap_gate(A,2,3); @tensor A[:]:=A[-1,1,2,-4,-5]*gate[-2,-3,1,2]; 
A=permute(A,(1,3,2,4,5,));#L,U,P,R,D


#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D


A_origin=deepcopy(A);


y_anti_pbc=true;
boundary_phase_y=0.0;

if y_anti_pbc
    gauge_gate1=gauge_gate(A,2,2*pi/3*boundary_phase_y);
    @tensor A[:]:=A[-1,1,-3,-4,-5]*gauge_gate1[-2,1];
end

#############################
# #convert to the order of PEPS code
A=permute(A,(1,5,4,2,3,));
A1=deepcopy(A);
A2=deepcopy(A);
A3=deepcopy(A);
#############################

println(space(A1,4))
V_odd,V_even=projector_virtual(space(A1,4))
#physical state only has even parity


l_Vodd=length(V_odd);
l_Veven=length(V_even);

A1=A1/norm(A1);
A2=A2/norm(A2);
A3=A3/norm(A3);


A1_Vodd=Vector(undef,l_Vodd);
A1_Veven=Vector(undef,l_Veven);
A3_Vodd=Vector(undef,l_Vodd);
A3_Veven=Vector(undef,l_Veven);




for cc1=1:l_Vodd
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc1][-4,1];
        A1_Vodd[cc1]=A_temp;

        @tensor A_temp[:]:=A3[-1,1,-3,-4,-5]*V_odd[cc1]'[1,-2];
        A3_Vodd[cc1]=A_temp;
end

for cc1=1:l_Veven
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc1][-4,1];
        A1_Veven[cc1]=A_temp;

        @tensor A_temp[:]:=A3[-1,1,-3,-4,-5]*V_even[cc1]'[1,-2];
        A3_Veven[cc1]=A_temp;
end



gate_L=parity_gate(A2,1); 

U_DD=unitary(fuse(space(A2,1)⊗space(A2,1)), space(A2,1)⊗space(A2,1));
U_PP=unitary(fuse(space(A2,5)⊗space(A2,5)), space(A2,5)⊗space(A2,5));
U_PPP=unitary(fuse(space(U_PP,1)⊗space(A2,5)), space(U_PP,1)⊗space(A2,5));

PPP_projector=projector_general_SU2_U1(space(U_PPP,1));
l_P=length(PPP_projector);
AAA_set=Vector(undef,l_P);

for cp=1:l_P
    println("cp="*string(cp));flush(stdout);
    P_projector=PPP_projector[cp];
    @tensor P_projector[:]:=P_projector[-1,1]*U_PPP[1,-2,-3];
    W=TensorMap(randn,space(U_DD,1)⊗space(A2,1)⊗space(U_DD,1)'⊗space(A2,1)', space(P_projector,1)');
    W=permute(W,(1,2,5,3,4,));
    W=W*0*im;
    for cv=1:length(A1_Vodd)
        A1_temp=deepcopy(A1_Vodd[cv]);
        A1_temp=-A1_temp;#parity gate of U1
        A2_temp=deepcopy(A2);
        A3_temp=deepcopy(A3_Vodd[cv]);
        @tensor A1_temp[:]:=A1_temp[1,-2,-3,-4,-5]*gate_L[-1,1];
        @tensor A2_temp[:]:=A2_temp[1,-2,-3,-4,-5]*gate_L[-1,1];
        @tensor A3_temp[:]:=A3_temp[1,-2,-3,-4,-5]*gate_L[-1,1];

        @tensor A1A2[:]:=A1_temp[1,3,4,-4,6]*A2_temp[2,-2,5,3,7]*U_DD[-1,1,2]*U_DD'[4,5,-3]*U_PP[-5,6,7];
        @tensor A1A2A3[:]:=A1A2[-1,2,-4,3,1]*A3_temp[-2,3,-5,2,4]*P_projector[-3,1,4];
        W=W+A1A2A3;



    end

    for cv=1:length(A1_Veven)
        A1_temp=deepcopy(A1_Veven[cv]);
        A2_temp=deepcopy(A2);
        A3_temp=deepcopy(A3_Veven[cv]);

        @tensor A1A2[:]:=A1_temp[1,3,4,-4,6]*A2_temp[2,-2,5,3,7]*U_DD[-1,1,2]*U_DD'[4,5,-3]*U_PP[-5,6,7];
        @tensor A1A2A3[:]:=A1A2[-1,2,-4,3,1]*A3_temp[-2,3,-5,2,4]*P_projector[-3,1,4];
        W=W+A1A2A3;
    end
    AAA_set[cp]=W;

end

global AAA_set, l_P;


function M_vr(vr0)
    vr=deepcopy(vr0)*0;#L1'L2',L3',L1L2,L3
    for cp=1:l_P
        @tensor vr_temp[:]:=vr0[3,4,1,2]*AAA_set[cp]'[-1,-2,5,3,4]*AAA_set[cp][-3,-4,5,1,2];
        vr=vr+vr_temp;
    end
    println("finished one Mv operation");flush(stdout);
    return vr;
end

function vl_M(vl0)
    vl=deepcopy(vl0)*0;#R1'R2',R3',R1R2,R3
    for cp=1:l_P
        @tensor vl_temp[:]:=vl0[3,4,1,2]*AAA_set[cp]'[3,4,5,-1,-2]*AAA_set[cp][1,2,5,-3,-4];
        vl=vl+vl_temp;
    end
    println("finished one Mv operation");flush(stdout);
    return vl;
end
v_init=TensorMap(randn, space(U_DD,1)'*space(A2,1)'*space(U_DD,1),space(A2,1)');
v_init=permute(v_init,(1,2,3,4,),());#L1'L2',L3',L1L2,L3
contraction_fun_R(x)=M_vr(x);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 3,:LM,Arnoldi(krylovdim=20));
VR=evr[findmax(abs.(eur))[2]];#L1'L2',L3',L1L2,L3

println(eur)


v_init=TensorMap(randn, space(U_DD,1)*space(A2,1)*space(U_DD,1)',space(A2,1));
v_init=permute(v_init,(1,2,3,4,),());#R1'R2',R3',R1R2,R3
contraction_fun_L(x)=vl_M(x);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 3,:LM,Arnoldi(krylovdim=20));
VL=evl[findmax(abs.(eul))[2]];#R1'R2',R3',R1R2,R3

println(eul)



##################

@tensor H[:]:=VL[-1,-2,1,2]*VR[-3,-4,1,2];#R1'R2',R3' ,L1'L2',L3'



eu,ev=eig(H,(1,2,),(3,4,));
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))
@tensor ev[:]:=U_DD'[-1,-2,1]*ev[1,-3,-4];
ev=permute(ev,(1,2,3,4,));#L1',L2',L3',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,4);#L2',L1',L3',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,4);#L2',L3',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,-1]*ev[1,2,3,-2];
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
Spin_set1=Spin[order]


####################################
VR=evr[2];
VL=evl[2];

@tensor H[:]:=VL[-1,-2,1,2]*VR[-3,-4,1,2];#R1'R2',R3' ,L1'L2',L3'



eu,ev=eig(H,(1,2,),(3,4,));
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))
@tensor ev[:]:=U_DD'[-1,-2,1]*ev[1,-3,-4];
ev=permute(ev,(1,2,3,4,));#L1',L2',L3',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,4);#L2',L1',L3',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,4);#L2',L3',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,-1]*ev[1,2,3,-2];
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

matwrite("ES_Gutzwiller_M1_Nv3_MPS"*".mat", Dict(
    "k_phase" => k_phase,
    "eu" => eu,
    "Qn"=>Qn,
    "Spin"=>Spin
); compress = false)


