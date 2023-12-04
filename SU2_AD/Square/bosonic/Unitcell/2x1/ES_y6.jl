using LinearAlgebra
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)



include("swap_funs.jl")
include("fermi_permute.jl")
include("double_layer_funs.jl")
include("D:\\My Documents\\Code\\Julia_codes\\Tensor network\\GfPEPS_parton\\Projector_funs.jl")


filenm="Optim_LS_D_4_chi_130.jld2";

data=load(filenm);
A=data["A"];



y_anti_pbc=false;
boundary_phase_y=0.5;

if y_anti_pbc
    gauge_gate1=gauge_gate(A,4,2*pi/6*boundary_phase_y);
    @tensor A[:]:=A[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
end

#############################

A1=deepcopy(A);
A2=deepcopy(A);
A3=deepcopy(A);
A4=deepcopy(A);
A5=deepcopy(A);
A6=deepcopy(A);
#############################

println(space(A1,4))
V_odd,V_even=projector_virtual(space(A1,4))
#physical state only has even parity


l_Vodd=length(V_odd);
l_Veven=length(V_even);


AA1_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
AA1_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
AA1_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
AA1_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer

AA6_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
AA6_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
AA6_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
AA6_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer



for cc1=1:l_Vodd
    for cc2=1:l_Vodd
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_odd[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Vodd_Vodd[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,-5]*V_odd[cc1][-2,1];
        @tensor A_temp[:]:=A6[-1,1,-3,-4,-5]*V_odd[cc2]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA6_Vodd_Vodd[cc1,cc2]=AA_temp;

    end
end

for cc1=1:l_Vodd
    for cc2=1:l_Veven
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_odd[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Vodd_Veven[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,-5]*V_odd[cc1][-2,1];
        @tensor A_temp[:]:=A6[-1,1,-3,-4,-5]*V_even[cc2]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA6_Vodd_Veven[cc1,cc2]=AA_temp;

    end
end

for cc1=1:l_Veven
    for cc2=1:l_Vodd
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_even[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Veven_Vodd[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,-5]*V_even[cc1][-2,1];
        @tensor A_temp[:]:=A6[-1,1,-3,-4,-5]*V_odd[cc2]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA6_Veven_Vodd[cc1,cc2]=AA_temp;

    end
end

for cc1=1:l_Veven
    for cc2=1:l_Veven
        @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_even[cc1]'[1,-4];
        @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc2][-4,1];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA1_Veven_Veven[cc1,cc2]=AA_temp;

        @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,-5]*V_even[cc1][-2,1];
        @tensor A_temp[:]:=A6[-1,1,-3,-4,-5]*V_even[cc2]'[1,-2];
        AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
        AA6_Veven_Veven[cc1,cc2]=AA_temp;

    end
end


AA2, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(deepcopy(A2'),deepcopy(A2));
AA3, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(deepcopy(A3'),deepcopy(A3));
AA4, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(deepcopy(A4'),deepcopy(A4));
AA5, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(deepcopy(A5'),deepcopy(A5));

AA2=AA2/norm(AA2);
AA3=AA3/norm(AA3);
AA4=AA4/norm(AA4);
AA5=AA5/norm(AA5);


gate_upper=parity_gate(U_L,2); @tensor gate_upper[:]:=gate_upper[2,1]*U_L[-1,1,3]*U_L'[2,3,-2];
gate_lower=parity_gate(U_L,3); @tensor gate_lower[:]:=gate_lower[2,1]*U_L[-1,3,1]*U_L'[3,2,-2];

global AA1_Vodd_Vodd, AA1_Vodd_Veven, AA1_Veven_Vodd, AA1_Veven_Veven
global AA6_Vodd_Vodd, AA6_Vodd_Veven, AA6_Veven_Vodd, AA6_Veven_Veven
global AA2,AA3,AA4,AA5
global gate_upper, gate_lower

function M_vr(l_Vodd,l_Veven,vr0)
    vr=deepcopy(vr0)*0;#L1,L2,L3,L4,dummy
    #sign from U1+U1'
    #sign from U1*(L1+L2+L3+L4)
    #sign from U1'*(L1'+L2'+L3'+L4')

    #Vodd, Vodd
    for cc1=1:l_Vodd
        for cc2=1:l_Vodd

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Vodd_Vodd[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5);
            AA6_temp=deepcopy(AA6_Vodd_Vodd[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=parity_gate(AA1_temp,1); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vr_temp[:]:=AA1_temp[-1,2,1,12]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*AA4_temp[-4,8,7,6]*AA5_temp[-5,10,9,8]*AA6_temp[-6,12,11,10]*vr_temp[1,3,5,7,9,11,-7];
            vr=vr+vr_temp;

        end
    end

    #Vodd, Veven
    for cc1=1:l_Vodd
        for cc2=1:l_Veven

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Vodd_Veven[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5);
            AA6_temp=deepcopy(AA6_Vodd_Veven[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_upper); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                  
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vr_temp[:]:=AA1_temp[-1,2,1,12]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*AA4_temp[-4,8,7,6]*AA5_temp[-5,10,9,8]*AA6_temp[-6,12,11,10]*vr_temp[1,3,5,7,9,11,-7];
            vr=vr+vr_temp;

        end
    end

    #Veven, Vodd
    for cc1=1:l_Veven
        for cc2=1:l_Vodd

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Veven_Vodd[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5);
            AA6_temp=deepcopy(AA6_Veven_Vodd[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_lower); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                    
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vr_temp[:]:=AA1_temp[-1,2,1,12]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*AA4_temp[-4,8,7,6]*AA5_temp[-5,10,9,8]*AA6_temp[-6,12,11,10]*vr_temp[1,3,5,7,9,11,-7];
            vr=vr+vr_temp;

        end
    end


    #Vodd, Veven
    for cc1=1:l_Veven
        for cc2=1:l_Veven

            vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Veven_Veven[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5);
            AA6_temp=deepcopy(AA6_Veven_Veven[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            #No sign for U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
            @tensor vr_temp[:]:=AA1_temp[-1,2,1,12]*AA2_temp[-2,4,3,2]*AA3_temp[-3,6,5,4]*AA4_temp[-4,8,7,6]*AA5_temp[-5,10,9,8]*AA6_temp[-6,12,11,10]*vr_temp[1,3,5,7,9,11,-7];
            vr=vr+vr_temp;

        end
    end


    return vr;
end

function vl_M(l_Vodd,l_Veven,vl0)
    vl=deepcopy(vl0)*0;#dummy,R1,R2,R3
    #sign from U1+U1'
    #sign from P2*(R1+R1')+P3*(R1+R1'+R2+R2')
    #sign from U1*(L1+L2+L3)
    #sign from U1'*(L1'+L2'+L3')

    #Vodd, Vodd
    for cc1=1:l_Vodd
        for cc2=1:l_Vodd

            vl_temp=deepcopy(vl0);#L1,L2,L3,dummy
            AA1_temp=deepcopy(AA1_Vodd_Vodd[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5);
            AA6_temp=deepcopy(AA6_Vodd_Vodd[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=parity_gate(AA1_temp,1); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vl_temp[:]:=AA1_temp[1,2,-2,12]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*AA4_temp[7,8,-5,6]*AA5_temp[9,10,-6,8]*AA6_temp[11,12,-7,10]*vl_temp[-1,1,3,5,7,9,11];
            vl=vl+vl_temp;

        end
    end

    #Vodd, Veven
    for cc1=1:l_Vodd
        for cc2=1:l_Veven

            vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
            AA1_temp=deepcopy(AA1_Vodd_Veven[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5);
            AA6_temp=deepcopy(AA6_Vodd_Veven[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_upper); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                    
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vl_temp[:]:=AA1_temp[1,2,-2,12]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*AA4_temp[7,8,-5,6]*AA5_temp[9,10,-6,8]*AA6_temp[11,12,-7,10]*vl_temp[-1,1,3,5,7,9,11];
            vl=vl+vl_temp;

        end
    end

    #Veven, Vodd
    for cc1=1:l_Veven
        for cc2=1:l_Vodd

            vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
            AA1_temp=deepcopy(AA1_Veven_Vodd[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5);
            AA6_temp=deepcopy(AA6_Veven_Vodd[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            gate=deepcopy(gate_lower); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                  
            @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
            @tensor vl_temp[:]:=AA1_temp[1,2,-2,12]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*AA4_temp[7,8,-5,6]*AA5_temp[9,10,-6,8]*AA6_temp[11,12,-7,10]*vl_temp[-1,1,3,5,7,9,11];
            vl=vl+vl_temp;

        end
    end


    #Vodd, Veven
    for cc1=1:l_Veven
        for cc2=1:l_Veven

            vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
            AA1_temp=deepcopy(AA1_Veven_Veven[cc1,cc2]);
            AA2_temp=deepcopy(AA2);
            AA3_temp=deepcopy(AA3);
            AA4_temp=deepcopy(AA4);
            AA5_temp=deepcopy(AA5);
            AA6_temp=deepcopy(AA6_Veven_Veven[cc1,cc2]);
            gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
            #No sign for U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
            @tensor vl_temp[:]:=AA1_temp[1,2,-2,12]*AA2_temp[3,4,-3,2]*AA3_temp[5,6,-4,4]*AA4_temp[7,8,-5,6]*AA5_temp[9,10,-6,8]*AA6_temp[11,12,-7,10]*vl_temp[-1,1,3,5,7,9,11];
            vl=vl+vl_temp;

        end
    end


    return vl;
end
v_init=TensorMap(randn, space(AA2,1)*space(AA2,1)*space(AA2,1)*space(AA2,1)*space(AA2,1)*space(AA2,1),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1));
v_init=permute(v_init,(1,2,3,4,5,6,7,),());#L1,L2,L3,L4,dummy
contraction_fun_R(x)=M_vr(l_Vodd,l_Veven,x);
@time _=contraction_fun_R(v_init);flush(stdout)
@time eur,evr=eigsolve(contraction_fun_R, v_init, 2,:LM,Arnoldi(krylovdim=10));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,L4,dummy

println(eur)


v_init=TensorMap(randn, space(AA2,3)*space(AA2,3)*space(AA2,3)*space(AA2,3)*space(AA2,3)*space(AA2,3),GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1)');
v_init=permute(v_init,(7,1,2,3,4,5,6,),());#dummy,R1,R2,R3,R4
contraction_fun_L(x)=vl_M(l_Vodd,l_Veven,x);
@time _=contraction_fun_L(v_init);flush(stdout)
@time eul,evl=eigsolve(contraction_fun_L, v_init, 2,:LM,Arnoldi(krylovdim=10));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3,R4

println(eul)





@tensor VL[:]:=VL[-1,1,2,3,4,5,6]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7]*U_L[4,-8,-9]*U_L[5,-10,-11]*U_L[6,-12,-13];#dummy, R1',R1,R2',R2,R3',R3,R4',R4
VL=permute(VL,(1,2,4,6,8,10,12,3,5,7,9,11,13,));#dummy, R1',R2',R3',R4',R5', R1,R2,R3,R4,R5


@tensor VR[:]:=VR[1,2,3,4,5,6,-13]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4]*U_L'[-9,-10,5]*U_L'[-11,-12,6];#L1',L1,L2',L2,L3',L3,L4',L4,dummy
VR=permute(VR,(2,4,6,8,10,12,1,3,5,7,9,11,13,));#L1,L2,L3,L4,L5,L1',L2',L3',L4',L5',dummy




@tensor H[:]:=VL[1,-1,-2,-3,-4,-5,-6,2,3,4,5,6,7]*VR[2,3,4,5,6,7,-7,-8,-9,-10,-11,-12,1];#R1',R2',R3',R4',R5' ,L1',L2',L3',L4',L5'



eu,ev=eig(H,(1,2,3,4,5,6,),(7,8,9,10,11,12));
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))

ev=permute(ev,(1,2,3,4,5,6,7,));#L1',L2',L3',L4',L5',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,7);#L2',L1',L3',L4',L5',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,7);#L2',L3',L1',L4',L5',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,7);#L2',L3',L4',L1',L5',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),4,5,7);#L2',L3',L4',L5',L1',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),5,6,7);#L2',L3',L4',L5',L6',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,4,5,6,-1]*ev[1,2,3,4,5,6,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;


order=sortperm(abs.(eu));
k_phase=diag(k_phase);
eu_set1=eu[order];
k_phase_set1=k_phase[order];
Qn_set1=Qn[order];
Spin_set1=Spin[order]


####################################
VR=evr[2];#L1,L2,L3,L4,dummy
VL=evl[2];#dummy,R1,R2,R3,R4

@tensor VL[:]:=VL[-1,1,2,3,4,5,6]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7]*U_L[4,-8,-9]*U_L[5,-10,-11]*U_L[6,-12,-13];#dummy, R1',R1,R2',R2,R3',R3,R4',R4
VL=permute(VL,(1,2,4,6,8,10,12,3,5,7,9,11,13,));#dummy, R1',R2',R3',R4',R5', R1,R2,R3,R4,R5


@tensor VR[:]:=VR[1,2,3,4,5,6,-13]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4]*U_L'[-9,-10,5]*U_L'[-11,-12,6];#L1',L1,L2',L2,L3',L3,L4',L4,dummy
VR=permute(VR,(2,4,6,8,10,12,1,3,5,7,9,11,13,));#L1,L2,L3,L4,L5,L1',L2',L3',L4',L5',dummy




@tensor H[:]:=VL[1,-1,-2,-3,-4,-5,-6,2,3,4,5,6,7]*VR[2,3,4,5,6,7,-7,-8,-9,-10,-11,-12,1];#R1',R2',R3',R4',R5' ,L1',L2',L3',L4',L5'



eu,ev=eig(H,(1,2,3,4,5,6,),(7,8,9,10,11,12));
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
Qn=get_Vspace_Qn(space(eu,1)); Qn=Int.(Qn);

eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))

ev=permute(ev,(1,2,3,4,5,6,7,));#L1',L2',L3',L4',L5',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,7);#L2',L1',L3',L4',L5',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,7);#L2',L3',L1',L4',L5',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,7);#L2',L3',L4',L1',L5',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),4,5,7);#L2',L3',L4',L5',L1',L6',dummy
ev_translation=permute_neighbour_ind(deepcopy(ev_translation),5,6,7);#L2',L3',L4',L5',L6',L1',dummy
#ev_translation=permute(ev',(2,3,1,4,));#L1',L2',L3',dummy

@tensor k_phase[:]:=ev_translation[1,2,3,4,5,6,-1]*ev[1,2,3,4,5,6,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;


order=sortperm(abs.(eu));
k_phase=diag(k_phase);
eu_set2=eu[order];
k_phase_set2=k_phase[order];
Qn_set2=Qn[order];
Spin_set2=Spin[order]


##########################

if y_anti_pbc

    matwrite("ES_Gutzwiller_APBC_M1_Nv6_MPS"*".mat", Dict(
        "k_phase_set1" => k_phase_set1,
        "eu_set1" => eu_set1,
        "Qn_set1"=>Qn_set1,
        "Spin_set1"=>Spin_set1,
        "k_phase_set2" => k_phase_set2,
        "eu_set2" => eu_set2,
        "Qn_set2"=>Qn_set2,
        "Spin_set2"=>Spin_set2
    ); compress = false)

else

    matwrite("ES_Gutzwiller_PBC_M1_Nv6_MPS"*".mat", Dict(
        "k_phase_set1" => k_phase_set1,
        "eu_set1" => eu_set1,
        "Qn_set1"=>Qn_set1,
        "Spin_set1"=>Spin_set1,
        "k_phase_set2" => k_phase_set2,
        "eu_set2" => eu_set2,
        "Qn_set2"=>Qn_set2,
        "Spin_set2"=>Spin_set2
    ); compress = false)
end





