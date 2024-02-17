using LinearAlgebra:diag,I,diagm 
using TensorKit
using KrylovKit
using JSON
using HDF5, JLD2, MAT
cd(@__DIR__)



include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\mps_algorithms\\Projector_funs.jl")


y_anti_pbc=false;
filenm="Optim_cell_LS_D_4_chi_40_2.368055.jld2";
data=load(filenm);
A=data["x"][1].T;
B=data["x"][2].T;

A=A*100;
B=B*100;



#convention of fermionic PEPS: |L,U,P><D,R|====L,U,P|><|R,D





#############################
# #convert to the order of PEPS code
A1=deepcopy(A);
A2=deepcopy(A);
A3=deepcopy(A);
A4=deepcopy(A);
A5=deepcopy(A);
A6=deepcopy(A);
if y_anti_pbc
    gauge_gate1=parity_gate(A1,4);
    @tensor A1[:]:=A1[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
end

B1=deepcopy(B);
B2=deepcopy(B);
B3=deepcopy(B);
B4=deepcopy(B);
B5=deepcopy(B);
B6=deepcopy(B);
if y_anti_pbc
    gauge_gate1=parity_gate(B1,4);
    @tensor B1[:]:=B1[-1,-2,-3,1,-5]*gauge_gate1[-4,1];
end


#############################


function construct_tensors(A1,A2,A3,A4,A5,A6)
    V_odd,V_even=projector_virtual(space(A1,4))

    Phy_odd,Phy_even=projector_physical(space(A1,5))

    l_Vodd=length(V_odd);
    l_Veven=length(V_even);


    AA1_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
    AA1_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
    AA1_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
    AA1_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer

    AA6_Phyodd_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
    AA6_Phyodd_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
    AA6_Phyodd_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
    AA6_Phyodd_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer

    AA6_Phyeven_Vodd_Vodd=Matrix(undef,l_Vodd,l_Vodd);#upper layer, lower layer
    AA6_Phyeven_Vodd_Veven=Matrix(undef,l_Vodd,l_Veven);#upper layer, lower layer
    AA6_Phyeven_Veven_Vodd=Matrix(undef,l_Veven,l_Vodd);#upper layer, lower layer
    AA6_Phyeven_Veven_Veven=Matrix(undef,l_Veven,l_Veven);#upper layer, lower layer


    for cc1=1:l_Vodd
        for cc2=1:l_Vodd
            @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_odd[cc1]'[1,-4];
            @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc2][-4,1];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA1_Vodd_Vodd[cc1,cc2]=AA_temp;

            @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,2]*V_odd[cc1][-2,1]*Phy_odd'[2,-5];
            @tensor A_temp[:]:=A6[-1,1,-3,-4,2]*V_odd[cc2]'[1,-2]*Phy_odd[-5,2];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA6_Phyodd_Vodd_Vodd[cc1,cc2]=AA_temp;

            @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,2]*V_odd[cc1][-2,1]*Phy_even'[2,-5];
            @tensor A_temp[:]:=A6[-1,1,-3,-4,2]*V_odd[cc2]'[1,-2]*Phy_even[-5,2];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA6_Phyeven_Vodd_Vodd[cc1,cc2]=AA_temp;

        end
    end

    for cc1=1:l_Vodd
        for cc2=1:l_Veven
            @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_odd[cc1]'[1,-4];
            @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc2][-4,1];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA1_Vodd_Veven[cc1,cc2]=AA_temp;

            @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,2]*V_odd[cc1][-2,1]*Phy_odd'[2,-5];
            @tensor A_temp[:]:=A6[-1,1,-3,-4,2]*V_even[cc2]'[1,-2]*Phy_odd[-5,2];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA6_Phyodd_Vodd_Veven[cc1,cc2]=AA_temp;

            @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,2]*V_odd[cc1][-2,1]*Phy_even'[2,-5];
            @tensor A_temp[:]:=A6[-1,1,-3,-4,2]*V_even[cc2]'[1,-2]*Phy_even[-5,2];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA6_Phyeven_Vodd_Veven[cc1,cc2]=AA_temp;

        end
    end

    for cc1=1:l_Veven
        for cc2=1:l_Vodd
            @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_even[cc1]'[1,-4];
            @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_odd[cc2][-4,1];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA1_Veven_Vodd[cc1,cc2]=AA_temp;

            @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,2]*V_even[cc1][-2,1]*Phy_odd'[2,-5];
            @tensor A_temp[:]:=A6[-1,1,-3,-4,2]*V_odd[cc2]'[1,-2]*Phy_odd[-5,2];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA6_Phyodd_Veven_Vodd[cc1,cc2]=AA_temp;

            @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,2]*V_even[cc1][-2,1]*Phy_even'[2,-5];
            @tensor A_temp[:]:=A6[-1,1,-3,-4,2]*V_odd[cc2]'[1,-2]*Phy_even[-5,2];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA6_Phyeven_Veven_Vodd[cc1,cc2]=AA_temp;

        end
    end

    for cc1=1:l_Veven
        for cc2=1:l_Veven
            @tensor Ap_temp[:]:=A1'[-1,-2,-3,1,-5]*V_even[cc1]'[1,-4];
            @tensor A_temp[:]:=A1[-1,-2,-3,1,-5]*V_even[cc2][-4,1];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA1_Veven_Veven[cc1,cc2]=AA_temp;

            @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,2]*V_even[cc1][-2,1]*Phy_odd'[2,-5];
            @tensor A_temp[:]:=A6[-1,1,-3,-4,2]*V_even[cc2]'[1,-2]*Phy_odd[-5,2];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA6_Phyodd_Veven_Veven[cc1,cc2]=AA_temp;

            @tensor Ap_temp[:]:=A6'[-1,1,-3,-4,2]*V_even[cc1][-2,1]*Phy_even'[2,-5];
            @tensor A_temp[:]:=A6[-1,1,-3,-4,2]*V_even[cc2]'[1,-2]*Phy_even[-5,2];
            AA_temp, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);
            AA6_Phyeven_Veven_Veven[cc1,cc2]=AA_temp;

        end
    end

    @tensor Ap_temp[:]:=A2'[-1,-2,-3,-4,1]*Phy_even'[1,-5];
    @tensor A_temp[:]:=A2[-1,-2,-3,-4,1]*Phy_even[-5,1];
    AA2_Phyeven, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

    @tensor Ap_temp[:]:=A2'[-1,-2,-3,-4,1]*Phy_odd'[1,-5];
    @tensor A_temp[:]:=A2[-1,-2,-3,-4,1]*Phy_odd[-5,1];
    AA2_Phyodd, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

    @tensor Ap_temp[:]:=A3'[-1,-2,-3,-4,1]*Phy_even'[1,-5];
    @tensor A_temp[:]:=A3[-1,-2,-3,-4,1]*Phy_even[-5,1];
    AA3_Phyeven, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

    @tensor Ap_temp[:]:=A3'[-1,-2,-3,-4,1]*Phy_odd'[1,-5];
    @tensor A_temp[:]:=A3[-1,-2,-3,-4,1]*Phy_odd[-5,1];
    AA3_Phyodd, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

    @tensor Ap_temp[:]:=A4'[-1,-2,-3,-4,1]*Phy_even'[1,-5];
    @tensor A_temp[:]:=A4[-1,-2,-3,-4,1]*Phy_even[-5,1];
    AA4_Phyeven, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

    @tensor Ap_temp[:]:=A4'[-1,-2,-3,-4,1]*Phy_odd'[1,-5];
    @tensor A_temp[:]:=A4[-1,-2,-3,-4,1]*Phy_odd[-5,1];
    AA4_Phyodd, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

    @tensor Ap_temp[:]:=A5'[-1,-2,-3,-4,1]*Phy_even'[1,-5];
    @tensor A_temp[:]:=A5[-1,-2,-3,-4,1]*Phy_even[-5,1];
    AA5_Phyeven, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

    @tensor Ap_temp[:]:=A5'[-1,-2,-3,-4,1]*Phy_odd'[1,-5];
    @tensor A_temp[:]:=A5[-1,-2,-3,-4,1]*Phy_odd[-5,1];
    AA5_Phyodd, U_L,U_D,U_R,U_U=build_double_layer_NoSwap(Ap_temp,A_temp);

    gate_upper=parity_gate(U_L,2); @tensor gate_upper[:]:=gate_upper[2,1]*U_L[-1,1,3]*U_L'[2,3,-2];
    gate_lower=parity_gate(U_L,3); @tensor gate_lower[:]:=gate_lower[2,1]*U_L[-1,3,1]*U_L'[3,2,-2];

    AA1_set=(AA1_Vodd_Vodd, AA1_Vodd_Veven, AA1_Veven_Vodd, AA1_Veven_Veven);
    AA2_set=(AA2_Phyodd, AA2_Phyeven);
    AA3_set=(AA3_Phyodd, AA3_Phyeven);
    AA4_set=(AA4_Phyodd, AA4_Phyeven);
    AA5_set=(AA5_Phyodd, AA5_Phyeven);
    AA6_set=(AA6_Phyeven_Vodd_Vodd, AA6_Phyeven_Vodd_Veven, AA6_Phyeven_Veven_Vodd, AA6_Phyeven_Veven_Veven,  AA6_Phyodd_Vodd_Vodd, AA6_Phyodd_Vodd_Veven, AA6_Phyodd_Veven_Vodd, AA6_Phyodd_Veven_Veven);
    gate_set=(gate_upper, gate_lower);

    return l_Vodd,l_Veven,AA1_set,AA2_set,AA3_set,AA4_set,AA5_set,AA6_set,gate_set,U_L,U_D,U_R,U_U
end



function M_vr(l_Vodd,l_Veven,vr0,  AA1_set,AA2_set,AA3_set,AA4_set, AA5_set, AA6_set,  gate_set)
    AA1_Vodd_Vodd, AA1_Vodd_Veven, AA1_Veven_Vodd, AA1_Veven_Veven=AA1_set;
    AA2_Phyodd, AA2_Phyeven=AA2_set;
    AA3_Phyodd, AA3_Phyeven=AA3_set;
    AA4_Phyodd, AA4_Phyeven=AA4_set;
    AA5_Phyodd, AA5_Phyeven=AA5_set;
    AA6_Phyeven_Vodd_Vodd, AA6_Phyeven_Vodd_Veven, AA6_Phyeven_Veven_Vodd, AA6_Phyeven_Veven_Veven,   AA6_Phyodd_Vodd_Vodd, AA6_Phyodd_Vodd_Veven, AA6_Phyodd_Veven_Vodd, AA6_Phyodd_Veven_Veven = AA6_set;
    gate_upper,gate_lower=gate_set;
    global U_DD

    vr=deepcopy(vr0)*0;#L1,L2,L3,L4,dummy
    #sign from U1+U1'
    #sign from P2*(R1+R1')+P3*(R1+R1'+R2+R2')+P4*(R1+R1'+R2+R2'+R3+R3')
    #sign from U1*(L1+L2+L3+L4)
    #sign from U1'*(L1'+L2'+L3'+L4')

    #Vodd, Vodd
    for cc1=1:l_Vodd
        for cc2=1:l_Vodd
            for parity_P2=0:1
                for parity_P3=0:1
                    for parity_P4=0:1
                        for parity_P5=0:1
                            for parity_P6=0:1
                                vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
                                AA1_temp=deepcopy(AA1_Vodd_Vodd[cc1,cc2]);
                                if parity_P2==1
                                    AA2_temp=deepcopy(AA2_Phyodd);
                                elseif parity_P2==0
                                    AA2_temp=deepcopy(AA2_Phyeven);
                                end
                                if parity_P3==1
                                    AA3_temp=deepcopy(AA3_Phyodd);
                                elseif parity_P3==0
                                    AA3_temp=deepcopy(AA3_Phyeven);
                                end
                                if parity_P4==1
                                    AA4_temp=deepcopy(AA4_Phyodd);
                                elseif parity_P4==0
                                    AA4_temp=deepcopy(AA4_Phyeven);
                                end
                                if parity_P5==1
                                    AA5_temp=deepcopy(AA5_Phyodd);
                                elseif parity_P5==0
                                    AA5_temp=deepcopy(AA5_Phyeven);
                                end
                                if parity_P6==1
                                    AA6_temp=deepcopy(AA6_Phyodd_Vodd_Vodd[cc1,cc2]);
                                elseif parity_P6==0
                                    AA6_temp=deepcopy(AA6_Phyeven_Vodd_Vodd[cc1,cc2]);
                                end
                                gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                                if parity_P2==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                                end
                                if parity_P3==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                                end
                                if parity_P4==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P5==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P6==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA5_temp,3); @tensor AA5_temp[:]:=AA5_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                gate=parity_gate(AA1_temp,1); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
                                @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA1AA2[:]:=AA1_temp[1,3,4,-4]*AA2_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA3AA4[:]:=AA3_temp[1,3,4,-4]*AA4_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA5AA6[:]:=AA5_temp[1,3,4,-4]*AA6_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor vr_temp[:]:=AA1AA2[-1,2,1,6]*AA3AA4[-2,4,3,2]*AA5AA6[-3,6,5,4]*vr_temp[1,3,5,-4];
                                vr=vr+vr_temp;
                            end
                        end
                    end
                end
            end
        end
    end

    #Vodd, Veven
    for cc1=1:l_Vodd
        for cc2=1:l_Veven
            for parity_P2=0:1
                for parity_P3=0:1
                    for parity_P4=0:1
                        for parity_P5=0:1
                            for parity_P6=0:1
                                vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
                                AA1_temp=deepcopy(AA1_Vodd_Veven[cc1,cc2]);
                                if parity_P2==1
                                    AA2_temp=deepcopy(AA2_Phyodd);
                                elseif parity_P2==0
                                    AA2_temp=deepcopy(AA2_Phyeven);
                                end
                                if parity_P3==1
                                    AA3_temp=deepcopy(AA3_Phyodd);
                                elseif parity_P3==0
                                    AA3_temp=deepcopy(AA3_Phyeven);
                                end
                                if parity_P4==1
                                    AA4_temp=deepcopy(AA4_Phyodd);
                                elseif parity_P4==0
                                    AA4_temp=deepcopy(AA4_Phyeven);
                                end
                                if parity_P5==1
                                    AA5_temp=deepcopy(AA5_Phyodd);
                                elseif parity_P5==0
                                    AA5_temp=deepcopy(AA5_Phyeven);
                                end
                                if parity_P6==1
                                    AA6_temp=deepcopy(AA6_Phyodd_Vodd_Veven[cc1,cc2]);
                                elseif parity_P6==0
                                    AA6_temp=deepcopy(AA6_Phyeven_Vodd_Veven[cc1,cc2]);
                                end
                                gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                                if parity_P2==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                                end
                                if parity_P3==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                                end
                                if parity_P4==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P5==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P6==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA5_temp,3); @tensor AA5_temp[:]:=AA5_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                gate=deepcopy(gate_upper); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                  
                                @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA1AA2[:]:=AA1_temp[1,3,4,-4]*AA2_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA3AA4[:]:=AA3_temp[1,3,4,-4]*AA4_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA5AA6[:]:=AA5_temp[1,3,4,-4]*AA6_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor vr_temp[:]:=AA1AA2[-1,2,1,6]*AA3AA4[-2,4,3,2]*AA5AA6[-3,6,5,4]*vr_temp[1,3,5,-4];
                                vr=vr+vr_temp;
                            end
                        end
                    end
                end
            end
        end
    end

    #Veven, Vodd
    for cc1=1:l_Veven
        for cc2=1:l_Vodd
            for parity_P2=0:1
                for parity_P3=0:1
                    for parity_P4=0:1
                        for parity_P5=0:1
                            for parity_P6=0:1
                                vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
                                AA1_temp=deepcopy(AA1_Veven_Vodd[cc1,cc2]);
                                if parity_P2==1
                                    AA2_temp=deepcopy(AA2_Phyodd);
                                elseif parity_P2==0
                                    AA2_temp=deepcopy(AA2_Phyeven);
                                end
                                if parity_P3==1
                                    AA3_temp=deepcopy(AA3_Phyodd);
                                elseif parity_P3==0
                                    AA3_temp=deepcopy(AA3_Phyeven);
                                end
                                if parity_P4==1
                                    AA4_temp=deepcopy(AA4_Phyodd);
                                elseif parity_P4==0
                                    AA4_temp=deepcopy(AA4_Phyeven);
                                end
                                if parity_P5==1
                                    AA5_temp=deepcopy(AA5_Phyodd);
                                elseif parity_P5==0
                                    AA5_temp=deepcopy(AA5_Phyeven);
                                end
                                if parity_P6==1
                                    AA6_temp=deepcopy(AA6_Phyodd_Veven_Vodd[cc1,cc2]);
                                elseif parity_P6==0
                                    AA6_temp=deepcopy(AA6_Phyeven_Veven_Vodd[cc1,cc2]);
                                end
                                gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                                if parity_P2==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                                end
                                if parity_P3==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                                end
                                if parity_P4==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P5==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P6==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA5_temp,3); @tensor AA5_temp[:]:=AA5_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                gate=deepcopy(gate_lower); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                    
                                @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA1AA2[:]:=AA1_temp[1,3,4,-4]*AA2_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA3AA4[:]:=AA3_temp[1,3,4,-4]*AA4_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA5AA6[:]:=AA5_temp[1,3,4,-4]*AA6_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor vr_temp[:]:=AA1AA2[-1,2,1,6]*AA3AA4[-2,4,3,2]*AA5AA6[-3,6,5,4]*vr_temp[1,3,5,-4];
                                vr=vr+vr_temp;
                            end
                        end
                    end
                end
            end
        end
    end


    #Vodd, Veven
    for cc1=1:l_Veven
        for cc2=1:l_Veven
            for parity_P2=0:1
                for parity_P3=0:1
                    for parity_P4=0:1
                        for parity_P5=0:1
                            for parity_P6=0:1
                                vr_temp=deepcopy(vr0);#L1,L2,L3,dummy
                                AA1_temp=deepcopy(AA1_Veven_Veven[cc1,cc2]);
                                if parity_P2==1
                                    AA2_temp=deepcopy(AA2_Phyodd);
                                elseif parity_P2==0
                                    AA2_temp=deepcopy(AA2_Phyeven);
                                end
                                if parity_P3==1
                                    AA3_temp=deepcopy(AA3_Phyodd);
                                elseif parity_P3==0
                                    AA3_temp=deepcopy(AA3_Phyeven);
                                end
                                if parity_P4==1
                                    AA4_temp=deepcopy(AA4_Phyodd);
                                elseif parity_P4==0
                                    AA4_temp=deepcopy(AA4_Phyeven);
                                end
                                if parity_P5==1
                                    AA5_temp=deepcopy(AA5_Phyodd);
                                elseif parity_P5==0
                                    AA5_temp=deepcopy(AA5_Phyeven);
                                end
                                if parity_P6==1
                                    AA6_temp=deepcopy(AA6_Phyodd_Veven_Veven[cc1,cc2]);
                                elseif parity_P6==0
                                    AA6_temp=deepcopy(AA6_Phyeven_Veven_Veven[cc1,cc2]);
                                end
                                gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                                if parity_P2==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                                end
                                if parity_P3==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                                end
                                if parity_P4==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P5==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P6==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA5_temp,3); @tensor AA5_temp[:]:=AA5_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                #No sign for U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
                                @tensor AA1AA2[:]:=AA1_temp[1,3,4,-4]*AA2_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA3AA4[:]:=AA3_temp[1,3,4,-4]*AA4_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA5AA6[:]:=AA5_temp[1,3,4,-4]*AA6_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor vr_temp[:]:=AA1AA2[-1,2,1,6]*AA3AA4[-2,4,3,2]*AA5AA6[-3,6,5,4]*vr_temp[1,3,5,-4];
                                vr=vr+vr_temp;
                            end
                        end
                    end
                end
            end
        end
    end

    println("finished one Mv operation");flush(stdout);

    return vr;
end

function vl_M(l_Vodd,l_Veven,vl0,  AA1_set,AA2_set,AA3_set,AA4_set, AA5_set, AA6_set,  gate_set)
    AA1_Vodd_Vodd, AA1_Vodd_Veven, AA1_Veven_Vodd, AA1_Veven_Veven=AA1_set;
    AA2_Phyodd, AA2_Phyeven=AA2_set;
    AA3_Phyodd, AA3_Phyeven=AA3_set;
    AA4_Phyodd, AA4_Phyeven=AA4_set;
    AA5_Phyodd, AA5_Phyeven=AA5_set;
    AA6_Phyeven_Vodd_Vodd, AA6_Phyeven_Vodd_Veven, AA6_Phyeven_Veven_Vodd, AA6_Phyeven_Veven_Veven,   AA6_Phyodd_Vodd_Vodd, AA6_Phyodd_Vodd_Veven, AA6_Phyodd_Veven_Vodd, AA6_Phyodd_Veven_Veven = AA6_set;
    gate_upper,gate_lower=gate_set;
    global U_DD

    vl=deepcopy(vl0)*0;#dummy,R1,R2,R3
    #sign from U1+U1'
    #sign from P2*(R1+R1')+P3*(R1+R1'+R2+R2')
    #sign from U1*(L1+L2+L3)
    #sign from U1'*(L1'+L2'+L3')

    #Vodd, Vodd
    for cc1=1:l_Vodd
        for cc2=1:l_Vodd
            for parity_P2=0:1
                for parity_P3=0:1
                    for parity_P4=0:1
                        for parity_P5=0:1
                            for parity_P6=0:1
                                vl_temp=deepcopy(vl0);#L1,L2,L3,dummy
                                AA1_temp=deepcopy(AA1_Vodd_Vodd[cc1,cc2]);
                                if parity_P2==1
                                    AA2_temp=deepcopy(AA2_Phyodd);
                                elseif parity_P2==0
                                    AA2_temp=deepcopy(AA2_Phyeven);
                                end
                                if parity_P3==1
                                    AA3_temp=deepcopy(AA3_Phyodd);
                                elseif parity_P3==0
                                    AA3_temp=deepcopy(AA3_Phyeven);
                                end
                                if parity_P4==1
                                    AA4_temp=deepcopy(AA4_Phyodd);
                                elseif parity_P4==0
                                    AA4_temp=deepcopy(AA4_Phyeven);
                                end
                                if parity_P5==1
                                    AA5_temp=deepcopy(AA5_Phyodd);
                                elseif parity_P5==0
                                    AA5_temp=deepcopy(AA5_Phyeven);
                                end
                                if parity_P6==1
                                    AA6_temp=deepcopy(AA6_Phyodd_Vodd_Vodd[cc1,cc2]);
                                elseif parity_P6==0
                                    AA6_temp=deepcopy(AA6_Phyeven_Vodd_Vodd[cc1,cc2]);
                                end
                                gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                                if parity_P2==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                                end
                                if parity_P3==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                                end
                                if parity_P4==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P5==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P6==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA5_temp,3); @tensor AA5_temp[:]:=AA5_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                gate=parity_gate(AA1_temp,1); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
                                @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA1AA2[:]:=AA1_temp[1,3,4,-4]*AA2_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA3AA4[:]:=AA3_temp[1,3,4,-4]*AA4_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA5AA6[:]:=AA5_temp[1,3,4,-4]*AA6_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor vl_temp[:]:=AA1AA2[1,2,-2,6]*AA3AA4[3,4,-3,2]*AA5AA6[5,6,-4,4]*vl_temp[-1,1,3,5];
                                vl=vl+vl_temp;
                            end
                        end
                    end
                end
            end
        end
    end

    #Vodd, Veven
    for cc1=1:l_Vodd
        for cc2=1:l_Veven
            for parity_P2=0:1
                for parity_P3=0:1
                    for parity_P4=0:1
                        for parity_P5=0:1
                            for parity_P6=0:1
                                vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
                                AA1_temp=deepcopy(AA1_Vodd_Veven[cc1,cc2]);
                                if parity_P2==1
                                    AA2_temp=deepcopy(AA2_Phyodd);
                                elseif parity_P2==0
                                    AA2_temp=deepcopy(AA2_Phyeven);
                                end
                                if parity_P3==1
                                    AA3_temp=deepcopy(AA3_Phyodd);
                                elseif parity_P3==0
                                    AA3_temp=deepcopy(AA3_Phyeven);
                                end
                                if parity_P4==1
                                    AA4_temp=deepcopy(AA4_Phyodd);
                                elseif parity_P4==0
                                    AA4_temp=deepcopy(AA4_Phyeven);
                                end
                                if parity_P5==1
                                    AA5_temp=deepcopy(AA5_Phyodd);
                                elseif parity_P5==0
                                    AA5_temp=deepcopy(AA5_Phyeven);
                                end
                                if parity_P6==1
                                    AA6_temp=deepcopy(AA6_Phyodd_Vodd_Veven[cc1,cc2]);
                                elseif parity_P6==0
                                    AA6_temp=deepcopy(AA6_Phyeven_Vodd_Veven[cc1,cc2]);
                                end
                                gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                                if parity_P2==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                                end
                                if parity_P3==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                                end
                                if parity_P4==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P5==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P6==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA5_temp,3); @tensor AA5_temp[:]:=AA5_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                gate=deepcopy(gate_upper); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                    
                                @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA1AA2[:]:=AA1_temp[1,3,4,-4]*AA2_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA3AA4[:]:=AA3_temp[1,3,4,-4]*AA4_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA5AA6[:]:=AA5_temp[1,3,4,-4]*AA6_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor vl_temp[:]:=AA1AA2[1,2,-2,6]*AA3AA4[3,4,-3,2]*AA5AA6[5,6,-4,4]*vl_temp[-1,1,3,5];
                                vl=vl+vl_temp;
                            end
                        end
                    end
                end
            end
        end
    end

    #Veven, Vodd
    for cc1=1:l_Veven
        for cc2=1:l_Vodd
            for parity_P2=0:1
                for parity_P3=0:1
                    for parity_P4=0:1
                        for parity_P5=0:1
                            for parity_P6=0:1
                                vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
                                AA1_temp=deepcopy(AA1_Veven_Vodd[cc1,cc2]);
                                if parity_P2==1
                                    AA2_temp=deepcopy(AA2_Phyodd);
                                elseif parity_P2==0
                                    AA2_temp=deepcopy(AA2_Phyeven);
                                end
                                if parity_P3==1
                                    AA3_temp=deepcopy(AA3_Phyodd);
                                elseif parity_P3==0
                                    AA3_temp=deepcopy(AA3_Phyeven);
                                end
                                if parity_P4==1
                                    AA4_temp=deepcopy(AA4_Phyodd);
                                elseif parity_P4==0
                                    AA4_temp=deepcopy(AA4_Phyeven);
                                end
                                if parity_P5==1
                                    AA5_temp=deepcopy(AA5_Phyodd);
                                elseif parity_P5==0
                                    AA5_temp=deepcopy(AA5_Phyeven);
                                end
                                if parity_P6==1
                                    AA6_temp=deepcopy(AA6_Phyodd_Veven_Vodd[cc1,cc2]);
                                elseif parity_P6==0
                                    AA6_temp=deepcopy(AA6_Phyeven_Veven_Vodd[cc1,cc2]);
                                end
                                gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                                if parity_P2==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                                end
                                if parity_P3==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                                end
                                if parity_P4==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P5==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P6==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA5_temp,3); @tensor AA5_temp[:]:=AA5_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                gate=deepcopy(gate_lower); #sign U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4')                  
                                @tensor AA1_temp[:]:=AA1_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA2_temp[:]:=AA2_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA3_temp[:]:=AA3_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA4_temp[:]:=AA4_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA5_temp[:]:=AA5_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA6_temp[:]:=AA6_temp[1,-2,-3,-4]*gate[-1,1];
                                @tensor AA1AA2[:]:=AA1_temp[1,3,4,-4]*AA2_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA3AA4[:]:=AA3_temp[1,3,4,-4]*AA4_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA5AA6[:]:=AA5_temp[1,3,4,-4]*AA6_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor vl_temp[:]:=AA1AA2[1,2,-2,6]*AA3AA4[3,4,-3,2]*AA5AA6[5,6,-4,4]*vl_temp[-1,1,3,5];
                                vl=vl+vl_temp;
                            end
                        end
                    end
                end
            end
        end
    end


    #Vodd, Veven
    for cc1=1:l_Veven
        for cc2=1:l_Veven
            for parity_P2=0:1
                for parity_P3=0:1
                    for parity_P4=0:1
                        for parity_P5=0:1
                            for parity_P6=0:1
                                vl_temp=deepcopy(vl0);#dummy,R1,R2,R3
                                AA1_temp=deepcopy(AA1_Veven_Veven[cc1,cc2]);
                                if parity_P2==1
                                    AA2_temp=deepcopy(AA2_Phyodd);
                                elseif parity_P2==0
                                    AA2_temp=deepcopy(AA2_Phyeven);
                                end
                                if parity_P3==1
                                    AA3_temp=deepcopy(AA3_Phyodd);
                                elseif parity_P3==0
                                    AA3_temp=deepcopy(AA3_Phyeven);
                                end
                                if parity_P4==1
                                    AA4_temp=deepcopy(AA4_Phyodd);
                                elseif parity_P4==0
                                    AA4_temp=deepcopy(AA4_Phyeven);
                                end
                                if parity_P5==1
                                    AA5_temp=deepcopy(AA5_Phyodd);
                                elseif parity_P5==0
                                    AA5_temp=deepcopy(AA5_Phyeven);
                                end
                                if parity_P6==1
                                    AA6_temp=deepcopy(AA6_Phyodd_Veven_Veven[cc1,cc2]);
                                elseif parity_P6==0
                                    AA6_temp=deepcopy(AA6_Phyeven_Veven_Veven[cc1,cc2]);
                                end
                                gate=parity_gate(AA1_temp,4); @tensor AA1_temp[:]:=AA1_temp[-1,-2,-3,1]*gate[-4,1];#sign of U1,U1'
                                if parity_P2==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P2*(R1+R1')
                                end
                                if parity_P3==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R1+R1')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P3*(R2+R2')
                                end
                                if parity_P4==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P5==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                if parity_P6==1
                                    gate=parity_gate(AA1_temp,3); @tensor AA1_temp[:]:=AA1_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA2_temp,3); @tensor AA2_temp[:]:=AA2_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA3_temp,3); @tensor AA3_temp[:]:=AA3_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA4_temp,3); @tensor AA4_temp[:]:=AA4_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                    gate=parity_gate(AA5_temp,3); @tensor AA5_temp[:]:=AA5_temp[-1,-2,1,-4]*gate[-3,1];#sign P4*(R1+R1'+R2+R2'+R3+R3')
                                end
                                #No sign for U1*(L1+L2+L3+L4) + U1'*(L1'+L2'+L3'+L4') 
                                @tensor AA1AA2[:]:=AA1_temp[1,3,4,-4]*AA2_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA3AA4[:]:=AA3_temp[1,3,4,-4]*AA4_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor AA5AA6[:]:=AA5_temp[1,3,4,-4]*AA6_temp[2,-2,5,3]*U_DD[-1,1,2]*U_DD'[4,5,-3];
                                @tensor vl_temp[:]:=AA1AA2[1,2,-2,6]*AA3AA4[3,4,-3,2]*AA5AA6[5,6,-4,4]*vl_temp[-1,1,3,5];
                                vl=vl+vl_temp;
                            end
                        end
                    end
                end
            end
        end
    end
    println("finished one Mv operation");flush(stdout);

    return vl;
end


l_Vodd,l_Veven,AA1_set,AA2_set,AA3_set,AA4_set,AA5_set,AA6_set,gateA_set,U_L,U_D,U_R,U_U=construct_tensors(A1,A2,A3,A4,A5,A6);
l_Vodd,l_Veven,BB1_set,BB2_set,BB3_set,BB4_set,BB5_set,BB6_set,gateB_set,U_L,U_D,U_R,U_U=construct_tensors(B1,B2,B3,B4,B5,B6);

U_DD=unitary(fuse(space(AA2_set[1],1)*space(AA2_set[1],1)), space(AA2_set[1],1)*space(AA2_set[1],1));
global U_DD



v_init=TensorMap(randn, fuse(space(AA2_set[1],1)*space(AA2_set[1],1))*fuse(space(AA2_set[1],1)*space(AA2_set[1],1))*fuse(space(AA2_set[1],1)*space(AA2_set[1],1)),Rep[SU₂]((0)=>1));
v_init=permute(v_init,(1,2,3,4,),());#L1,L2,L3,L4,dummy
contraction_fun_R(x)=M_vr(l_Vodd,l_Veven, M_vr(l_Vodd,l_Veven, x, BB1_set,BB2_set,BB3_set,BB4_set,BB5_set,BB6_set,gateB_set), AA1_set,AA2_set,AA3_set,AA4_set,AA5_set,AA6_set,gateA_set);
@time eur,evr=eigsolve(contraction_fun_R, v_init, 1,:LM,Arnoldi(krylovdim=40));
VR=evr[findmax(abs.(eur))[2]];#L1,L2,L3,L4,dummy
VR_aa=deepcopy(VR);

v_init=TensorMap(randn, fuse(space(AA2_set[1],1)*space(AA2_set[1],1))'*fuse(space(AA2_set[1],1)*space(AA2_set[1],1))'*fuse(space(AA2_set[1],1)*space(AA2_set[1],1))',Rep[SU₂]((0)=>1)');
v_init=permute(v_init,(4,1,2,3,),());#dummy,R1,R2,R3,R4
contraction_fun_L(x)=vl_M(l_Vodd,l_Veven, vl_M(l_Vodd,l_Veven,x, AA1_set,AA2_set,AA3_set,AA4_set,AA5_set,AA6_set,gateA_set), BB1_set,BB2_set,BB3_set,BB4_set,BB5_set,BB6_set,gateB_set);
@time eul,evl=eigsolve(contraction_fun_L, v_init, 1,:LM,Arnoldi(krylovdim=40));
VL=evl[findmax(abs.(eul))[2]];#dummy,R1,R2,R3,R4





@tensor VR[:]:=VR[1,2,3,-7]*U_DD'[-1,-2,1]*U_DD'[-3,-4,2]*U_DD'[-5,-6,3];
@tensor VL[:]:=VL[-1,1,2,3]*U_DD[1,-2,-3]*U_DD[2,-4,-5]*U_DD[3,-6,-7];


@tensor VL[:]:=VL[-1,1,2,3,4,5,6]*U_L[1,-2,-3]*U_L[2,-4,-5]*U_L[3,-6,-7]*U_L[4,-8,-9]*U_L[5,-10,-11]*U_L[6,-12,-13];#dummy, R1',R1,R2',R2,R3',R3,R4',R4,R5',R5,R6',R6
VL=permute(VL,(1,2,4,6,8,10,12,3,5,7,9,11,13,));#dummy, R1',R2',R3',R4', R1,R2,R3,R4


@tensor VR[:]:=VR[1,2,3,4,5,6,-13]*U_L'[-1,-2,1]*U_L'[-3,-4,2]*U_L'[-5,-6,3]*U_L'[-7,-8,4]*U_L'[-9,-10,5]*U_L'[-11,-12,6];#L1',L1,L2',L2,L3',L3,L4',L4,dummy
VR=permute(VR,(2,4,6,8,10,12,1,3,5,7,9,11,13,));#L1,L2,L3,L4,L1',L2',L3',L4',dummy




@tensor H[:]:=VL[1,-1,-2,-3,-4,-5,-6,2,3,4,5,6,7]*VR[2,3,4,5,6,7,-7,-8,-9,-10,-11,-12,1];#R1',R2',R3',R4' ,L1',L2',L3',L4'




eu,ev=eig(H,(1,2,3,4,5,6,),(7,8,9,10,11,12,))
Spin=get_Vspace_Spin(space(eu,1));Spin=Float64.(Spin);
eu=diag(convert(Array,eu));
eu=eu/sum(eu)



println(sort(abs.(eu)))


if y_anti_pbc
    ev=permute(ev,(1,2,3,4,5,6,7,));#L1',L2',L3',L4',dummy
    op=parity_gate(ev,1);
    @tensor ev_translation[:]:=op[1,-1]*ev'[1,-2,-3,-4,-5,-6,-7];
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),1,2,7);#L2',L1',L3',L4',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,7);#L2',L3',L1',L4',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,7);#L2',L3',L4',L1',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),4,5,7);#L2',L3',L4',L1',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),5,6,7);#L2',L3',L4',L1',dummy
else
    ev=permute(ev,(1,2,3,4,5,6,7,));#L1',L2',L3',L4',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev'),1,2,7);#L2',L1',L3',L4',L5',L6',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),2,3,7);#L2',L3',L1',L4',L5',L6',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),3,4,7);#L2',L3',L4',L1',L5',L6',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),4,5,7);#L2',L3',L4',L5',L1',L6',dummy
    ev_translation=permute_neighbour_ind(deepcopy(ev_translation),5,6,7);#L2',L3',L4',L5',L6',L1',dummy
end


@tensor k_phase[:]:=ev_translation[1,2,3,4,5,6,-1]*ev[1,2,3,4,5,6,-2];
k_phase=convert(Array,k_phase);
#@assert norm(diagm(diag(k_phase))-k_phase)/norm(k_phase)<1e-10;


order=sortperm(abs.(eu));
if length(order)>500
    order=order[end-500:end];
end
k_phase=diag(k_phase);
eu=eu[order];
k_phase=k_phase[order];
Spin=Spin[order]


##########################

if y_anti_pbc
    matnm="ES_unprojected_M1_Nv6_APBC"*".mat";
else
    matnm="ES_unprojected_M1_Nv6_PBC"*".mat";
end
matwrite(matnm, Dict(
    "k_phase" => k_phase,
    "eu" => eu,
    "Spin"=>Spin
); compress = false)



