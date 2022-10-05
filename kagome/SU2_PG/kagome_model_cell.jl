using LinearAlgebra
using TensorKit


function evaluate_ob_UpTriangle_single_layer(parameters, U_phy, U_D_phy, A_cell, CTM, method1, method2)
    @assert (size(A_cell,1)==2);
    @assert (size(A_cell,2)==2);
    H_triangle, H_bond, H12_tensorkit, H31_tensorkit, H23_tensorkit=Hamiltonians(U_phy,parameters["J1"],parameters["J2"],parameters["J3"],parameters["Jchi"],parameters["Jtrip"])
        
    if method1=="E_triangle" #calculate up triangle energy
        norm_1cell=ob_1cell_closed(CTM,A_cell,method2);#1 set of unitcell

        A_op_cell=deepcopy(A_cell);
        A=A_cell[2,2];
        @tensor A[:]:=A[-1,-2,-3,1]*U_D_phy'[2,3,1]*H_triangle[3,4]*U_D_phy[-4,2,4];
        A_op_cell[2,2]=A;
        E_up=ob_1cell_closed(CTM,A_op_cell,method2)/norm_1cell;
        return E_up

    elseif method1=="E_bond"
        #calculate single unit-cell observable
        norm_1cell=ob_1cell_closed(CTM,A_cell,method2);#1 set of unitcell

        A_op_cell=deepcopy(A_cell);
        A=A_cell[2,2];
        @tensor A[:]:=A[-1,-2,-3,1]*U_D_phy'[2,3,1]*H12_tensorkit[3,4]*U_D_phy[-4,2,4];
        A_op_cell[2,2]=A;
        E_up_12=ob_1cell_closed(CTM,A_op_cell,method2)/norm_1cell;

        A_op_cell=deepcopy(A_cell);
        A=A_cell[2,2];
        @tensor A[:]:=A[-1,-2,-3,1]*U_D_phy'[2,3,1]*H31_tensorkit[3,4]*U_D_phy[-4,2,4];
        A_op_cell[2,2]=A;
        E_up_31=ob_1cell_closed(CTM,A_op_cell,method2)/norm_1cell;

        A_op_cell=deepcopy(A_cell);
        A=A_cell[2,2];
        @tensor A[:]:=A[-1,-2,-3,1]*U_D_phy'[2,3,1]*H23_tensorkit[3,4]*U_D_phy[-4,2,4];
        A_op_cell[2,2]=A;
        E_up_23=ob_1cell_closed(CTM,A_op_cell,method2)/norm_1cell;

        return E_up_12, E_up_31, E_up_23
    end
end






function Hamiltonians(U_phy,J1,J2,J3,Jchi,Jtrip)

    # Heisenberg interaction
    Id=I(2);
    sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
    @tensor H12[:]:=sx[-1,-4]*sx[-2,-5]*Id[-3,-6]+sy[-1,-4]*sy[-2,-5]*Id[-3,-6]+sz[-1,-4]*sz[-2,-5]*Id[-3,-6];
    @tensor H31[:]:=sx[-1,-4]*Id[-2,-5]*sx[-3,-6]+sy[-1,-4]*Id[-2,-5]*sy[-3,-6]+sz[-1,-4]*Id[-2,-5]*sz[-3,-6];
    @tensor H23[:]:=Id[-1,-4]*sx[-2,-5]*sx[-3,-6]+Id[-1,-4]*sy[-2,-5]*sy[-3,-6]+Id[-1,-4]*sz[-2,-5]*sz[-3,-6];
    @tensor H123chiral[:]:=sx[-1,-4]*sy[-2,-5]*sz[-3,-6]-sx[-1,-4]*sz[-2,-5]*sy[-3,-6]+sy[-1,-4]*sz[-2,-5]*sx[-3,-6]-sy[-1,-4]*sx[-2,-5]*sz[-3,-6]+sz[-1,-4]*sx[-2,-5]*sy[-3,-6]-sz[-1,-4]*sy[-2,-5]*sx[-3,-6];
    H12_tensorkit=J1*TensorMap(H12, domain(U_phy) ← domain(U_phy));
    H31_tensorkit=J1*TensorMap(H31, domain(U_phy) ← domain(U_phy));
    H23_tensorkit=J1*TensorMap(H23, domain(U_phy) ← domain(U_phy));
    H123chiral_tensorkit=Jtrip*TensorMap(H123chiral, domain(U_phy) ← domain(U_phy));
    @tensor H12_tensorkit[:]:=U_phy'[4,5,6,-1]*H12_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
    @tensor H31_tensorkit[:]:=U_phy'[4,5,6,-1]*H31_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
    @tensor H23_tensorkit[:]:=U_phy'[4,5,6,-1]*H23_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
    @tensor H123chiral_tensorkit[:]:=U_phy'[4,5,6,-1]*H123chiral_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];

    @tensor H_Heisenberg[:]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4];

    H_triangle=H12_tensorkit+H31_tensorkit+H23_tensorkit+H123chiral_tensorkit;
    H_bond=J1*H_Heisenberg;
    return H_triangle, H_bond, H12_tensorkit, H31_tensorkit, H23_tensorkit 
end


function ob_1cell_closed(CTM,A_cell,method)
    @assert (size(A_cell,1)==2);
    @assert (size(A_cell,2)==2);
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    
    if method=="full_cell"
        @tensor MM_LU[:]:=Cset[1][2,2][1,2]*Tset[1][1,2][2,3,-3]*Tset[4][2,1][-1,4,1]*A_cell[1,1][4,-2,-4,3]; 
        @tensor MM_RU[:]:=Tset[1][2,2][-1,3,1]* Cset[2][1,2][1,2]* A_cell[2,1][-2,-4,4,3]* Tset[2][1,1][2,4,-3];

        @tensor MM_LD[:]:=Tset[4][2,2][1,3,-1]*A_cell[1,2][3,4,-4,-2]*Cset[4][2,1][2,1]*Tset[3][1,1][-3,4,2]; 

        @tensor MM_RD[:]:=Tset[2][1,2][-4,-3,2]*Tset[3][2,1][1,-2,-1]*Cset[3][1,1][2,1]; 
        @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*A_cell[2,2][-2,1,2,-4]; 

        MM_LU=permute(MM_LU,(1,2,),(3,4,));
        MM_RU=permute(MM_RU,(1,2,),(3,4,));
        MM_LD=permute(MM_LD,(1,2,),(3,4,));
        MM_RD=permute(MM_RD,(1,2,),(3,4,));

        up=MM_LU*MM_RU;
        down=MM_LD*MM_RD;
        @tensor Norm[:]:=up[1,2,3,4]*down[1,2,3,4];
    elseif method=="reduced_cell"
        C1=Cset[1][1,1];
        C2=Cset[2][1,1];
        C3=Cset[3][1,1];
        C4=Cset[4][1,1];
        T1=Tset[1][2,1];
        T2=Tset[2][1,2];
        T3=Tset[3][2,1];
        T4=Tset[4][1,2];
        A=A_cell[2,2];
        @tensor Norm[:]:=C1[2,1]*C2[5,6]*C3[8,9]*C4[11,12]*T1[1,3,5]*T2[6,7,8]*T3[9,10,11]*T4[12,4,2]*A[4,10,7,3];
    end
    return blocks(Norm)[Irrep[SU₂](0)][1]
end



