using LinearAlgebra
using TensorKit


function evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, method)

    norm_1site=ob_1site_closed(CTM,AA_fused);
    H_triangle, H_bond, H12_tensorkit, H31_tensorkit, H23_tensorkit=Hamiltonians(U_phy,parameters["J1"],parameters["J2"],parameters["J3"],parameters["Jchi"],parameters["Jtrip"])
    
    AA1, U_ss=build_double_layer_open(A_unfused,"1",U_phy,U_L,U_D,U_R,U_U);
    AA2, U_ss=build_double_layer_open(A_unfused,"2",U_phy,U_L,U_D,U_R,U_U);
    AA3, U_ss=build_double_layer_open(A_unfused,"3",U_phy,U_L,U_D,U_R,U_U);

    if method=="E_triangle" #calculate up and down triangle energy
        AA_H, _,_,_,_=build_double_layer(A_fused,H_triangle);
        E_up=ob_1site_closed(CTM,AA_H)/norm_1site;
        E_up=E_up[1];

        rho_LU_RU_LD=ob_LU_RU_LD(CTM,AA_fused,AA2,AA1,AA3);
        rho_LU_RU_LD=permute(rho_LU_RU_LD,(1,3,2,),());#anti-clock-wise order
        @tensor rho_LU_RU_LD[:]:=U_ss[-1,-4,1]*U_ss[-2,-5,2]*U_ss[-3,-6,3]*rho_LU_RU_LD[1,2,3];
        @tensor rho_LU_RU_LD[:]:=U_phy'[4,5,6,-1]*rho_LU_RU_LD[4,5,6,1,2,3]*U_phy[-2,1,2,3];
        #@tensor e[:]:=rho_LU_RU_LD[1,2]*H_triangle[2,1];
        rho_LU_RU_LD=convert(Array,rho_LU_RU_LD);
        @tensor norm_LU_RU_LD[:]:=rho_LU_RU_LD[1,1];
        H_triangle=convert(Array,H_triangle);
        @tensor E_down[:]:=rho_LU_RU_LD[1,2]*H_triangle[2,1];
        E_down=E_down[1]/norm_LU_RU_LD[1];
        return E_up, E_down

    elseif method=="E_bond"
        #calculate single unit-cell observable
        AA_12_fused, _,_,_,_=build_double_layer(A_fused,H12_tensorkit);
        E_up_12=ob_1site_closed(CTM,AA_12_fused)/norm_1site;
        E_up_12=E_up_12[1];
        AA_31_fused, _,_,_,_=build_double_layer(A_fused,H31_tensorkit);
        E_up_31=ob_1site_closed(CTM,AA_31_fused)/norm_1site;
        E_up_31=E_up_31[1];
        AA_23_fused, _,_,_,_=build_double_layer(A_fused,H23_tensorkit);
        E_up_23=ob_1site_closed(CTM,AA_23_fused)/norm_1site;
        E_up_23=E_up_23[1];

        rhox=ob_2sites_x(CTM,AA2,AA1);
        rhoy=ob_2sites_y(CTM,AA2,AA3);
        rho_LD_RU=ob_LD_RU(CTM,AA_fused,AA3,AA1);

        @tensor rhox[:]:=U_ss[-1,-3,1]*U_ss[-2,-4,2]*rhox[1,2];
        @tensor rhoy[:]:=U_ss[-1,-3,1]*U_ss[-2,-4,2]*rhoy[1,2];
        @tensor rho_LD_RU[:]:=U_ss[-1,-3,1]*U_ss[-2,-4,2]*rho_LD_RU[1,2];

        rhox=convert(Array,rhox);
        rhoy=convert(Array,rhoy);
        rho_LD_RU=convert(Array,rho_LD_RU);

        @tensor norm_rhox[:]:=rhox[1,2,1,2];
        @tensor norm_rhoy[:]:=rhoy[1,2,1,2];
        @tensor norm_LD_RU[:]:=rho_LD_RU[1,2,1,2];

        H_bond=convert(Array,H_bond);
        @tensor E_down_12[:]:=rhox[1,2,3,4]*H_bond[3,4,1,2];
        E_down_12=E_down_12[1]/norm_rhox[1];

        @tensor E_down_23[:]:=rhoy[1,2,3,4]*H_bond[3,4,1,2];
        E_down_23=E_down_23[1]/norm_rhoy[1];

        @tensor E_down_31[:]:=rho_LD_RU[1,2,3,4]*H_bond[3,4,1,2];
        E_down_31=E_down_31[1]/norm_LD_RU[1];
        return E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23
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


function build_double_layer_open(A_unfused,inds,U_phy,U_L,U_D,U_R,U_U)


    #seperate the physical index that to be open
    if inds=="123"
        #display("this calculation is expensive and not verified")
        # A=permute(A,(1,2,3,),(4,5,));
        # V_D=space(A, 4);
        # V_DD=fuse(V_D' ⊗ V_D);
        # V_d=space(A, 5);
        # V_D_d=V_D ⊗ V_d;
        # V_Dd=fuse(V_D_d);
        # V_dd=fuse(V_d' ⊗ V_d);
        # U_Dd=unitary(V_Dd,V_D_d);
        # @tensor A_fused[:]:=A[-1,-2,-3,1,2]*U_Dd[-4,1,2];

        # A_fused=permute(A_fused,(1,2,),(3,4,));
        # uM,sM,vM=tsvd(A_fused);
        # uM=uM*sM
    
        # uM=permute(uM,(1,2,3,),());
        # V=space(vM,1);
        # U=unitary(fuse(V' ⊗ V), V' ⊗ V);
        # @tensor double_LD[:]:=uM'[-1,-2,1]*U'[1,-3,-4];
        # @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

        # vM=permute(vM,(1,2,3,),());
        # @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4];
        # @tensor double_RU[:]:=vM'[1,-2,-4]*double_RU[-1,1,-3,-5];

        # double_LD=permute(double_LD,(1,2,),(3,4,5,));
        # double_LD=U_L*double_LD;
        # double_LD=permute(double_LD,(2,3,),(1,4,));
        # double_LD=U_D*double_LD;
        # double_LD=permute(double_LD,(2,1,),(3,));
    
        # double_RU=permute(double_RU,(1,4,5,),(2,3,));
        # double_RU=double_RU*U_R;
        # double_RU=permute(double_RU,(1,4,),(2,3,));
    
        # U_Ud=unitary(space(U_U,3) ⊗ V_dd, V_Dd' ⊗ V_Dd);
        # @tensor double_RU[:]:=double_RU[-1,-2,1,2]*U_Ud[-3,-4,1,2];
        # double_RU=permute(double_RU,(1,),(2,3,4,));
        # AA_open_fused=permute(double_LD*double_RU,(1,2,3,4,5),());
        
        # U_dd=unitary(V_d' ⊗ V_d, V_dd);
        # return AA_open_fused, U_dd
    elseif inds in ["1","2","3"]
        #display(space(A))
        #A=permute(A,(1,2,3,),(4,5,));
        V_D=space(A_unfused, 4);
        V_s=space(A_unfused,7);

        if inds=="1"
            A_unfused=permute(A_unfused,(1,2,3,4,5,6,7,),());
        elseif inds=="2"
            A_unfused=permute(A_unfused,(1,2,3,4,6,5,7,),());
        elseif inds=="3"
            A_unfused=permute(A_unfused,(1,2,3,4,7,5,6,),());
        end
        fuse_spin=unitary(fuse(V_s ⊗ V_s), V_s ⊗ V_s);
        @tensor A_fused[:]:=A_unfused[-1,-2,-3,-4,-5,1,2]*fuse_spin[-6,1,2];
        V_ss=fuse(V_s' ⊗ V_s);

        uM_dag,sM_dag,vM_dag=tsvd(permute(A_fused',(1,2,3,),(4,5,6,)));
        uM_dag=uM_dag*sM_dag;
        uM,sM,vM=tsvd(permute(A_fused,(1,2,),(3,4,5,6,)));
        uM=uM*sM;
        
        uM_dag=permute(uM_dag,(1,2,3,4,),());
        uM=permute(uM,(1,2,3,),());
        Vp=space(vM_dag,1);
        V=space(vM,1);
        U=unitary(fuse(Vp ⊗ V), Vp ⊗ V);
        @tensor double_LD[:]:=uM_dag[-1,-2,-3,1]*U'[1,-4,-5];
        @tensor double_LD[:]:=double_LD[-1,-3,-5,1,-6]*uM[-2,-4,1];

        vM_dag=permute(vM_dag,(1,2,3,4,),());
        vM=permute(vM,(1,2,3,4,5,),());
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5,-6];
        @tensor double_RU[:]:=vM_dag[1,-3,-5,2]*double_RU[-1,1,-2,-4,-6,2];

        double_LD=permute(double_LD,(1,2,),(3,4,5,6,));
        double_LD=U_L*double_LD;
        double_LD=permute(double_LD,(2,3,),(1,4,5,));
        double_LD=U_D*double_LD;
        double_LD=permute(double_LD,(2,1,3,4),());
    
        double_RU=permute(double_RU,(1,2,5,6,),(3,4,));
        double_RU=double_RU*U_U;
        U_s_s=unitary(V_ss, V_s' ⊗ V_s);
        @tensor double_RU[:]:=double_RU[-1,-2,1,2,-3]*U_s_s[-4,1,2];

        A_fused=[];#clear memory
        A_unfused=[];#clear memory
        @tensor double_RU[:]:=double_RU[-1,1,-4,-5]*U_R[-2,1,-3];

        @tensor AA_open_fused[:]:=double_LD[-1,-2,2,1]*double_RU[1,2,-3,-4,-5];

        return AA_open_fused, U_s_s'

    end
end

function ob_1site_closed(CTM,AA_fused)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    @tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
    @tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA_fused[2,5,-2,3]*Tset[3][-3,5,4];
    @tensor Norm[:]:=envL[1,2,3]*envR[1,2,3];
    Norm=blocks(Norm)[Irrep[SU₂](0)];
    return Norm;
end


function ob_2sites_x(CTM,AA1,AA2)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    @tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
    @tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA1[2,5,-2,3,-4]*Tset[3][-3,5,4];
    @tensor envR[:]:=Tset[1][-1,3,1]*AA2[-2,5,2,3,-4]*Tset[3][4,5,-3]*envR[1,2,4];
    @tensor rho[:]:=envL[1,2,3,-1]*envR[1,2,3,-2];
    return rho;
end


function ob_2sites_y(CTM,AA1,AA2)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    @tensor envU[:]:=Cset[2][1,-1]*Tset[1][2,-2,1]*Cset[1][-3,2];
    @tensor envD[:]:=Cset[3][-1,1]*Tset[3][1,-2,2]*Cset[4][2,-3];
    @tensor envU[:]:=envU[1,2,4]*Tset[2][1,3,-1]*AA1[5,-2,3,2,-4]*Tset[4][-3,5,4];
    @tensor envD[:]:=Tset[2][-1,3,1]*AA2[5,2,3,-2,-4]*Tset[4][4,5,-3]*envD[1,2,4];
    @tensor rho[:]:=envU[1,2,3,-1]*envD[1,2,3,-2];
    return rho;
end


function ob_LD_RU(CTM,AA_fused,AA_LD,AA_RU)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_RU[-2,-4,4,3,-5]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset[4][2,1]*Tset[3][-4,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[1,2,3,4,-1]*down[-2,1,2,3,4];
end

function ob_LU_RD(CTM,AA_fused,AA_LU,AA_RD)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-4]*Tset[4][-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_fused[-2,-4,4,3]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-1]*AA_fused[3,4,-4,-2]*Cset[4][2,1]*Tset[3][-3,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[-1,1,2,3,4]*down[1,2,3,4,-2];
end


function ob_LU_RU_LD(CTM,AA_fused,AA_LU,AA_RU,AA_LD)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-4]*Tset[4][-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_RU[-2,-4,4,3,-5]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset[4][2,1]*Tset[3][-4,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[-1,1,2,3,4,-2]*down[-3,1,2,3,4];
end

function ob_LD_RU_RD(CTM,AA_fused,AA_LD,AA_RU,AA_RD)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];

    @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_RU[-2,-4,4,3,-5]* Tset[2][2,4,-3];

    @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset[4][2,1]*Tset[3][-4,4,2]; 
    @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:=up[1,2,3,4,-2]*down[-1,1,2,3,4,-3];
end