using LinearAlgebra
using TensorKit
using Zygote:@ignore_derivatives

function evaluate_NN(A::TensorMap, AA, U_L,U_D,U_R,U_U, CTM, ctm_setting)

    AA_open, U_s_s=build_double_layer_open(A);

    H_Heisenberg, H123chiral, H12, H31, H23 =@ignore_derivatives Hamiltonians();

    rhox=ob_2sites_x(CTM,AA_open,AA_open);
    @tensor rhox[:]:=rhox[1,2]*U_s_s[-1,-3,1]*U_s_s[-2,-4,2];#s1',s2',s1,s2
    norm_x=@tensor rhox[1,2,1,2];
    Ex=@tensor rhox[1,2,3,4]*H_Heisenberg[1,2,3,4];
    Ex=Ex/norm_x; 
    
    rhoy=ob_2sites_y(CTM,AA_open,AA_open);
    @tensor rhoy[:]:=rhoy[1,2]*U_s_s[-1,-3,1]*U_s_s[-2,-4,2];#s1',s2',s1,s2
    norm_y=@tensor rhoy[1,2,1,2];
    Ey=@tensor rhoy[1,2,3,4]*H_Heisenberg[1,2,3,4];
    Ey=Ey/norm_y; 
    
    return Ex,Ey
end

function evaluate_NNN(A::TensorMap, AA, U_L,U_D,U_R,U_U, CTM, ctm_setting)

    AA_open, U_s_s=build_double_layer_open(A);

    H_Heisenberg, H123chiral, H12, H31, H23 =@ignore_derivatives Hamiltonians();

    
    rho_LD_RU=ob_LD_RU(CTM,AA,AA_open,AA_open);  
    @tensor rho_LD_RU[:]:=rho_LD_RU[1,2]*U_s_s[-1,-3,1]*U_s_s[-2,-4,2];#s1',s2',s1,s2
    norm_LD_RU=@tensor rho_LD_RU[1,2,1,2];
    E_LD_RU=@tensor rho_LD_RU[1,2,3,4]*H_Heisenberg[1,2,3,4];
    E_LD_RU=E_LD_RU/norm_LD_RU; 
    

    rho_LU_RD=ob_LU_RD(CTM,AA,AA_open,AA_open);
    @tensor rho_LU_RD[:]:=rho_LU_RD[1,2]*U_s_s[-1,-3,1]*U_s_s[-2,-4,2];#s1',s2',s1,s2
    norm_LU_RD=@tensor rho_LU_RD[1,2,1,2];
    E_LU_RD=@tensor rho_LU_RD[1,2,3,4]*H_Heisenberg[1,2,3,4];
    E_LU_RD=E_LU_RD/norm_LU_RD;

    return E_LD_RU,E_LU_RD
end

function evaluate_chirality(A::TensorMap, AA, U_L,U_D,U_R,U_U, CTM, ctm_setting)

    AA_open, U_s_s=build_double_layer_open(A);

    H_Heisenberg, H123chiral, H12, H31, H23 =@ignore_derivatives Hamiltonians();

    rho_LU_RU_LD=ob_LU_RU_LD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
    @tensor rho_LU_RU_LD[:]:=rho_LU_RU_LD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
    norm_LU_RU_LD=@tensor rho_LU_RU_LD[1,2,3,1,2,3];
    E_LU_RU_LD=@tensor rho_LU_RU_LD[1,2,3,4,5,6]*H123chiral[1,2,3,4,5,6];
    E_LU_RU_LD=E_LU_RU_LD/norm_LU_RU_LD; 
    

    rho_LD_RU_RD=ob_LD_RU_RD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
    rho_LD_RU_RD=permute(rho_LD_RU_RD,(3,1,2,));
    @tensor rho_LD_RU_RD[:]:=rho_LD_RU_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
    norm_LD_RU_RD=@tensor rho_LD_RU_RD[1,2,3,1,2,3];
    E_LD_RU_RD=@tensor rho_LD_RU_RD[1,2,3,4,5,6]*H123chiral[1,2,3,4,5,6];
    E_LD_RU_RD=E_LD_RU_RD/norm_LD_RU_RD ; 
    

    rho_LU_LD_RD=ob_LU_LD_RD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
    rho_LU_LD_RD=permute(rho_LU_LD_RD,(2,1,3,));
    @tensor rho_LU_LD_RD[:]:=rho_LU_LD_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
    norm_LU_LD_RD=@tensor rho_LU_LD_RD[1,2,3,1,2,3];
    E_LU_LD_RD=@tensor rho_LU_LD_RD[1,2,3,4,5,6]*H123chiral[1,2,3,4,5,6];
    E_LU_LD_RD=E_LU_LD_RD/norm_LU_LD_RD ; 
    

    rho_LU_RU_RD=ob_LU_RU_RD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
    rho_LU_RU_RD=permute(rho_LU_RU_RD,(2,3,1,));
    @tensor rho_LU_RU_RD[:]:=rho_LU_RU_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
    norm_LU_RU_RD=@tensor rho_LU_RU_RD[1,2,3,1,2,3];
    E_LU_RU_RD=@tensor rho_LU_RU_RD[1,2,3,4,5,6]*H123chiral[1,2,3,4,5,6];
    E_LU_RU_RD=E_LU_RU_RD/norm_LU_RU_RD;
        
    return E_LU_RU_LD, E_LD_RU_RD, E_LU_LD_RD, E_LU_RU_RD
end

function evaluate_triangle(H_triangle, A::TensorMap, AA, U_L,U_D,U_R,U_U, CTM, ctm_setting)

    AA_open, U_s_s=build_double_layer_open(A);

    

    rho_LU_RU_LD=ob_LU_RU_LD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
    @tensor rho_LU_RU_LD[:]:=rho_LU_RU_LD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
    norm_LU_RU_LD=@tensor rho_LU_RU_LD[1,2,3,1,2,3];
    E_LU_RU_LD=@tensor rho_LU_RU_LD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
    E_LU_RU_LD=E_LU_RU_LD/norm_LU_RU_LD; 
    

    rho_LD_RU_RD=ob_LD_RU_RD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
    rho_LD_RU_RD=permute(rho_LD_RU_RD,(3,1,2,));
    @tensor rho_LD_RU_RD[:]:=rho_LD_RU_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
    norm_LD_RU_RD=@tensor rho_LD_RU_RD[1,2,3,1,2,3];
    E_LD_RU_RD=@tensor rho_LD_RU_RD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
    E_LD_RU_RD=E_LD_RU_RD/norm_LD_RU_RD ; 
    

    rho_LU_LD_RD=ob_LU_LD_RD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
    rho_LU_LD_RD=permute(rho_LU_LD_RD,(2,1,3,));
    @tensor rho_LU_LD_RD[:]:=rho_LU_LD_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
    norm_LU_LD_RD=@tensor rho_LU_LD_RD[1,2,3,1,2,3];
    E_LU_LD_RD=@tensor rho_LU_LD_RD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
    E_LU_LD_RD=E_LU_LD_RD/norm_LU_LD_RD ; 
    

    rho_LU_RU_RD=ob_LU_RU_RD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
    rho_LU_RU_RD=permute(rho_LU_RU_RD,(2,3,1,));
    @tensor rho_LU_RU_RD[:]:=rho_LU_RU_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
    norm_LU_RU_RD=@tensor rho_LU_RU_RD[1,2,3,1,2,3];
    E_LU_RU_RD=@tensor rho_LU_RU_RD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
    E_LU_RU_RD=E_LU_RU_RD/norm_LU_RU_RD;
        
    return E_LU_RU_LD, E_LD_RU_RD, E_LU_LD_RD, E_LU_RU_RD
end




function build_double_layer_open(A0)

    A_=permute(A0,(1,2,),(3,4,5));
    U_L=@ignore_derivatives unitary(fuse(space(A_, 1)' ⊗ space(A_, 1)), space(A_, 1)' ⊗ space(A_, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A_, 2)' ⊗ space(A_, 2)), space(A_, 2)' ⊗ space(A_, 2))*(1+0*im);
    # U_R=(U_L)';
    # U_U=(U_D)';
    U_R=@ignore_derivatives unitary(space(A_, 3) ⊗ space(A_, 3)', fuse(space(A_, 3)' ⊗ space(A_, 3)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A_, 4) ⊗ space(A_, 4)', fuse(space(A_, 4)' ⊗ space(A_, 4)))*(1+0*im);

    V_D=@ignore_derivatives space(A0, 4);
    V_s=@ignore_derivatives space(A0, 5);

    A=permute(A0,(1,2,3,4,5,));

    V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);

    # uM_dag,sM_dag,vM_dag=tsvd(permute(A_fused',(1,2,3,),(4,5,6,)));
    # uM_dag=uM_dag*sM_dag;
    Ap=permute(A0',(1,2,5,),(3,4,));
    Up_tem=@ignore_derivatives unitary(fuse(space(Ap,1)*space(Ap,2)*space(Ap,3)), space(Ap,1)*space(Ap,2)*space(Ap,3))*(1+0*im);
    vM_dag=Up_tem*Ap;
    uM_dag=Up_tem';


    U_tem=@ignore_derivatives unitary(fuse(space(A0,1)*space(A0,2)), space(A0,1)*space(A0,2))*(1+0*im);
    vM=U_tem*permute(A0,(1,2,),(3,4,5,));
    uM=U_tem';

    
    uM_dag=permute(uM_dag,(1,2,3,4,),());
    uM=permute(uM,(1,2,3,),());
    Vp=space(vM_dag,1);
    V=space(vM,1);
    U=@ignore_derivatives unitary(fuse(Vp ⊗ V), Vp ⊗ V);
    @tensor double_LD[:]:=uM_dag[-1,-2,-3,1]*U'[1,-4,-5];
    @tensor double_LD[:]:=double_LD[-1,-3,-5,1,-6]*uM[-2,-4,1];

    vM_dag=permute(vM_dag,(1,2,3,),());
    vM=permute(vM,(1,2,3,4,));
    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    @tensor double_RU[:]:=vM_dag[1,-2,-4]*double_RU[-1,1,-3,-5,-6];

    double_LD=permute(double_LD,(1,2,),(3,4,5,6,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,5,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,3,4),());#L,D,physical,virtual

    double_RU=permute(double_RU,(1,2,3,6,),(4,5,));
    double_RU=double_RU*U_U;
    @tensor double_RU[:]:=double_RU[-1,1,2,-4,-3]*U_R[1,2,-2];

    U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);


    @tensor AA_open[:]:=double_LD[-1,-2,1,3]*double_RU[3,-3,-4,2]*U_s_s[-5,1,2];

    U_s_s=U_s_s';

    return AA_open, U_s_s 

end

function ob_1site_closed(CTM,A_fused,AA_fused,op,construct_double_layer)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    if construct_double_layer
        @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
        @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
        @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA_fused[2,5,-2,3]*Tset.T3[-3,5,4];
        Norm=@tensor envL[1,2,3]*envR[1,2,3];
    else
        if op==[]
            Ap=A_fused';
        else
            @tensor Ap[:]:=A_fused'[-1,-2,-3,-4,1]*op[-5,1];
        end
        @tensor MLU[:]:=Cset.C1[1,2]*Tset.T1[2,6,4,-4]*Tset.T4[-1,5,3,1]*A_fused[3,-3,-6,4,7]*Ap[5,-2,-5,6,7];

         Norm=@tensor MLU[7,8,9,4,5,6]*Cset.C2[4,3]*Tset.T2[3,5,6,10]*Cset.C3[10,2]*Tset.T3[2,8,9,1]*Cset.C4[1,7];
    end


    return Norm;
end


function ob_2sites_x(CTM,AA1,AA2)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA1[2,5,-2,3,-4]*Tset.T3[-3,5,4];
    @tensor envR[:]:=Tset.T1[-1,3,1]*AA2[-2,5,2,3,-4]*Tset.T3[4,5,-3]*envR[1,2,4];
    @tensor rho[:]:=envL[1,2,3,-1]*envR[1,2,3,-2];
    return rho;
end


function ob_2sites_y(CTM,AA1,AA2)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envU[:]:=Cset.C2[1,-1]*Tset.T1[2,-2,1]*Cset.C1[-3,2];
    @tensor envD[:]:=Cset.C3[-1,1]*Tset.T3[1,-2,2]*Cset.C4[2,-3];
    @tensor envU[:]:=envU[1,2,4]*Tset.T2[1,3,-1]*AA1[5,-2,3,2,-4]*Tset.T4[-3,5,4];
    @tensor envD[:]:=Tset.T2[-1,3,1]*AA2[5,2,3,-2,-4]*Tset.T4[4,5,-3]*envD[1,2,4];
    @tensor rho[:]:=envU[1,2,3,-1]*envD[1,2,3,-2];
    return rho;
end




function ob_LD_RU(CTM,AA_fused,AA_LD,AA_RU)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset.C4[2,1]*Tset.T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[1,2,3,4,-2]*down[-1,1,2,3,4];
    return rho
end

function ob_LU_RD(CTM,AA_fused,AA_LU,AA_RD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-4]*Tset.T4[-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_fused[3,4,-4,-2]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[-1,1,2,3,4]*down[1,2,3,4,-2];
    return rho
end


function ob_LU_RU_LD(CTM,AA_fused,AA_LU,AA_RU,AA_LD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-4]*Tset.T4[-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset.C4[2,1]*Tset.T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[-1,1,2,3,4,-2]*down[-3,1,2,3,4];
    return rho
end


function ob_LD_RU_RD(CTM,AA_fused,AA_LD,AA_RU,AA_RD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset.C4[2,1]*Tset.T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[1,2,3,4,-2]*down[-1,1,2,3,4,-3];
    return rho
end


function ob_LU_LD_RD(CTM,AA_fused,AA_LU,AA_LD,AA_RD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-4]*Tset.T4[-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset.C4[2,1]*Tset.T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[-1,1,2,3,4]*down[-2,1,2,3,4,-3];
    return rho
end



function ob_LU_RU_RD(CTM,AA_fused,AA_LU,AA_RU,AA_RD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-4]*Tset.T4[-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-2]*AA_fused[3,4,-5,-3]*Cset.C4[2,1]*Tset.T3[-4,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

    MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    @tensor rho[:]:= up[-1,1,2,3,4,-2]*down[1,2,3,4,-3];
    return rho
end