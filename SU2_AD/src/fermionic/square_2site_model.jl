function spin_op()
    Pm=zeros(4,4)*im;Pm[1,1]=1;Pm[2,4]=1;Pm[3,2]=1;Pm[4,3]=1;
    V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(1,1/2)=>1,(2,0)=>1);#element order after converting to dense: <0,0>, <up,down>, <up,0>, <0,down>, 

    #spin-spin correlation operator 
    SS=zeros(2,2,2,2,2,2,2,2)*im;#Aup,Adown, Bup,Bdown
    SS[2,1,1,2,1,2,2,1]=1/2;#spsm
    SS[1,2,2,1,2,1,1,2]=1/2;#smmsp
    SS[2,1,2,1,2,1,2,1]=1/4;#szsz
    SS[2,1,1,2,2,1,1,2]=-1/4;#szsz
    SS[1,2,2,1,1,2,2,1]=-1/4;#szsz
    SS[1,2,1,2,1,2,1,2]=1/4;#szsz
    SS=reshape(SS,4,4,4,4);
    @tensor SS[:]:=SS[1,2,3,4]*Pm[-1,1]*Pm[-2,2]*Pm[-3,3]*Pm[-4,4];
    SS_op_F=TensorMap(SS, V' ⊗ V' ← V' ⊗ V');

    #Si dot Sj cross Sk
    SS=zeros(2,2,2,2,2,2,2,2,2,2,2,2)*im;#Aup,Adown, Bup,Bdown, Cup,Cdown
    SS[2,1,1,2,2,1, 2,1,2,1,1,2]=im/4;
    SS[1,2,2,1,2,1, 2,1,2,1,1,2]=-im/4;

    SS[1,2,2,1,2,1, 2,1,1,2,2,1]=im/4;
    SS[2,1,2,1,1,2, 2,1,1,2,2,1]=-im/4;

    SS[2,1,2,1,1,2, 1,2,2,1,2,1]=im/4;
    SS[2,1,1,2,2,1, 1,2,2,1,2,1]=-im/4;

    SS[1,2,2,1,1,2, 1,2,1,2,2,1]=im/4;
    SS[2,1,1,2,1,2, 1,2,1,2,2,1]=-im/4;

    SS[2,1,1,2,1,2, 1,2,2,1,1,2]=im/4;
    SS[1,2,1,2,2,1, 1,2,2,1,1,2]=-im/4;

    SS[1,2,1,2,2,1, 2,1,1,2,1,2]=im/4;
    SS[1,2,2,1,1,2, 2,1,1,2,1,2]=-im/4;

    SS=reshape(SS,4,4,4,4,4,4);
    @tensor SS[:]:=SS[1,2,3,4,5,6]*Pm[-1,1]*Pm[-2,2]*Pm[-3,3]*Pm[-4,4]*Pm[-5,5]*Pm[-6,6];
    Schiral_op_F=TensorMap(SS, V' ⊗ V' ⊗ V' ← V' ⊗ V' ⊗ V');


    #gutzwiller projector
    P=zeros(2,2,2)*im;
    P[1,2,1]=1;
    P[2,1,2]=1;
    P=reshape(P,2,4);
    @tensor P[:]:=P[-1,1]*Pm[-2,1];
    Vspin=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1);
    P_G=TensorMap(P, Vspin'  ←  V');

    @tensor SS_op_S[:]:=P_G[-1,1]*P_G[-2,2]*SS_op_F[1,2,3,4]*P_G'[3,-3]*P_G'[4,-4];
    SS_op_S=permute(SS_op_S,(1,2,),(3,4,))

    @tensor Schiral_op_S[:]:=P_G[-1,1]*P_G[-2,2]*P_G[-3,3]*Schiral_op_F[1,2,3,4,5,6]*P_G'[4,-4]*P_G'[5,-5]*P_G'[6,-6];
    Schiral_op_S=permute(Schiral_op_S,(1,2,3,),(4,5,6,));
    return SS_op_S,Schiral_op_S
end

function Hamiltonian_terms_2site()
    SS_op,Schiral_op=spin_op();

    v1=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((6,0)=>1)*GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1)'*GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1)';
    U_phy1=unitary(fuse(v1),v1);
    v2=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((4,0)=>1,(4,1)=>1)*GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,0)=>1)*GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,0)=>1);
    U_phy2=unitary(fuse(v2),v2);



    
    #println("construct physical operators");flush(stdout);
    #spin-spin operator act on a single site
    um,sm,vm=@ignore_derivatives tsvd(permute(SS_op,(1,3,),(2,4,)));
    vm=sm*vm;vm=permute(vm,(2,3,),(1,));

    @tensor SS_cell[:]:=SS_op[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];#spin-spin operator inside a unitcell

    @tensor S_R_a[:]:=um[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_L_a[:]:=um[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor S_R_b[:]:=vm[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor S_L_b[:]:=vm[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    
       
    @tensor SS_cell[:]:=SS_cell[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_L_a[:]:=S_L_a[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_R_a[:]:=S_R_a[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_L_b[:]:=S_L_b[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_R_b[:]:=S_R_b[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];




    ####################################
    #chiral operator act on a single site: Si Sj Sk
    um,sm,vm=@ignore_derivatives tsvd(Schiral_op,(1,4,),(2,3,5,6,));
    vm=sm*vm;
    Si=permute(um,(1,2,3,));#P,P',D1
    um,sm,vm=@ignore_derivatives tsvd(vm,(1,2,4,),(3,5,));
    vm=sm*vm;
    Sj=permute(um,(2,3,1,4,));#P,P', D1,D2
    Sk=permute(vm,(2,3,1,))#P,P',D2
    @tensor SiSj[:]:=Si[-1,-3,1]*Sj[-2,-4,1,-5]; 
    @tensor SjSk[:]:=Sj[-1,-3,-5,1]*Sk[-2,-4,1]; 
    #@tensor aa[:]:=Si[-1,-4,1]*Sj[-2,-5,1,2]*Sk[-3,-6,2];
    U_Schiral=unitary(fuse(space(Sj,3)⊗space(Sj,4)), space(Sj,3)⊗space(Sj,4));
    @tensor Sj[:]:=Sj[-1,-2,1,2]*U_Schiral[-3,1,2];#combine two extra indices of Sj



    @tensor Si_right[:]:=Si[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Si_left[:]:=Si[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sj_right[:]:=Sj[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sj_left[:]:=Sj[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];
    @tensor Sk_right[:]:=Sk[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];
    @tensor Sk_left[:]:=Sk[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];

    @tensor SjSi_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SiSj_op[:]:=SiSj[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];
    @tensor SkSj_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,2,3]*U_phy1'[1,4,5,-2];
    @tensor SjSk_op[:]:=SjSk[2,3,4,5,-3]*U_phy1[-1,1,3,2]*U_phy1'[1,5,4,-2];



    @tensor Si_left[:]:=Si_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Si_right[:]:=Si_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sj_left[:]:=Sj_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sj_right[:]:=Sj_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sk_left[:]:=Sk_left[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor Sk_right[:]:=Sk_right[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SiSj_op[:]:=SiSj_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SjSi_op[:]:=SjSi_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SjSk_op[:]:=SjSk_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor SkSj_op[:]:=SkSj_op[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

    return SS_cell, S_L_a, S_R_a, S_L_b, S_R_b, Si_left, Si_right, Sj_left, Sj_right, Sk_left, Sk_right, SiSj_op, SjSi_op, SjSk_op, SkSj_op,U_Schiral
end

function energy_2site(parameter,A_fused,AA_fused,CTM)
    SS_cell, S_L_a, S_R_a, S_L_b, S_R_b, Si_left, Si_right, Sj_left, Sj_right, Sk_left, Sk_right, SiSj_op, SjSi_op, SjSk_op, SkSj_op,U_Schiral=@ignore_derivatives Hamiltonian_terms_2site();

    #println("construct double layer tensor with operator");flush(stdout);
    AA_SS=build_double_layer_swap_op(A_fused,SS_cell,false);
    AA_SLa=build_double_layer_swap_op(A_fused,S_L_a,true);
    AA_SRa=build_double_layer_swap_op(A_fused,S_R_a,true);
    AA_SLb=build_double_layer_swap_op(A_fused,S_L_b,true);
    AA_SRb=build_double_layer_swap_op(A_fused,S_R_b,true);

    AA_SiL=build_double_layer_swap_op(A_fused,Si_left,true);
    AA_SiR=build_double_layer_swap_op(A_fused,Si_right,true);
    AA_SjL=build_double_layer_swap_op(A_fused,Sj_left,true);
    AA_SjR=build_double_layer_swap_op(A_fused,Sj_right,true);
    AA_SkL=build_double_layer_swap_op(A_fused,Sk_left,true);
    AA_SkR=build_double_layer_swap_op(A_fused,Sk_right,true);
    AA_SiSj=build_double_layer_swap_op(A_fused,SiSj_op,true);
    AA_SjSi=build_double_layer_swap_op(A_fused,SjSi_op,true);
    AA_SjSk=build_double_layer_swap_op(A_fused,SjSk_op,true);
    AA_SkSj=build_double_layer_swap_op(A_fused,SkSj_op,true);


    #println("Calculate energy terms:");flush(stdout);


    Norm_1=ob_1site_closed(CTM,AA_fused)
    Norm_2x=norm_2sites_x(CTM,AA_fused)
    Norm_2y=norm_2sites_y(CTM,AA_fused)
    Norm_4=norm_2x2(CTM,AA_fused);

    #J1 term
    E_1_a=ob_1site_closed(CTM,AA_SS)/Norm_1
    E_1_b=ob_2sites_x(CTM,AA_SRa,AA_SLb)/Norm_2x
    E_1_c=ob_2sites_y(CTM,AA_SLa,AA_SLb)/Norm_2y
    E_1_d=ob_2sites_y(CTM,AA_SRa,AA_SRb)/Norm_2y

    #J2 term
    E_2_a=ob_2sites_y(CTM,AA_SLa,AA_SRb)/Norm_2y
    E_2_b=ob_2sites_y(CTM,AA_SRa,AA_SLb)/Norm_2y
    E_2_c=ob_LU_RD(CTM,AA_fused,AA_SRa,AA_SLb)/Norm_4
    E_2_d=ob_RU_LD(CTM,AA_fused,AA_SLa,AA_SRb)/Norm_4



    #chiral term
    E_C_a=ob_2sites_y(CTM,AA_SiSj,AA_SkR)/Norm_2y
    E_C_b=ob_2sites_y(CTM,AA_SiR,AA_SkSj)/Norm_2y
    E_C_c=ob_2sites_y(CTM,AA_SjSk,AA_SiL)/Norm_2y
    E_C_d=ob_2sites_y(CTM,AA_SiSj,AA_SkL)/Norm_2y

    E_C_e=ob_LD_LU_RU(CTM,AA_fused,AA_SiR,AA_SjR,AA_SkL,U_Schiral)/Norm_4
    E_C_f=ob_LU_RU_RD(CTM,AA_fused,AA_SiR,AA_SjL,AA_SkL,U_Schiral)/Norm_4
    E_C_g=ob_RU_RD_LD(CTM,AA_fused,AA_SiL,AA_SjL,AA_SkR,U_Schiral)/Norm_4
    E_C_h=ob_RD_LD_LU(CTM,AA_fused,AA_SiL,AA_SjR,AA_SkR,U_Schiral)/Norm_4

    # println("J1 terms:")
    # println(E_1_a)
    # println(E_1_b)
    # println(E_1_c)
    # println(E_1_d)
    # println("J2 terms:")
    # println(E_2_a)
    # println(E_2_b)
    # println(E_2_c)
    # println(E_2_d)
    # println("Jchi terms:")
    # println(E_C_a)
    # println(E_C_b)
    # println(E_C_c)
    # println(E_C_d)
    # println(E_C_e)
    # println(E_C_f)
    # println(E_C_g)
    # println(E_C_h)


    E1=E_1_a+E_1_b+E_1_c+E_1_d;
    E1=E1/4;
    E2=E_2_a+E_2_b+E_2_c+E_2_d;
    E2=E2/4;
    EC=E_C_a+E_C_b+E_C_c+E_C_d+E_C_e+E_C_f+E_C_g+E_C_h;  
    EC=EC/8*4;#plaquette

    J1=parameters["J1"];
    J2=parameters["J2"];
    Jchi=parameters["Jchi"];

    E=J1*E1*2+J2*E2*2+Jchi*(EC);
    return E
end

function ob_RU_LD(CTM,AA_fused,AA_RU,AA_LD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 


    @tensor up[:]:=MM_LU[-1,-2,1,2]*MM_RU[1,2,-3,-4,-5];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];

    return ob
end


function ob_1site_closed(CTM,AA_fused)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA_fused[2,5,-2,3]*Tset.T3[-3,5,4];
    ob=@tensor envL[1,2,3]*envR[1,2,3];
    return ob;
end

function norm_2sites_x(CTM,AA_fused)

    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA_fused[2,5,-2,3]*Tset.T3[-3,5,4];
    @tensor envR[:]:=Tset.T1[-1,3,1]*AA_fused[-2,5,2,3]*Tset.T3[4,5,-3]*envR[1,2,4];
    ob=@tensor envL[1,2,3]*envR[1,2,3];
    return ob
end

function ob_2sites_x(CTM,AA1,AA2)

    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envL[:]:=Cset.C1[1,-1]*Tset.T4[2,-2,1]*Cset.C4[-3,2];
    @tensor envR[:]:=Cset.C2[-1,1]*Tset.T2[1,-2,2]*Cset.C3[2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset.T1[1,3,-1]*AA1[2,5,-2,3,-4]*Tset.T3[-3,5,4];
    @tensor envR[:]:=Tset.T1[-1,3,1]*AA2[-2,5,2,3,-4]*Tset.T3[4,5,-3]*envR[1,2,4];
    ob=@tensor envL[1,2,3,4]*envR[1,2,3,4];
    return ob
end

function norm_2sites_y(CTM,AA_fused)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envU[:]:=Cset.C2[1,-1]*Tset.T1[2,-2,1]*Cset.C1[-3,2];
    @tensor envD[:]:=Cset.C3[-1,1]*Tset.T3[1,-2,2]*Cset.C4[2,-3];
    @tensor envU[:]:=envU[1,2,4]*Tset.T2[1,3,-1]*AA_fused[5,-2,3,2]*Tset.T4[-3,5,4];
    @tensor envD[:]:=Tset.T2[-1,3,1]*AA_fused[5,2,3,-2]*Tset.T4[4,5,-3]*envD[1,2,4];
    ob=@tensor envU[1,2,3]*envD[1,2,3];
    return ob
end

function ob_2sites_y(CTM,AA1,AA2)
    Cset=CTM.Cset;
    Tset=CTM.Tset;
    @tensor envU[:]:=Cset.C2[1,-1]*Tset.T1[2,-2,1]*Cset.C1[-3,2];
    @tensor envD[:]:=Cset.C3[-1,1]*Tset.T3[1,-2,2]*Cset.C4[2,-3];
    @tensor envU[:]:=envU[1,2,4]*Tset.T2[1,3,-1]*AA1[5,-2,3,2,-4]*Tset.T4[-3,5,4];
    @tensor envD[:]:=Tset.T2[-1,3,1]*AA2[5,2,3,-2,-4]*Tset.T4[4,5,-3]*envD[1,2,4];
    ob=@tensor envU[1,2,3,4]*envD[1,2,3,4];
    return ob
end

function norm_2x2(CTM,AA_fused)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_fused[3,4,-4,-2]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    ob=@tensor up[1,2,3,4]*down[1,2,3,4];
    return ob
end

function ob_LD_LU_RU(CTM,AA_fused,AA_LD,AA_LU,AA_RU,U_chiral)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_LU[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 


    @tensor up[:]:=U_chiral'[-5,4,1]*MM_LU[-1,-2,2,3,1]*MM_RU[2,3,-3,-4,4];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end


function ob_LU_RU_RD(CTM,AA_fused,AA_LU,AA_RU,AA_RD,U_chiral)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_LU[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_fused[3,4,-4,-2]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 


    @tensor up[:]:=MM_LU[-1,-2,2,3,4]*MM_RU[2,3,-3,-4,1]*U_chiral'[4,-5,1];
    @tensor down[:]:=MM_LD[-1,-2,1,2]*MM_RD[1,2,-3,-4,-5];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end


function ob_RU_RD_LD(CTM,AA_fused,AA_RU,AA_RD,AA_LD,U_chiral)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 


    @tensor up[:]:=MM_LU[-1,-2,1,2]*MM_RU[1,2,-3,-4,-5];
    @tensor down[:]:=MM_LD[-1,-2,2,3,4]*MM_RD[2,3,-3,-4,1]*U_chiral'[-5,4,1];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end

function ob_RD_LD_LU(CTM,AA_fused,AA_RD,AA_LD,AA_LU,U_chiral)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_LU[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5];


    @tensor up[:]:=MM_LU[-1,-2,1,2,-5]*MM_RU[1,2,-3,-4];
    @tensor down[:]:=U_chiral'[4,-5,1]*MM_LD[-1,-2,2,3,1]*MM_RD[2,3,-3,-4,4];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end


function ob_LU_RD(CTM,AA_fused,AA_LU,AA_RD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_LU[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_fused[3,4,-4,-2]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5];


    @tensor up[:]:=MM_LU[-1,-2,1,2,-5]*MM_RU[1,2,-3,-4];
    @tensor down[:]:=MM_LD[-1,-2,1,2]*MM_RD[1,2,-3,-4,-5];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end


function ob_RU_LD(CTM,AA_fused,AA_RU,AA_LD)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 


    @tensor up[:]:=MM_LU[-1,-2,1,2]*MM_RU[1,2,-3,-4,-5];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4];
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
    return ob
end



# function evaluate_correl_spinspin(direction, AA_fused, AA_op1, AA_op2, CTM, method, distance)
#     correl_funs=Vector(undef,distance);

#     C1=CTM.Cset.C1;
#     C2=CTM.Cset.C2;
#     C3=CTM.Cset.C3;
#     C4=CTM.Cset.C4;
#     T1=CTM.Tset.T1;
#     T2=CTM.Tset.T2;
#     T3=CTM.Tset.T3;
#     T4=CTM.Tset.T4;
#     if method=="dimerdimer"#operator on a single site conserves su2 symmetry
#         if direction=="x"
#             @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4]*T3[-3,6,7];
#             @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
#             ov=@tensor va[1,2,3]*vb[1,2,3]
#             correl_funs[1]=ov;
            
#             for dis=2:distance
#                 @tensor va[:]:=va[1,3,5]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
#                 ov=@tensor va[1,2,3]*vb[1,2,3]
#                 correl_funs[dis]=ov;
#             end
#             return correl_funs
#         end
#     elseif method=="spinspin" #operator on a single site breaks su2 symmetry, so there is an extra index obtained from svd of two-site operator
#         if direction=="x"
#             @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4,-4]*T3[-3,6,7];
#             @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4,-4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
#             ov=@tensor va[1,2,3,4]*vb[1,2,3,4]
#             correl_funs[1]=ov;
            
#             for dis=2:distance
#                 @tensor va[:]:=va[1,3,5,-4]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
#                 ov=@tensor va[1,2,3,4]*vb[1,2,3,4]
#                 correl_funs[dis]=ov;
#             end
#             return correl_funs
#         end
#     end
# end


# function correl_TransOp(vl,Tup,Tdown,AAfused)
#     if AAfused==[]
        
#         @tensor vl[:]:=vl[-1,1,3]*Tup[1,2,-2]*Tdown[-3,2,3];
        
#     else
        
#         @tensor vl[:]:=vl[-1,1,3,5]*Tup[1,2,-2]*AAfused[3,4,-3,2]*Tdown[-4,4,5];
        
#     end
#     return vl
# end



# function cal_correl(M, AA_fused,AA_SS,AA_SAL,AA_SBL,AA_SAR,AA_SBR, chi,CTM, distance)
#     #M: number of virtual modes 
    


#     #single-unitcell correlations
#     norm=ob_1site_closed(CTM,AA_fused);
    
#     SS_cell_ob=ob_1site_closed(CTM,AA_SS);
#     SS_cell_ob=SS_cell_ob/norm;

    
#     norms=evaluate_correl_spinspin("x", AA_fused, AA_fused, AA_fused, CTM, "dimerdimer", 10);
#     norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
#     norms=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_fused, AA_fused, CTM, "dimerdimer", distance);
#     dimer_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SS, AA_SS, CTM, "dimerdimer", distance);

#     SASA_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SAL, AA_SAR, CTM, "spinspin", distance);
#     SASB_ob=evaluate_correl_spinspin("x", AA_fused/norm_coe, AA_SAL, AA_SBR, CTM, "spinspin", distance);

#     dimer_ob=dimer_ob./norms;
#     SASA_ob=SASA_ob./norms;
#     SASB_ob=SASB_ob./norms;


#     eus_x, Qspin_x, QN_x=solve_correl_length(5,AA_fused/norm_coe,CTM,"x");




#     mat_filenm="correl_M"*string(M)*"_chi"*string(chi)*".mat";
#     matwrite(mat_filenm, Dict(
#         "SS_cell_ob" => SS_cell_ob,
#         "dimer_ob" => dimer_ob,
#         "SASA_ob" => SASA_ob,
#         "SASB_ob" => SASB_ob,
#         "eus_x" => eus_x,
#         "Qspin_x"=> Qspin_x,
#         "QN_x"=> QN_x,
#         "CTM_space"=> string(space(CTM.Cset.C1))
#     ); compress = false)
# end