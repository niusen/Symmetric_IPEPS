function Rank(T::TensorMap)
    return length(domain(T))+length(codomain(T))
end

function spin_half_operators_Z2()
    

    V=Rep[ℤ₂](1=>2);



    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]/2; 
    
    #order: (0,0), (0,1), (1,0), (1,1)
    
    Sx=(sp+sm)/2;
    Sy=(sp-sm)/(2*im);
    Sz=sz;


    #Spin operator
    @tensor SpSm[:]:=sp[-1,-2]*sm[-3,-4];
    @tensor SmSp[:]:=sm[-1,-2]*sp[-3,-4];
    @tensor SzSz[:]:=sz[-1,-2]*sz[-3,-4];
    SS=SpSm/2+SmSp/2+SzSz;
    SS=permutedims(SS,(1,3,2,4));#s1's2's1s2
    SS=TensorMap(SS, V ⊗ V ← V ⊗ V);
    SS=permute(SS,(1,3,),(2,4,));#s1',s1,s2',s2
    u0,s0,v0=tsvd(SS; trunc=truncerr(1e-12));
    @assert norm(u0*s0*v0-SS)<1e-14;
    Sa=permute(u0*s0,(3,1,2,));
    Sb=permute(v0,(1,2,3,));


    @tensor Hchiral[:]:=Sx[-1,-4]*Sy[-2,-5]*Sz[-3,-6]-Sx[-1,-4]*Sz[-2,-5]*Sy[-3,-6]+Sy[-1,-4]*Sz[-2,-5]*Sx[-3,-6]-Sy[-1,-4]*Sx[-2,-5]*Sz[-3,-6]+Sz[-1,-4]*Sx[-2,-5]*Sy[-3,-6]-Sz[-1,-4]*Sy[-2,-5]*Sx[-3,-6];
    Hchiral=TensorMap(Hchiral, V ⊗ V ⊗ V ← V ⊗ V ⊗ V);
    u,s,v=tsvd(permute(Hchiral,(1,4,),(2,3,5,6)); trunc=truncerr(1e-12));
    chirality_S1=u;
    S2S3=s*v;
    u,s,v=tsvd(permute(S2S3,(1,2,4,),(3,5)); trunc=truncerr(1e-12));
    chirality_S2=u;
    chirality_S3=s*v;
    @tensor Hchiral_[:]:=chirality_S1[-1,-4,1]*chirality_S2[1,-2,-5,2]*chirality_S3[2,-3,-6];
    Hchiral_=permute(Hchiral_,(1,2,3,),(4,5,6,));
    @assert norm(Hchiral-Hchiral_)/norm(Hchiral)<1e-12;

    Sx_op=TensorMap(Sx,V,V);
    Sy_op=TensorMap(Sy,V,V);
    Sz_op=TensorMap(Sz,V,V);

    return  Sx_op,Sy_op,Sz_op, Sa, Sb, chirality_S1,chirality_S2,chirality_S3
end



function Hamiltonian_SU2_SU2()
    #SU(4) P_{ij} operator
    P_ij=zeros(4,4,4,4);#d1',d2',d1,d2
    for ca=1:4
        for cb=1:4
            P_ij[ca,cb,cb,ca]=1;
        end
    end

    Vp=Rep[SU₂ × SU₂]((1/2,1/2)=>1);
    P_ij=TensorMap(P_ij,Vp*Vp,Vp*Vp);

    P_ijk=zeros(4,4,4,4,4,4);#d1',d2',d3',d1,d2,d3
    for ca=1:4
        for cb=1:4
            for cc=1:4
                P_ijk[ca,cb,cc,cb,cc,ca]=1;
            end
        end
    end
    P_ijk=TensorMap(P_ijk,Vp*Vp*Vp,Vp*Vp*Vp);

    P_kji=permute(P_ijk,(3,2,1,),(6,5,4,));

    return P_ij,P_ijk,P_kji
end

function build_double_layer_A_B(Ap,A)
    # println(space(Ap))
    # println(space(A))
    
    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5));    

    U_L=@ignore_derivatives unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    U_D=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    U_R=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # uMp,sMp,vMp=tsvd(Ap);
    # uMp=uMp*sMp;
    # uM,sM,vM=tsvd(A);
    # uM=uM*sM;

    U_tem=@ignore_derivatives unitary(fuse(space(A,1)*space(A,2)), space(A,1)*space(A,2))*(1+0*im);
    vM=U_tem*A;
    uM=U_tem';
    U_temp=@ignore_derivatives unitary(fuse(space(Ap,1)*space(Ap,2)), space(Ap,1)*space(Ap,2))*(1+0*im);
    vMp=U_temp*Ap;
    uMp=U_temp';

    uMp=permute(uMp,(1,2,3,),())
    uM=permute(uM,(1,2,3,),())
    Vp=@ignore_derivatives space(uMp,3);
    V=@ignore_derivatives space(vM,1);
    U=@ignore_derivatives unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

    @tensor double_LD[:]:=uMp[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vMp=permute(vMp,(1,2,3,4,),());
    vM=permute(vM,(1,2,3,4,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];

    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))
    double_RU=permute(double_RU,(1,4,5,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,3,));
    AA_fused=double_LD*double_RU;

    ##########################
    AA_fused=permute(AA_fused,(1,2,3,4,));

    return AA_fused, U_L,U_D,U_R,U_U
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

function ob_onsite(CTM,O1,A_cell,AA_cell,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];


    @tensor A1[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-5,1]

    if ctm_setting.grad_checkpoint
        A1_double,_,_,_,_=Zygote.checkpointed(build_double_layer_A_B, A_cell[pos_LU[1]][pos_LU[2]]',A1);
    else
        A1_double,_,_,_,_=build_double_layer_A_B(A_cell[pos_LU[1]][pos_LU[2]]',A1);
    end

    ob=ob_2x2(CTM,A1_double, AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
end



function ob_2x2(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_LU_[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-1]*AA_LD_[3,4,-4,-2]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    rho=@tensor up[1,2,3,4,]*down[1,2,3,4];
    return rho
end




function rho_2x2_LD_RD_RU(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_LU_[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-1]*AA_LD_[3,4,-4,-2,-5]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4,-5]; 

    @tensor up[:]:=MM_LU[-1,-2,1,2]*MM_RU[1,2,-3,-4,-5];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4,-6];
    @tensor rho[:]:= up[1,2,3,4,-3]*down[1,2,3,4,-1,-2];
    return rho
end


function rho_2x2_LD_RU_LU(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_LU_[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-1]*AA_LD_[3,4,-4,-2,-5]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-3,4,2];  
    @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4]; 

    @tensor up[:]:=MM_LU[-1,-2,1,2,-5]*MM_RU[1,2,-3,-4,-6];
    @tensor down[:]:=MM_LD[-1,-2,1,2,-5]*MM_RD[1,2,-3,-4];
    @tensor rho[:]:= up[1,2,3,4,-3,-2]*down[1,2,3,4,-1];
    return rho
end

function hopping_x_no_sign(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    @tensor A_LU[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
   

    U=@ignore_derivatives unitary(fuse(space(A_LU,3)⊗space(A_LU,6)), space(A_LU,3)⊗space(A_LU,6)); 
    @tensor A_LU[:]:=A_LU[-1,-2,1,-4,-5,2]*U[-3,1,2];
    @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,2]*U'[1,2,-1];


    if ctm_setting.grad_checkpoint
        AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_A_B, A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_A_B, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    else
        AA_LU_double,_,_,_,_=build_double_layer_A_B(A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
        AA_RU_double,_,_,_,_=build_double_layer_A_B(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    end


    

    ob=ob_2x2(CTM,AA_LU_double,AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
end


function hopping_y_no_sign(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    #the first index of O is dummy
    @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RD[:]:= A_cell[pos_RD[1]][pos_RD[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]


    U1=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
    @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U1[-2,1,2];
    @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,2]*U1'[1,2,-4];


    if ctm_setting.grad_checkpoint
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_A_B, A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_A_B, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    else
        AA_RD_double,_,_,_,_=build_double_layer_A_B(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
        AA_RU_double,_,_,_,_=build_double_layer_A_B(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    end


    
    ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_RD_double,cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
end

function hopping_diagonala_no_sign(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];


    @tensor A_LD[:]:= A_cell[pos_LD[1]][pos_LD[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
    # @tensor A_LD[:]:= A_cell[pos_LD[1]][pos_LD[2]][-1,-2,-3,-4,1]*O1[-5,1]
    # @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-5,1]
    O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));

    A_RD=A_cell[pos_RD[1]][pos_RD[2]];

    U1=@ignore_derivatives unitary(fuse(space(A_LD,3)⊗space(A_LD,6)), space(A_LD,3)⊗space(A_LD,6)); 
    U2=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
    @tensor A_LD[:]:=A_LD[-1,-2,1,-4,-5,2]*U1[-3,1,2];
    @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U2[-2,1,2];
    @tensor A_RD[:]:=A_RD[1,-2,-3,3,-5]*O_string[4,2]*U1'[1,2,-1]*U2'[3,4,-4];


    if ctm_setting.grad_checkpoint
        AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_A_B, A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_A_B, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_A_B, A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
    else
        AA_LD_double,_,_,_,_=build_double_layer_A_B(A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
        AA_RU_double,_,_,_,_=build_double_layer_A_B(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_RD_double,_,_,_,_=build_double_layer_A_B(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
    end

    ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_LD_double,AA_RD_double,cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob        
end


function evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    
    global Lx,Ly

    if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        if energy_setting.model in ("triangle_spin_model_spiral",)
            
            operator_terms=spin_half_operators_Z2;
        end
    elseif isa(space(A_cell[1][1],1),GradedSpace{ProductSector{Tuple{SU2Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{SU2Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonian_SU2_SU2;
    end

    if energy_setting.model=="triangle_spin_model_spiral"

        sx,sy,sz, Sa, Sb, chirality_S1,chirality_S2,chirality_S3=@ignore_derivatives operator_terms()
        J=parameters["J"];


        SSx_set=zeros(Lx,Ly)*im;
        SSy_set=zeros(Lx,Ly)*im;
        SSdiagonal_set=zeros(Lx,Ly)*im;
        # triangle_right_bot_set=zeros(Lx,Ly)*im;
        # triangle_left_top_set=zeros(Lx,Ly)*im;

        E_total=0;
        px=1;
        py=1;

        #(cx,cy): coordinate of left-top C1 tensor
        cx=mod1(px-1,Lx);
        cy=mod1(py-1,Ly);

        coord_LU=[1,1];
        coord_LD=[1,2];
        coord_RD=[2,2];
        coord_RU=[2,1];




        op_LU=@ignore_derivatives exp(-im*2*pi/3*(coord_LU[1]-coord_LU[2])*sz);
        op_LD=@ignore_derivatives exp(-im*2*pi/3*(coord_LD[1]-coord_LD[2])*sz);
        op_RD=@ignore_derivatives exp(-im*2*pi/3*(coord_RD[1]-coord_RD[2])*sz);
        op_RU=@ignore_derivatives exp(-im*2*pi/3*(coord_RU[1]-coord_RU[2])*sz);

        # println(space(op_LU'))
        # println(space(Sa))
        # println(space(op_LU))
        @tensor Sa_LU[:]:=op_LU'[-2,1]*Sa[-1,1,2]*op_LU[2,-3];
        @tensor Sb_LU[:]:=op_LU'[-2,1]*Sb[-1,1,2]*op_LU[2,-3];

        @tensor Sa_LD[:]:=op_LD'[-2,1]*Sa[-1,1,2]*op_LD[2,-3];
        @tensor Sb_LD[:]:=op_LD'[-2,1]*Sb[-1,1,2]*op_LD[2,-3];

        @tensor Sa_RD[:]:=op_RD'[-2,1]*Sa[-1,1,2]*op_RD[2,-3];
        @tensor Sb_RD[:]:=op_RD'[-2,1]*Sb[-1,1,2]*op_RD[2,-3];

        @tensor Sa_RU[:]:=op_RU'[-2,1]*Sa[-1,1,2]*op_RU[2,-3];
        @tensor Sb_RU[:]:=op_RU'[-2,1]*Sb[-1,1,2]*op_RU[2,-3];

        SS_x=hopping_x_no_sign(CTM_cell,Sa_LU,Sb_RU,A_cell,AA_cell,cx,cy,ctm_setting);
        SS_y=hopping_y_no_sign(CTM_cell,Sa_RU,Sb_RD,A_cell,AA_cell,cx,cy,ctm_setting);
        SS_diagonal=hopping_diagonala_no_sign(CTM_cell,Sa_LD,Sb_RU,A_cell,AA_cell,cx,cy,ctm_setting);

        E_total=E_total+J*real(SS_x+SS_y+SS_diagonal);


        @ignore_derivatives SSx_set[cx,cy]=SS_x;
        @ignore_derivatives SSy_set[cx,cy]=SS_y;
        @ignore_derivatives SSdiagonal_set[cx,cy]=SS_diagonal;

        E_total=E_total/(Lx*Ly);
        return E_total,  SSx_set,SSy_set,SSdiagonal_set
    elseif energy_setting.model=="triangle_SU4_spin"
  
        #for 120 degree magnetic order in the Hofstadter M2 model. Unit-cell for 120 degree order should be at least 3x3.  
        P_ij, P_ijk, P_kji = @ignore_derivatives Hamiltonian_terms();
        J=parameters["J"];
        K=parameters["K"];
        Φ=parameters["Φ"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonal_set=zeros(Lx,Ly)*im;
        triangle_right_bot_set=zeros(Lx,Ly)*im;
        triangle_left_top_set=zeros(Lx,Ly)*im;

        E_total=0;

  
        AA_open_cell=initial_tuple_cell(Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                global U_ss
                AA_open,U_ss=build_double_layer_open(A_cell[cx][cy]);
                AA_open_cell=fill_tuple(AA_open_cell, AA_open, cx,cy);
            end
        end

        @tensor P_ij[:]:=P_ij[1,3,2,4]*U_ss[1,2,-1]*U_ss[3,4,-2];
        @tensor P_ijk[:]:=P_ijk[1,3,5,2,4,6]*U_ss[1,2,-1]*U_ss[3,4,-2]*U_ss[5,6,-3];
        @tensor P_kji[:]:=P_kji[1,3,5,2,4,6]*U_ss[1,2,-1]*U_ss[3,4,-2]*U_ss[5,6,-3];
        

        for cx=1:Lx
            for cy=1:Ly

                pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
                pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
                pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
                pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];
            

                # rho_LD_RD_RU=rho_2x2_LD_RD_RU(CTM_cell,AA_LU,AA_RU_open,AA_LD_open,AA_RD_open,cx,cy);
                # rho_LD_RU_LU=rho_2x2_LD_RU_LU(CTM_cell,AA_LU_open,AA_RU_open,AA_LD_open,AA_RD,cx,cy);
                rho_LD_RD_RU=rho_2x2_LD_RD_RU(CTM_cell,AA_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_LD[1]][pos_LD[2]],AA_open_cell[pos_RD[1]][pos_RD[2]],cx,cy);
                rho_LD_RU_LU=rho_2x2_LD_RU_LU(CTM_cell,AA_open_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);

                norm_LD_RD_RU=@tensor rho_LD_RD_RU[1,3,5]*U_ss[2,2,1]*U_ss[4,4,3]*U_ss[6,6,5];
                norm_LD_RU_LU=@tensor rho_LD_RU_LU[1,3,5]*U_ss[2,2,1]*U_ss[4,4,3]*U_ss[6,6,5];

                @tensor rho_LD_RD[:]:=rho_LD_RD_RU[-1,-2,1]*U_ss[2,2,1];
                @tensor rho_RD_RU[:]:=rho_LD_RD_RU[1,-1,-2]*U_ss[2,2,1];
                @tensor rho_RU_LD[:]:=rho_LD_RD_RU[-2,1,-1]*U_ss[2,2,1];

                ex=@tensor rho_LD_RD[1,2]*P_ij[1,2];
                ey=@tensor rho_RD_RU[1,2]*P_ij[1,2];
                e_diagonal=@tensor rho_RU_LD[1,2]*P_ij[1,2];
                triangle_right_bot=@tensor rho_LD_RD_RU[1,2,3]*P_ijk[1,2,3];
                triangle_left_top=@tensor rho_LD_RU_LU[1,2,3]*P_ijk[1,2,3];

                ex=ex/norm_LD_RD_RU;
                ey=ey/norm_LD_RD_RU;
                e_diagonal=e_diagonal/norm_LD_RD_RU;
                triangle_right_bot=triangle_right_bot/norm_LD_RD_RU;
                triangle_left_top=triangle_left_top/norm_LD_RU_LU;

                @ignore_derivatives ex_set[cx,cy]=ex;
                @ignore_derivatives ey_set[cx,cy]=ey;
                @ignore_derivatives e_diagonal_set[cx,cy]=e_diagonal;
                @ignore_derivatives triangle_right_bot_set[cx,cy]=triangle_right_bot;
                @ignore_derivatives triangle_left_top_set[cx,cy]=triangle_left_top;

                E_total=E_total+J*real(ex+ey+e_diagonal) +3*K*cos(Φ)*real(triangle_left_top+triangle_left_top'+triangle_right_bot+triangle_right_bot') +3*K*sin(Φ)*imag(im*triangle_left_top-im*triangle_left_top'+im*triangle_right_bot-im*triangle_right_bot');
                
            end
        end

        E_total=E_total/(Lx*Ly);
        return E_total,  ex_set, ey_set, e_diagonal_set, triangle_right_bot_set, triangle_left_top_set
    elseif energy_setting.model=="Heisenberg"
        Sa,Sb, Id_SS,  P_123_a,P_123_b,P_123_c,  P_132_a,P_132_b,P_132_c,   Id_P123_ab, Id_P123_bc,   chirality_123_a,chirality_123_b,chirality_123_c, Id_chirality_ab,Id_chirality_bc=@ignore_derivatives SUN_spin(energy_setting.N);

        J=parameters["J"];
        K=parameters["K"];
        Φ=parameters["Φ"];
        @assert K==0;

        @tensor P_ij[:]:=Sa[1,2,-1,-3]*Sb[2,1,-2,-4];
        @tensor P_ijk[:]:=P_123_a[1,2,-1,-4]*P_123_b[2,3,-2,-5]*P_123_c[3,1,-3,-6];
        @tensor P_kji[:]:=P_132_a[1,2,-1,-4]*P_132_b[2,3,-2,-5]*P_132_c[3,1,-3,-6];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonal_set=zeros(Lx,Ly)*im;
        triangle_right_bot_set=zeros(Lx,Ly)*im;
        triangle_left_top_set=zeros(Lx,Ly)*im;

        E_total=0;

  
        AA_open_cell=initial_tuple_cell(Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                global U_ss
                AA_open,U_ss=build_double_layer_open(A_cell[cx][cy]);
                AA_open_cell=fill_tuple(AA_open_cell, AA_open, cx,cy);
            end
        end

        @tensor P_ij[:]:=P_ij[1,3,2,4]*U_ss[1,2,-1]*U_ss[3,4,-2];
        @tensor P_ijk[:]:=P_ijk[1,3,5,2,4,6]*U_ss[1,2,-1]*U_ss[3,4,-2]*U_ss[5,6,-3];
        @tensor P_kji[:]:=P_kji[1,3,5,2,4,6]*U_ss[1,2,-1]*U_ss[3,4,-2]*U_ss[5,6,-3];
        

        for cx=1:Lx
            for cy=1:Ly

                pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
                pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
                pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
                pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];
            

                # rho_LD_RD_RU=rho_2x2_LD_RD_RU(CTM_cell,AA_LU,AA_RU_open,AA_LD_open,AA_RD_open,cx,cy);
                # rho_LD_RU_LU=rho_2x2_LD_RU_LU(CTM_cell,AA_LU_open,AA_RU_open,AA_LD_open,AA_RD,cx,cy);
                rho_LD_RD_RU=rho_2x2_LD_RD_RU(CTM_cell,AA_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_LD[1]][pos_LD[2]],AA_open_cell[pos_RD[1]][pos_RD[2]],cx,cy);
                rho_LD_RU_LU=rho_2x2_LD_RU_LU(CTM_cell,AA_open_cell[pos_LU[1]][pos_LU[2]],AA_open_cell[pos_RU[1]][pos_RU[2]],AA_open_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);

                norm_LD_RD_RU=@tensor rho_LD_RD_RU[1,3,5]*U_ss[2,2,1]*U_ss[4,4,3]*U_ss[6,6,5];
                norm_LD_RU_LU=@tensor rho_LD_RU_LU[1,3,5]*U_ss[2,2,1]*U_ss[4,4,3]*U_ss[6,6,5];

                @tensor rho_LD_RD[:]:=rho_LD_RD_RU[-1,-2,1]*U_ss[2,2,1];
                @tensor rho_RD_RU[:]:=rho_LD_RD_RU[1,-1,-2]*U_ss[2,2,1];
                @tensor rho_RU_LD[:]:=rho_LD_RD_RU[-2,1,-1]*U_ss[2,2,1];

                ex=@tensor rho_LD_RD[1,2]*P_ij[1,2];
                ey=@tensor rho_RD_RU[1,2]*P_ij[1,2];
                e_diagonal=@tensor rho_RU_LD[1,2]*P_ij[1,2];
                triangle_right_bot=@tensor rho_LD_RD_RU[1,2,3]*P_ijk[1,2,3];
                triangle_left_top=@tensor rho_LD_RU_LU[1,2,3]*P_ijk[1,2,3];

                ex=ex/norm_LD_RD_RU;
                ey=ey/norm_LD_RD_RU;
                e_diagonal=e_diagonal/norm_LD_RD_RU;
                triangle_right_bot=triangle_right_bot/norm_LD_RD_RU;
                triangle_left_top=triangle_left_top/norm_LD_RU_LU;

                @ignore_derivatives ex_set[cx,cy]=ex;
                @ignore_derivatives ey_set[cx,cy]=ey;
                @ignore_derivatives e_diagonal_set[cx,cy]=e_diagonal;
                @ignore_derivatives triangle_right_bot_set[cx,cy]=triangle_right_bot;
                @ignore_derivatives triangle_left_top_set[cx,cy]=triangle_left_top;

                E_total=E_total+J*real(ex+ey+e_diagonal) +3*K*cos(Φ)*real(triangle_left_top+triangle_left_top'+triangle_right_bot+triangle_right_bot') +3*K*sin(Φ)*imag(im*triangle_left_top-im*triangle_left_top'+im*triangle_right_bot-im*triangle_right_bot');
                
            end
        end

        E_total=E_total/(Lx*Ly);
        return E_total,  ex_set, ey_set, e_diagonal_set, triangle_right_bot_set, triangle_left_top_set
    end
end


function evaluate_spin_cell(A_cell::Tuple, AA_cell, CTM_cell, ctm_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    
    global Lx,Ly
    if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        sx_op,sy_op,sz_op, Sa, Sb, chirality_S1,chirality_S2,chirality_S3=@ignore_derivatives spin_half_operators_Z2()
    else
        println("Virtual symmetry is not Z2, no need to compute spin polarization.")
    end

    sx_op=sx_op*2;#max spin length 1
    sy_op=sy_op*2;#max spin length 1
    sz_op=sz_op*2;#max spin length 1

    sx_set=zeros(Lx,Ly)*im;
    sy_set=zeros(Lx,Ly)*im;
    sz_set=zeros(Lx,Ly)*im;
    for cx=1:Lx
        for cy=1:Ly
            #(cx,cy): coordinate of left-top C1 tensor

            sx0=ob_onsite(CTM_cell,sx_op,A_cell,AA_cell,cx,cy,ctm_setting);
            sy0=ob_onsite(CTM_cell,sy_op,A_cell,AA_cell,cx,cy,ctm_setting);
            sz0=ob_onsite(CTM_cell,sz_op,A_cell,AA_cell,cx,cy,ctm_setting);
            @ignore_derivatives sx_set[cx,cy]=sx0;
            @ignore_derivatives sy_set[cx,cy]=sy0;
            @ignore_derivatives sz_set[cx,cy]=sz0;
        end
    end 
    return sx_set,sy_set,sz_set
end









