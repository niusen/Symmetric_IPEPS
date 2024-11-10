
function ob_2x2_iPESS(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy)
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_LU_[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_LD_[3,4,-5,-3]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
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

function get_AA_simple(double_B_set,double_T_set,pos)
    @tensor AA[:]:=double_B_set[pos[1]][pos[2]][-1,1,-4]*double_T_set[pos[1]][pos[2]][-2,-3,1];
    return AA
end




function ob_onsite_iPESS(CTM,O1,B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];


    # @tensor A1[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-5,1]

    B_LU=B_set[pos_LU[1],pos_LU[2]];#(LU,M)
    T_LU=T_set[pos_LU[1],pos_LU[2]];#(M,dRD)
    B_LU0=deepcopy(B_LU);
    T_LU0=deepcopy(T_LU);

    @tensor T_LU[:]:=T_LU[-1,1,-3,-4]*O1[-2,1];#M,d,R,D

    B_LU=permute(B_LU,(1,2,),(3,));
    T_LU=permute(T_LU,(1,),(2,3,4,));
    B_LU_double, _ = build_double_layer_swap_Tm(B_LU0',B_LU, false);#L M U
    T_LU_double, _ = build_double_layer_swap_Bm(T_LU0',T_LU, true);#D R M

    @tensor AA_LU[:]:=B_LU_double[-1,1,-4]*T_LU_double[-2,-3,1];
    ##############################
    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU, AA_RU0,AA_LD0,AA_RD0,cx,cy);
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy);
    ob=ob/Norm;
    return ob
end

function hopping_x_iPESS(CTM,O1,O2,B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    #########################################
    # @tensor A_LU[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    # gate=@ignore_derivatives parity_gate(A_LU,1); 
    # @tensor A_LU[:]:=A_LU[1,-2,-3,-4,-5,-6]*gate[-1,1];
    # gate=@ignore_derivatives parity_gate(A_LU,2); 
    # @tensor A_LU[:]:=A_LU[-1,1,-3,-4,-5,-6]*gate[-2,1];
    # gate=@ignore_derivatives parity_gate(A_LU,4); 
    # @tensor A_LU[:]:=A_LU[-1,-2,-3,1,-5,-6]*gate[-4,1];
    # U=@ignore_derivatives unitary(fuse(space(A_LU,3)⊗space(A_LU,6)), space(A_LU,3)⊗space(A_LU,6)); 
    # @tensor A_LU[:]:=A_LU[-1,-2,1,-4,-5,2]*U[-3,1,2];

    B_LU=B_set[pos_LU[1],pos_LU[2]];#(LU,M)
    T_LU=T_set[pos_LU[1],pos_LU[2]];#(M,dRD)
    B_LU0=deepcopy(B_LU);
    T_LU0=deepcopy(T_LU);

    @tensor T_LU[:]:=T_LU[-1,1,-3,-4]*O1[-5,-2,1];#M,d,R,D,virtual
    gate=@ignore_derivatives parity_gate(B_LU,1);#L 
    @tensor B_LU[:]:=B_LU[1,-2,-3]*gate[-1,1];#L,U,M
    gate=@ignore_derivatives parity_gate(T_LU,4); #D
    @tensor T_LU[:]:=T_LU[-1,-2,-3,1,-5]*gate[-4,1];#M,d,R,D,virtual
    gate=@ignore_derivatives parity_gate(B_LU,2);#U 
    @tensor B_LU[:]:=B_LU[-1,1,-3]*gate[-2,1];#L,U,M
    U=@ignore_derivatives unitary(fuse(space(T_LU,3)⊗space(T_LU,5)), space(T_LU,3)⊗space(T_LU,5)); 
    @tensor T_LU[:]:=T_LU[-1,-2,1,-4,2]*U[-3,1,2];#M,d,R',D

    B_LU=permute(B_LU,(1,2,),(3,));
    T_LU=permute(T_LU,(1,),(2,3,4,));
    B_LU_double, _ = build_double_layer_swap_Tm(B_LU0',B_LU, false);#L M U
    T_LU_double, _ = build_double_layer_swap_Bm(T_LU0',T_LU, true);#D R M
    ###########################################
    # @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
    # gate=@ignore_derivatives parity_gate(A_RU,1); 
    # @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,-6]*gate[-1,1];
    # gate=@ignore_derivatives parity_gate(A_RU,4); 
    # @tensor A_RU[:]:=A_RU[-1,-2,-3,1,-5,-6]*gate[-4,1];
    # @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,2]*U'[1,2,-1];

    B_RU=B_set[pos_RU[1],pos_RU[2]];#(LU,M)
    T_RU=T_set[pos_RU[1],pos_RU[2]];#(M,dRD)
    B_RU0=deepcopy(B_RU);
    T_RU0=deepcopy(T_RU);

    @tensor T_RU[:]:= T_RU[-1,1,-3,-4]*O2[-5,-2,1];#M,d,R,D,virtual'
    gate=@ignore_derivatives parity_gate(B_RU,1); #L
    @tensor B_RU[:]:=B_RU[1,-2,-3]*gate[-1,1];#L,U,M
    gate=@ignore_derivatives parity_gate(B_RU,2); #U
    @tensor B_RU[:]:=B_RU[-1,1,-3]*gate[-2,1];#L,U,M
    
    U2=@ignore_derivatives unitary(fuse(space(T_RU,1)⊗space(T_RU,5)), space(T_RU,1)⊗space(T_RU,5));
    @tensor T_RU[:]:=T_RU[1,-2,-3,-4,2]*U2[-1,1,2];##M',d,R,D

    O_string=@ignore_derivatives unitary(space(O1,1)',space(O1,1)');
    @tensor B_RU[:]:=B_RU[-1,-2,-3]*O_string[-4,-5];#(L,U,M), (virtual,virtual')=>(L,U,M, virtual,virtual')
    @tensor B_RU[:]:=B_RU[1,-2,3,2,4]*U'[1,2,-1]*U2'[3,4,-3];#L,U,M


    B_RU=permute(B_RU,(1,2,),(3,));
    T_RU=permute(T_RU,(1,),(2,3,4,));
    B_RU_double, _ = build_double_layer_swap_Tm(B_RU0',B_RU, false);#L M U
    T_RU_double, _ = build_double_layer_swap_Bm(T_RU0',T_RU, true);#D R M
    ####################################
    @tensor AA_LU[:]:=B_LU_double[-1,1,-4]*T_LU_double[-2,-3,1];
    @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    # return AA_LU,AA_RU
      

    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU,AA_RU,AA_LD0,AA_RD0,cx,cy);
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy);
    ob=ob/Norm;
    return ob
end

function hopping_y_iPESS(CTM,O1,O2,B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    #############################################
    #the first index of O is dummy
    # @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    # gate=@ignore_derivatives parity_gate(A_RU,1); 
    # @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,-6]*gate[-1,1];
    # gate=@ignore_derivatives parity_gate(A_RU,2); 
    # @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,-6]*gate[-2,1];
    # gate=@ignore_derivatives parity_gate(A_RU,4); 
    # @tensor A_RU[:]:=A_RU[-1,-2,-3,1,-5,-6]*gate[-4,1];
    # U1=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
    # @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U1[-2,1,2];
    ####
    B_RU=B_set[pos_RU[1],pos_RU[2]];#(LU,M)
    T_RU=T_set[pos_RU[1],pos_RU[2]];#(M,dRD)
    B_RU0=deepcopy(B_RU);
    T_RU0=deepcopy(T_RU);

    @tensor T_RU[:]:= T_RU[-1,1,-3,-4]*O1[-5,-2,1];#M,d,R,D,virtual
    gate=@ignore_derivatives parity_gate(B_RU,1); #L
    @tensor B_RU[:]:=B_RU[1,-2,-3]*gate[-1,1];#L,U,M
    gate=@ignore_derivatives parity_gate(T_RU,4); #D
    @tensor T_RU[:]:=T_RU[-1,-2,-3,1,-5]*gate[-4,1];#M,d,R,D,virtual
    gate=@ignore_derivatives parity_gate(B_RU,2); #U
    @tensor B_RU[:]:=B_RU[-1,1,-3]*gate[-2,1];#L,U,M
    U1=@ignore_derivatives unitary(fuse(space(T_RU,4)⊗space(T_RU,5)), space(T_RU,4)⊗space(T_RU,5)); 
    @tensor T_RU[:]:=T_RU[-1,-2,-3,1,2]*U1[-4,1,2];#M,d,R,D'

    B_RU=permute(B_RU,(1,2,),(3,));
    T_RU=permute(T_RU,(1,),(2,3,4,));
    B_RU_double, _ = build_double_layer_swap_Tm(B_RU0',B_RU, false);#L M U
    T_RU_double, _ = build_double_layer_swap_Bm(T_RU0',T_RU, true);#D R M
    ####################################
    # @tensor A_RD[:]:= A_cell[pos_RD[1]][pos_RD[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
    # @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,2]*U1'[1,2,-4];
    ####
    B_RD=B_set[pos_RD[1],pos_RD[2]];#(LU,M)
    T_RD=T_set[pos_RD[1],pos_RD[2]];#(M,dRD)
    B_RD0=deepcopy(B_RD);
    T_RD0=deepcopy(T_RD);

    
    @tensor T_RD[:]:= T_RD[-1,1,-3,-4]*O2[-5,-2,1];#M,d,R,D,virtual
    U2=@ignore_derivatives unitary(fuse(space(T_RD,1)⊗space(T_RD,5)), space(T_RD,1)⊗space(T_RD,5));
    @tensor T_RD[:]:=T_RD[1,-2,-3,-4,2]*U2[-1,1,2];#M',d,R,D

    O_string=@ignore_derivatives unitary(space(O1,1)',space(O1,1)');
    @tensor B_RD[:]:= B_RD[-1,-2,-3]*O_string[-4,-5];#(L,U,M), (virtual',virtual)=>(L,U,M, virtual',virtual)
    @tensor B_RD[:]:=B_RD[-1,1,3,2,4]*U1'[1,2,-2]*U2'[3,4,-3];#L,U,M

    B_RD=permute(B_RD,(1,2,),(3,));
    T_RD=permute(T_RD,(1,),(2,3,4,));

    B_RD_double, _ = build_double_layer_swap_Tm(B_RD0',B_RD, false);#L M U
    T_RD_double, _ = build_double_layer_swap_Bm(T_RD0',T_RD, true);#D R M
    ###################################################
    @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    @tensor AA_RD[:]:=B_RD_double[-1,1,-4]*T_RD_double[-2,-3,1];
    # return AA_RU,AA_RD


    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);
    
    ob=ob_2x2_iPESS(CTM,AA_LU0,AA_RU,AA_LD0,AA_RD,cx,cy);
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy);
    ob=ob/Norm;
    return ob
end

function hopping_diagonala_iPESS(CTM,O1,O2,B_set,T_set, double_B_set, double_T_set, cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    ###################################################

    # @tensor A_LD[:]:= A_cell[pos_LD[1]][pos_LD[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    # gate=@ignore_derivatives parity_gate(A_LD,1); #L
    # @tensor A_LD[:]:=A_LD[1,-2,-3,-4,-5,-6]*gate[-1,1];
    # gate=@ignore_derivatives parity_gate(A_LD,2); #D
    # @tensor A_LD[:]:=A_LD[-1,1,-3,-4,-5,-6]*gate[-2,1];
    # gate=@ignore_derivatives parity_gate(A_LD,4); #U
    # @tensor A_LD[:]:=A_LD[-1,-2,-3,1,-5,-6]*gate[-4,1];
    # U1=@ignore_derivatives unitary(fuse(space(A_LD,3)⊗space(A_LD,6)), space(A_LD,3)⊗space(A_LD,6)); 
    # @tensor A_LD[:]:=A_LD[-1,-2,1,-4,-5,2]*U1[-3,1,2];
    ######
    B_LD=B_set[pos_LD[1],pos_LD[2]];#(LU,M)
    T_LD=T_set[pos_LD[1],pos_LD[2]];#(M,dRD)
    B_LD0=deepcopy(B_LD);
    T_LD0=deepcopy(T_LD);

    @tensor T_LD[:]:= T_LD[-1,1,-2,-3]*O1[-5,-4,1];#M,R,D,d,virtual
    gate=@ignore_derivatives parity_gate(B_LD,1); #L
    @tensor B_LD[:]:=B_LD[1,-2,-3]*gate[-1,1];#L,U,M
    gate=@ignore_derivatives parity_gate(T_LD,3); #D
    @tensor T_LD[:]:=T_LD[-1,-2,1,-4,-5]*gate[-3,1];#M,R,D,d,virtual
    gate=@ignore_derivatives parity_gate(B_LD,2); #U
    @tensor B_LD[:]:=B_LD[-1,1,-3]*gate[-2,1];#L,U,M
    U1=@ignore_derivatives unitary(fuse(space(T_LD,2)⊗space(T_LD,5)), space(T_LD,2)⊗space(T_LD,5)); 
    @tensor T_LD[:]:=T_LD[-1,1,-3,-4,2]*U1[-2,1,2];#M,R',D,d
    T_LD=permute(T_LD,(1,4,2,3,));#M,d,R',D

    B_LD=permute(B_LD,(1,2,),(3,));
    T_LD=permute(T_LD,(1,),(2,3,4,));

    B_LD_double, _ = build_double_layer_swap_Tm(B_LD0',B_LD, false);#L M U
    T_LD_double, _ = build_double_layer_swap_Bm(T_LD0',T_LD, true);#D R M

    #############################################
    # @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
    # gate=@ignore_derivatives parity_gate(A_RU,3); #R
    # @tensor A_RU[:]:=A_RU[-1,-2,1,-4,-5,-6]*gate[-3,1];
    # gate=@ignore_derivatives parity_gate(A_RU,5); #d
    # @tensor A_RU[:]:=A_RU[-1,-2,-3,-4,1,-6]*gate[-5,1];
    # U2=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
    # @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U2[-2,1,2];
    ######
    B_RU=B_set[pos_RU[1],pos_RU[2]];#(LU,M)
    T_RU=T_set[pos_RU[1],pos_RU[2]];#(M,dRD)
    B_RU0=deepcopy(B_RU);
    T_RU0=deepcopy(T_RU);

    @tensor T_RU[:]:= T_RU[-1,1,-2,-3]*O2[-5,-4,1];#M,R,D,d,virtual
    gate=@ignore_derivatives parity_gate(T_RU,2); #R
    @tensor T_RU[:]:=T_RU[-1,1,-3,-4,-5]*gate[-2,1];
    gate=@ignore_derivatives parity_gate(T_RU,4); #d
    @tensor T_RU[:]:=T_RU[-1,-2,-3,1,-5]*gate[-4,1];
    U2=@ignore_derivatives unitary(fuse(space(T_RU,3)⊗space(T_RU,5)), space(T_RU,3)⊗space(T_RU,5)); 
    @tensor T_RU[:]:=T_RU[-1,-2,1,-4,2]*U2[-3,1,2];#M,R,D',d
    T_RU=permute(T_RU,(1,4,2,3,));#M,d,R,D

    B_RU=permute(B_RU,(1,2,),(3,));
    T_RU=permute(T_RU,(1,),(2,3,4,));

    B_RU_double, _ = build_double_layer_swap_Tm(B_RU0',B_RU, false);#L M U
    T_RU_double, _ = build_double_layer_swap_Bm(T_RU0',T_RU, true);#D R M
    ################################################
    # O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));
    # gate=@ignore_derivatives parity_gate(A_cell[pos_RD[1]][pos_RD[2]],2); # D
    # @tensor A_RD[:]:=A_cell[pos_RD[1]][pos_RD[2]][-1,1,-3,-4,-5]*gate[-2,1];
    # gate=@ignore_derivatives parity_gate(A_RD,3); # R
    # @tensor A_RD[:]:=A_RD[-1,-2,1,-4,-5]*gate[-3,1];
    # gate=@ignore_derivatives parity_gate(A_RD,5); # d
    # @tensor A_RD[:]:=A_RD[-1,-2,-3,-4,1]*gate[-5,1];
    # @tensor A_RD[:]:=A_RD[1,-2,-3,3,-5]*O_string[4,2]*U1'[1,2,-1]*U2'[3,4,-4];
    ######
    B_RD=B_set[pos_RD[1],pos_RD[2]];#(LU,M)
    T_RD=T_set[pos_RD[1],pos_RD[2]];#(M,dRD)
    B_RD0=deepcopy(B_RD);
    T_RD0=deepcopy(T_RD);

    O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));
    gate=@ignore_derivatives parity_gate(T_RD,4); # D
    @tensor T_RD[:]:=T_RD[-1,-2,-3,1]*gate[-4,1];#M,d,R,D
    gate=@ignore_derivatives parity_gate(T_RD,3); # R
    @tensor T_RD[:]:=T_RD[-1,-2,1,-4]*gate[-3,1];#M,d,R,D
    gate=@ignore_derivatives parity_gate(T_RD,2); # d
    @tensor T_RD[:]:=T_RD[-1,1,-3,-4]*gate[-2,1];#M,d,R,D
    @tensor B_RD[:]:=B_RD[1,3,-3]*O_string[4,2]*U1'[1,2,-1]*U2'[3,4,-2];#L,U,M


    B_RD=permute(B_RD,(1,2,),(3,));
    T_RD=permute(T_RD,(1,),(2,3,4,));

    B_RD_double, _ = build_double_layer_swap_Tm(B_RD0',B_RD, false);#L M U
    T_RD_double, _ = build_double_layer_swap_Bm(T_RD0',T_RD, true);#D R M
    ################################################
    @tensor AA_LD[:]:=B_LD_double[-1,1,-4]*T_LD_double[-2,-3,1];
    @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
    @tensor AA_RD[:]:=B_RD_double[-1,1,-4]*T_RD_double[-2,-3,1];
    # return AA_LD,AA_RU,AA_RD
    ################################################

    AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
    AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
    AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
    AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

    ob=ob_2x2_iPESS(CTM,AA_LU0,AA_RU,AA_LD,AA_RD,cx,cy);
    Norm=ob_2x2_iPESS(CTM,AA_LU0,AA_RU0,AA_LD0,AA_RD0,cx,cy);
    ob=ob/Norm;
    return ob        
end




function evaluate_ob_cell_iPESS(parameters, B_set,T_set, double_B_set, double_T_set, CTM_cell, ctm_setting, energy_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    

    global Lx,Ly

    if isa(space(B_set[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        
        if energy_setting.model in  ("Triangle_Hofstadter_Hubbard", "spinful_triangle_lattice", "standard_triangle_Hubbard")
            Hamiltonian_terms=Hamiltonians_spinful_Z2;
        elseif (energy_setting.model == "Triangle_Hofstadter_spinless")
            Hamiltonian_terms=Hamiltonians_spinless_Z2;
        end
    elseif isa(space(B_set[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(B_set[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(B_set[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        if mod(energy_setting.Magnetic_cell,2)==1 #odd number of sites in unitcell
            @assert mod(Ly,2)==0;
            #if use U1 symmetry, use different dummy physical space along y direction along Ly, where Ly should be even number
        end
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end


    if energy_setting.model=="spinful_triangle_lattice"
    
        @assert mod(Lx,2)==0
        #for 120 degree magnetic order in the Hofstadter M2 model. Unit-cell for 120 degree order should be at least 3x3.  
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
        t1=parameters["t1"];
        t2=parameters["t2"];
        ϕ=parameters["ϕ"];
        μ=parameters["μ"];
        U=parameters["U"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonala_set=zeros(Lx,Ly)*im;
        e0_set=zeros(Lx,Ly)*im;
        eU_set=zeros(Lx,Ly)*im;

        
        E_total=0;

        for cx=1:Lx
            for cy=1:Ly


                # pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
                # pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
                # pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
                # pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

                # AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
                # AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
                # AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
                # AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);
                


                ex=hopping_x_iPESS(CTM_cell,Cdag_set[mod1(cx+1,2)],C_set[mod1(cx+2,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                ey=hopping_y_iPESS(CTM_cell,Cdag_set[mod1(cx+2,2)],C_set[mod1(cx+2,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                e_diagonala=hopping_diagonala_iPESS(CTM_cell,Cdag_set[mod1(cx+1,2)],C_set[mod1(cx+2,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                e0=ob_onsite_iPESS(CTM_cell,N_occu_set[mod1(cx+1,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                eU=ob_onsite_iPESS(CTM_cell,n_double_set[mod1(cx+1,2)]-(1/2)*N_occu_set[mod1(cx+1,2)]+(1/4)*Ident_set[mod1(cx+1,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                @ignore_derivatives ex_set[cx,cy]=ex;
                @ignore_derivatives ey_set[cx,cy]=ey;
                @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
                @ignore_derivatives e0_set[cx,cy]=e0;
                @ignore_derivatives eU_set[cx,cy]=eU;
                if mod(cx,2)==1
                    E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala')  +U*eU);
                else
                    E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')+t1*(ey+ey')+t2*(e_diagonala+e_diagonala')  +U*eU);
                end
            end
        end

        E_total=E_total/(Lx*Ly);
        return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set
    

    elseif energy_setting.model in ("Triangle_Hofstadter_Hubbard","Triangle_Hofstadter_spinless")
        @assert mod(Lx,energy_setting.Magnetic_cell)==0;
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();

        parameters_site=@ignore_derivatives get_Hofstadter_coefficients(Lx,Ly,parameters,energy_setting);
        tx_coe_set=parameters_site["tx_coe_set"];
        ty_coe_set=parameters_site["ty_coe_set"];
        t2_coe_set=parameters_site["t2_coe_set"];
        U_coe_set=parameters_site["U_coe_set"];
        μ_coe_set=parameters_site["μ_coe_set"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonala_set=zeros(Lx,Ly)*im;
        e0_set=zeros(Lx,Ly)*im;
        eU_set=zeros(Lx,Ly)*im;
        
        E_total=0;
        for px=1:Lx
            for py=1:Ly
                #(cx,cy): coordinate of left-top C1 tensor
                cx=mod1(px-1,Lx);
                cy=mod1(py-1,Ly);
                ex=hopping_x_iPESS(CTM_cell,Cdag_set[mod1(py,Ly)],C_set[mod1(py,Ly)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                ey=hopping_y_iPESS(CTM_cell,Cdag_set[mod1(py,Ly)],C_set[mod1(py+1,Ly)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                e_diagonala=hopping_diagonala_iPESS(CTM_cell,Cdag_set[mod1(py+1,Ly)],C_set[mod1(py,Ly)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                e0=ob_onsite_iPESS(CTM_cell,N_occu_set[mod1(py,Ly)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                eU=ob_onsite_iPESS(CTM_cell,n_double_set[mod1(py,Ly)]-(1/2)*N_occu_set[mod1(py,Ly)]+(1/4)*Ident_set[mod1(py,Ly)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                @ignore_derivatives ex_set[px,py]=ex;
                @ignore_derivatives ey_set[px,py]=ey;
                @ignore_derivatives e_diagonala_set[px,py]=e_diagonala;
                @ignore_derivatives e0_set[px,py]=e0;
                @ignore_derivatives eU_set[px,py]=eU;
                tx_coe=tx_coe_set[px,py];
                ty_coe=ty_coe_set[px,py];
                t2_coe=t2_coe_set[px,py];
                U_coe=U_coe_set[px,py];
                μ_coe=μ_coe_set[px,py];
                E_temp=tx_coe*ex +ty_coe*ey +t2_coe*e_diagonala -μ_coe*e0/2  +U_coe*eU/2;
                E_total=E_total+real(E_temp+E_temp');
                
            end
        end 
        E_total=E_total/(Lx*Ly);
        return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set
    elseif energy_setting.model =="standard_triangle_Hubbard" 
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
        t1=parameters["t1"];
        t2=parameters["t2"];
        μ=parameters["μ"];
        U=parameters["U"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonala_set=zeros(Lx,Ly)*im;
        e0_set=zeros(Lx,Ly)*im;
        eU_set=zeros(Lx,Ly)*im;
        
        E_total=0;
        for px=1:Lx
            for py=1:Ly
                #(cx,cy): coordinate of left-top C1 tensor
                cx=mod1(px-1,Lx);
                cy=mod1(py-1,Ly);
                ex=hopping_x_iPESS(CTM_cell,Cdag_set[mod1(py,2)],C_set[mod1(py,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                ey=hopping_y_iPESS(CTM_cell,Cdag_set[mod1(py,2)],C_set[mod1(py+1,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                e_diagonala=hopping_diagonala_iPESS(CTM_cell,Cdag_set[mod1(py+1,2)],C_set[mod1(py,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                e0=ob_onsite_iPESS(CTM_cell,N_occu_set[mod1(py,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                eU=ob_onsite_iPESS(CTM_cell,n_double_set[mod1(py,2)]-(1/2)*N_occu_set[mod1(py,2)]+(1/4)*Ident_set[mod1(py,2)],B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
                @ignore_derivatives ex_set[px,py]=ex;
                @ignore_derivatives ey_set[px,py]=ey;
                @ignore_derivatives e_diagonala_set[px,py]=e_diagonala;
                @ignore_derivatives e0_set[px,py]=e0;
                @ignore_derivatives eU_set[px,py]=eU;

                # E_temp=-t1*ex -t1*ey -t2*e_diagonala -μ*e0/2  +U*eU/2;
                E_temp=-t1*ex -t1*ey -t2*e_diagonala  +U*eU/2; # do not include chemical potential
                E_total=E_total+real(E_temp+E_temp');
                
            end
        end 
        E_total=E_total/(Lx*Ly);
        return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set
    end
end




function evaluate_spin_cell_iPESS(B_set,T_set, double_B_set, double_T_set, CTM_cell, ctm_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    
    global Lx,Ly
    if isa(space(B_set[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        sx_op,sy_op,sz_op=spin_operator_Z2();
    else
        println("Virtual symmetry is not Z2, no need to compute spin polarization.")
    end
    sx_set=zeros(Lx,Ly)*im;
    sy_set=zeros(Lx,Ly)*im;
    sz_set=zeros(Lx,Ly)*im;
    for cx=1:Lx
        for cy=1:Ly
            #(cx,cy): coordinate of left-top C1 tensor

            # pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
            # pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
            # pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
            # pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

            # AA_LU0=get_AA_simple(double_B_set,double_T_set,pos_LU);
            # AA_LD0=get_AA_simple(double_B_set,double_T_set,pos_LD);
            # AA_RU0=get_AA_simple(double_B_set,double_T_set,pos_RU);
            # AA_RD0=get_AA_simple(double_B_set,double_T_set,pos_RD);

            sx0=ob_onsite_iPESS(CTM_cell,sx_op,B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
            sy0=ob_onsite_iPESS(CTM_cell,sy_op,B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
            sz0=ob_onsite_iPESS(CTM_cell,sz_op,B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting);
            @ignore_derivatives sx_set[cx,cy]=sx0;
            @ignore_derivatives sy_set[cx,cy]=sy0;
            @ignore_derivatives sz_set[cx,cy]=sz0;
        end
    end 
    return sx_set,sy_set,sz_set
end