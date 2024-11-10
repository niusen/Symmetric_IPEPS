function verify_hopping_diagonala_iPESS(CTM,A_cell,AA_cell, B_set,T_set, double_B_set, double_T_set, cx,cy,ctm_setting,energy_setting)

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

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();




    # ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,2)],C_set[mod1(cx+2,2)],A_cell,AA_cell,cx,cy,ctm_setting);
    # ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,2)],C_set[mod1(cx+2,2)],A_cell,AA_cell,cx,cy,ctm_setting);

    O1=Cdag_set[1];
    O2=C_set[2];


    function hopping_diagonala_new(CTM,O1,O2,B_set,T_set, double_B_set, double_T_set, cx,cy,ctm_setting)
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
        return AA_LD,AA_RU,AA_RD
        ################################################
    
        # # ob=ob_2x2_iPESS(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_LD_double,AA_RD_double,cx,cy);
        # # Norm=ob_2x2_iPESS(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
        # ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU,AA_LD,AA_RD,cx,cy);
        # Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
        # ob=ob/Norm;
        return ob        
    end

    function hopping_diagonala_old(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
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
    
    
        gate=@ignore_derivatives parity_gate(A_LD,1); 
        @tensor A_LD[:]:=A_LD[1,-2,-3,-4,-5,-6]*gate[-1,1];
        gate=@ignore_derivatives parity_gate(A_LD,2); 
        @tensor A_LD[:]:=A_LD[-1,1,-3,-4,-5,-6]*gate[-2,1];
        gate=@ignore_derivatives parity_gate(A_LD,4); 
        @tensor A_LD[:]:=A_LD[-1,-2,-3,1,-5,-6]*gate[-4,1];
    
        gate=@ignore_derivatives parity_gate(A_cell[pos_RD[1]][pos_RD[2]],2); 
        @tensor A_RD[:]:=A_cell[pos_RD[1]][pos_RD[2]][-1,1,-3,-4,-5]*gate[-2,1];
        gate=@ignore_derivatives parity_gate(A_RD,3); 
        @tensor A_RD[:]:=A_RD[-1,-2,1,-4,-5]*gate[-3,1];
        gate=@ignore_derivatives parity_gate(A_RD,5); 
        @tensor A_RD[:]:=A_RD[-1,-2,-3,-4,1]*gate[-5,1];
    
        gate=@ignore_derivatives parity_gate(A_RU,3); 
        @tensor A_RU[:]:=A_RU[-1,-2,1,-4,-5,-6]*gate[-3,1];
        gate=@ignore_derivatives parity_gate(A_RU,5); 
        @tensor A_RU[:]:=A_RU[-1,-2,-3,-4,1,-6]*gate[-5,1];
    
    
        U1=@ignore_derivatives unitary(fuse(space(A_LD,3)⊗space(A_LD,6)), space(A_LD,3)⊗space(A_LD,6)); 
        U2=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
        @tensor A_LD[:]:=A_LD[-1,-2,1,-4,-5,2]*U1[-3,1,2];
        @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U2[-2,1,2];
        @tensor A_RD[:]:=A_RD[1,-2,-3,3,-5]*O_string[4,2]*U1'[1,2,-1]*U2'[3,4,-4];
    
    
        if ctm_setting.grad_checkpoint
            AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
            AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
            AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
        else
            AA_LD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
            AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
            AA_RD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
        end
    
    
        return AA_LD_double, AA_RU_double, AA_RD_double
    end

    AA_LD_, AA_RU_, AA_RD_=hopping_diagonala_old(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting);

    AA_LD,AA_RU,AA_RD=hopping_diagonala_new(CTM,O1,O2,B_set,T_set, double_B_set, double_T_set, cx,cy,ctm_setting)

    @assert norm(AA_LD_-AA_LD)/norm(AA_LD)<1e-12;
    @assert norm(AA_RU_-AA_RU)/norm(AA_RU)<1e-12;
    @assert norm(AA_RD_-AA_RD)/norm(AA_RD)<1e-12;

    @show norm(AA_LD_-AA_LD)/norm(AA_LD);
    @show norm(AA_RU_-AA_RU)/norm(AA_RU);
    @show norm(AA_RD_-AA_RD)/norm(AA_RD);

end





function verify_hopping_y_iPESS(CTM,A_cell,AA_cell, B_set,T_set, double_B_set, double_T_set, cx,cy,ctm_setting,energy_setting)

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

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();




    # ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,2)],C_set[mod1(cx+2,2)],A_cell,AA_cell,cx,cy,ctm_setting);
    # ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,2)],C_set[mod1(cx+2,2)],A_cell,AA_cell,cx,cy,ctm_setting);

    O1=Cdag_set[1];
    O2=C_set[2];


    function hopping_y_new(CTM,O1,O2,B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting)
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
        return AA_RU,AA_RD
        
        # ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_RD_double,cx,cy);
        # Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
        # ob=ob/Norm;
        # return ob
    end

    function hopping_y_old(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
        global Lx,Ly
        pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
        pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
        pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
        pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];
    
        #the first index of O is dummy
        @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
        @tensor A_RD[:]:= A_cell[pos_RD[1]][pos_RD[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
    
    
        
        gate=@ignore_derivatives parity_gate(A_RU,1); 
        @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,-6]*gate[-1,1];
        gate=@ignore_derivatives parity_gate(A_RU,2); 
        @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,-6]*gate[-2,1];
        gate=@ignore_derivatives parity_gate(A_RU,4); 
        @tensor A_RU[:]:=A_RU[-1,-2,-3,1,-5,-6]*gate[-4,1];
    
    
        U1=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
        @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U1[-2,1,2];
        @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,2]*U1'[1,2,-4];
    
    

        AA_RD,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
        AA_RU,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    
    
        return AA_RU,AA_RD
    end

    AA_RU_, AA_RD_=hopping_y_old(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting);

    AA_RU,AA_RD=hopping_y_new(CTM,O1,O2,B_set,T_set, double_B_set, double_T_set, cx,cy,ctm_setting)

    @assert norm(AA_RU_-AA_RU)/norm(AA_RU)<1e-12;
    @assert norm(AA_RD_-AA_RD)/norm(AA_RD)<1e-12;

    @show norm(AA_RU_-AA_RU)/norm(AA_RU);
    @show norm(AA_RD_-AA_RD)/norm(AA_RD);

end





function verify_hopping_x_iPESS(CTM,A_cell,AA_cell, B_set,T_set, double_B_set, double_T_set, cx,cy,ctm_setting,energy_setting)

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

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();




    # ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,2)],C_set[mod1(cx+2,2)],A_cell,AA_cell,cx,cy,ctm_setting);
    # ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,2)],C_set[mod1(cx+2,2)],A_cell,AA_cell,cx,cy,ctm_setting);

    O1=Cdag_set[1];
    O2=C_set[2];


    function hopping_x_new(CTM,O1,O2,B_set,T_set, double_B_set, double_T_set,cx,cy,ctm_setting)
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
        @tensor AA_RU[:]:=B_RU_double[-1,1,-4]*T_RU_double[-2,-3,1];
        @tensor AA_LU[:]:=B_LU_double[-1,1,-4]*T_LU_double[-2,-3,1];
        return AA_LU,AA_RU
        
    
    
        # ob=ob_2x2(CTM,AA_LU_double,AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
        # Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
        # ob=ob/Norm;
        # return ob
    end

    function hopping_x_old(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
        global Lx,Ly
        pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
        pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
        pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
        pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];
    
        @tensor A_LU[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
        @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
       
    
            
        gate=@ignore_derivatives parity_gate(A_LU,1); 
        @tensor A_LU[:]:=A_LU[1,-2,-3,-4,-5,-6]*gate[-1,1];
        gate=@ignore_derivatives parity_gate(A_LU,2); 
        @tensor A_LU[:]:=A_LU[-1,1,-3,-4,-5,-6]*gate[-2,1];
        gate=@ignore_derivatives parity_gate(A_LU,4); 
        @tensor A_LU[:]:=A_LU[-1,-2,-3,1,-5,-6]*gate[-4,1];
    
        gate=@ignore_derivatives parity_gate(A_RU,1); 
        @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,-6]*gate[-1,1];
        gate=@ignore_derivatives parity_gate(A_RU,4); 
        @tensor A_RU[:]:=A_RU[-1,-2,-3,1,-5,-6]*gate[-4,1];
    
    
        U=@ignore_derivatives unitary(fuse(space(A_LU,3)⊗space(A_LU,6)), space(A_LU,3)⊗space(A_LU,6)); 
        @tensor A_LU[:]:=A_LU[-1,-2,1,-4,-5,2]*U[-3,1,2];
        @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,2]*U'[1,2,-1];

        AA_LU,_,_,_,_=build_double_layer_swap(A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
        AA_RU,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);

        return AA_LU,AA_RU
    end

    AA_LU_, AA_RU_=hopping_x_old(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting);

    AA_LU,AA_RU=hopping_x_new(CTM,O1,O2,B_set,T_set, double_B_set, double_T_set, cx,cy,ctm_setting)

    @assert norm(AA_LU_-AA_LU)/norm(AA_LU)<1e-12;
    @assert norm(AA_RU_-AA_RU)/norm(AA_RU)<1e-12;

    @show norm(AA_LU_-AA_LU)/norm(AA_LU);
    @show norm(AA_RU_-AA_RU)/norm(AA_RU);

end