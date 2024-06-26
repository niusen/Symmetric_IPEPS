


function contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,AA_LU,AA_RU,AA_LD,AA_RD)
    global left_right_env_method;
    if left_right_env_method=="exact"
        @tensor VL[:]:=VL0[1,3,5,7]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2]*AA_LD[5,6,-3,4]*mps_bot[x_range[1]][7,-4,6];
        @tensor VR[:]:=VR0[1,3,5,7]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2]*AA_RD[-3,6,5,4]*mps_bot[x_range[2]][-4,7,6];
    elseif left_right_env_method=="trun"
        @tensor VL[:]:=VL0[1][4,6,7]*VL0[2][7,2,1]*mps_top[x_range[1]][4,-1,5]*AA_LU[6,8,-2,5]*AA_LD[2,3,-3,8]*mps_bot[x_range[1]][1,-4,3];
        @tensor VR[:]:=VR0[1][1,3,8]*VR0[2][8,5,4]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,7,3,2]*AA_RD[-3,6,5,7]*mps_bot[x_range[2]][-4,4,6];
    end
    ob=@tensor VL[1,2,3,4]*VR[1,2,3,4];
    return ob
end


function prepare_LU(OO,A_cell)
    pos_LU=[1,2];
    # pos_RU=[2,2];
    # pos_LD=[1,1];
    # pos_RD=[2,1];

    @tensor A_LU[:]:= A_cell[pos_LU[1],pos_LU[2]][-1,-2,-3,-4,1]*OO[-5,1]
    AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LU[1],pos_LU[2]]',A_LU);

    return AA_LU_double
end

function prepare_RU(OO,A_cell)
    # pos_LU=[1,2];
    pos_RU=[2,2];
    # pos_LD=[1,1];
    # pos_RD=[2,1];
    @tensor A_RU[:]:= A_cell[pos_RU[1],pos_RU[2]][-1,-2,-3,-4,1]*OO[-5,1]
    AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1],pos_RU[2]]',A_RU);
    return AA_RU_double
end
function prepare_LD(OO,A_cell)
    # pos_LU=[1,2];
    # pos_RU=[2,2];
    pos_LD=[1,1];
    # pos_RD=[2,1];
    @tensor A_LD[:]:= A_cell[pos_LD[1],pos_LD[2]][-1,-2,-3,-4,1]*OO[-5,1]
    AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LD[1],pos_LD[2]]',A_LD);
    return AA_LD_double
end
function prepare_RD(OO,A_cell)
    # pos_LU=[1,2];
    # pos_RU=[2,2];
    # pos_LD=[1,1];
    pos_RD=[2,1];
    @tensor A_RD[:]:= A_cell[pos_RD[1],pos_RD[2]][-1,-2,-3,-4,1]*OO[-5,1]
    AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1],pos_RD[2]]',A_RD);
    return AA_RD_double
end

function prepare_hopping_x_LD_RD(O1,O2,A_cell)
    # pos_LU=[1,2];
    # pos_RU=[2,2];
    pos_LD=[1,1];
    pos_RD=[2,1];

    @tensor A_LD[:]:= A_cell[pos_LD[1],pos_LD[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RD[:]:= A_cell[pos_RD[1],pos_RD[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
        
    gate=@ignore_derivatives parity_gate(A_LD,1); 
    @tensor A_LD[:]:=A_LD[1,-2,-3,-4,-5,-6]*gate[-1,1];
    gate=@ignore_derivatives parity_gate(A_LD,2); 
    @tensor A_LD[:]:=A_LD[-1,1,-3,-4,-5,-6]*gate[-2,1];
    gate=@ignore_derivatives parity_gate(A_LD,4); 
    @tensor A_LD[:]:=A_LD[-1,-2,-3,1,-5,-6]*gate[-4,1];

    gate=@ignore_derivatives parity_gate(A_RD,1); 
    @tensor A_RD[:]:=A_RD[1,-2,-3,-4,-5,-6]*gate[-1,1];
    gate=@ignore_derivatives parity_gate(A_RD,4); 
    @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,-6]*gate[-4,1];

    U=@ignore_derivatives unitary(fuse(space(A_LD,3)⊗space(A_LD,6)), space(A_LD,3)⊗space(A_LD,6)); 
    @tensor A_LD[:]:=A_LD[-1,-2,1,-4,-5,2]*U[-3,1,2];
    @tensor A_RD[:]:=A_RD[1,-2,-3,-4,-5,2]*U'[1,2,-1];

    AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LD[1],pos_LD[2]]',A_LD);
    AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1],pos_RD[2]]',A_RD);

    return AA_LD_double,AA_RD_double
end


function prepare_hopping_x_LU_RU(O1,O2,A_cell)

    pos_LU=[1,2];
    pos_RU=[2,2];

    @tensor A_LU[:]:= A_cell[pos_LU[1],pos_LU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RU[:]:= A_cell[pos_RU[1],pos_RU[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
        
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

    AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LU[1],pos_LU[2]]',A_LU);
    AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1],pos_RU[2]]',A_RU);

    return AA_LU_double,AA_RU_double
end


function prepare_hopping_y_RU_RD(O1,O2,A_cell)
    # pos_LU=[1,2];
    pos_RU=[2,2];
    # pos_LD=[1,1];
    pos_RD=[2,1];


    #the first index of O is dummy
    @tensor A_RU[:]:= A_cell[pos_RU[1],pos_RU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RD[:]:= A_cell[pos_RD[1],pos_RD[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
    
    gate=@ignore_derivatives parity_gate(A_RU,1); 
    @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,-6]*gate[-1,1];
    gate=@ignore_derivatives parity_gate(A_RU,2); 
    @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,-6]*gate[-2,1];
    gate=@ignore_derivatives parity_gate(A_RU,4); 
    @tensor A_RU[:]:=A_RU[-1,-2,-3,1,-5,-6]*gate[-4,1];

    U1=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
    @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U1[-2,1,2];
    @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,2]*U1'[1,2,-4];

    AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1],pos_RD[2]]',A_RD);
    AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1],pos_RU[2]]',A_RU);

    return AA_RD_double,AA_RU_double
end


function prepare_hopping_y_LU_LD(O1,O2,A_cell)

    pos_LU=[1,2];
    pos_LD=[1,1];


    #the first index of O is dummy
    @tensor A_LU[:]:= A_cell[pos_LU[1],pos_LU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_LD[:]:= A_cell[pos_LD[1],pos_LD[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
    
    gate=@ignore_derivatives parity_gate(A_LU,1); 
    @tensor A_LU[:]:=A_LU[1,-2,-3,-4,-5,-6]*gate[-1,1];
    gate=@ignore_derivatives parity_gate(A_LU,2); 
    @tensor A_LU[:]:=A_LU[-1,1,-3,-4,-5,-6]*gate[-2,1];
    gate=@ignore_derivatives parity_gate(A_LU,4); 
    @tensor A_LU[:]:=A_LU[-1,-2,-3,1,-5,-6]*gate[-4,1];

    U1=@ignore_derivatives unitary(fuse(space(A_LU,2)⊗space(A_LU,6)), space(A_LU,2)⊗space(A_LU,6)); 
    @tensor A_LU[:]:=A_LU[-1,1,-3,-4,-5,2]*U1[-2,1,2];
    @tensor A_LD[:]:=A_LD[-1,-2,-3,1,-5,2]*U1'[1,2,-4];

    AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LD[1],pos_LD[2]]',A_LD);
    AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LU[1],pos_LU[2]]',A_LU);

    return AA_LD_double,AA_LU_double
end

function prepare_hopping_LD_RU(O1,O2,A_cell)
    # pos_LU=[1,2];
    pos_RU=[2,2];
    pos_LD=[1,1];
    pos_RD=[2,1];


    @tensor A_LD[:]:= A_cell[pos_LD[1],pos_LD[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RU[:]:= A_cell[pos_RU[1],pos_RU[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
    # @tensor A_LD[:]:= A_cell[pos_LD[1]][pos_LD[2]][-1,-2,-3,-4,1]*O1[-5,1]
    # @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-5,1]
    O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));


    gate=@ignore_derivatives parity_gate(A_LD,1); 
    @tensor A_LD[:]:=A_LD[1,-2,-3,-4,-5,-6]*gate[-1,1];
    gate=@ignore_derivatives parity_gate(A_LD,2); 
    @tensor A_LD[:]:=A_LD[-1,1,-3,-4,-5,-6]*gate[-2,1];
    gate=@ignore_derivatives parity_gate(A_LD,4); 
    @tensor A_LD[:]:=A_LD[-1,-2,-3,1,-5,-6]*gate[-4,1];

    gate=@ignore_derivatives parity_gate(A_cell[pos_RD[1],pos_RD[2]],2); 
    @tensor A_RD[:]:=A_cell[pos_RD[1],pos_RD[2]][-1,1,-3,-4,-5]*gate[-2,1];
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



    AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LD[1],pos_LD[2]]',A_LD);
    AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1],pos_RU[2]]',A_RU);
    AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1],pos_RD[2]]',A_RD);

    return AA_LD_double,  AA_RU_double,   AA_RD_double
end

function norm_ob_2x2(mps_bot_set,mps_top_set, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range)

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################

    Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);

    return Norm_
end

function compute_ob_2x2_triangle(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);

    AA_LD,AA_RD=prepare_hopping_x_LD_RD(Cdag_set[mod1(x_range[1]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#tx
    ex=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,AA_RD);
    ex=ex/Norm_;

    AA_RD,AA_RU=prepare_hopping_y_RU_RD(Cdag_set[mod1(x_range[2]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#ty
    ey=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],AA_RD);
    ey=ey/Norm_;

    AA_LD, AA_RU, AA_RD=prepare_hopping_LD_RU(Cdag_set[mod1(x_range[1]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#t_ld_ru
    e_ld_ru=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,AA_LD,AA_RD);
    e_ld_ru=e_ld_ru/Norm_;

    AA_RD=prepare_RD(N_occu_set[mod1(x_range[1]-1,2)],iPEPS_2x2);#e0
    occu=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD);
    occu=occu/Norm_;
    
    AA_RD=prepare_RD(n_double_set[mod1(x_range[2]-1,2)]-(1/2)*N_occu_set[mod1(x_range[2]-1,2)]+(1/4)*Ident_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#U
    eU=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD);
    eU=eU/Norm_;

    return ex,ey,e_ld_ru,occu,eU
end





function compute_ob_2x2_test(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    return Norm_
 

    # AA_LD, AA_RU, AA_RD=prepare_hopping_LD_RU(Cdag_set[mod1(x_range[1]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#t_ld_ru
    # e_ld_ru=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,AA_LD,AA_RD);
    #e_ld_ru=e_ld_ru/Norm_;

    # return e_ld_ru
end



function compute_ob_2x2_top(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);

    AA_LU,AA_RU=prepare_hopping_x_LU_RU(Cdag_set[mod1(x_range[1]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#tx
    ex=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,AA_LU,AA_RU,iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    ex=ex/Norm_;

    AA_RU=prepare_RU(N_occu_set[mod1(x_range[1]-1,2)],iPEPS_2x2);#e0
    occu=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    occu=occu/Norm_;
    
    AA_RU=prepare_RU(n_double_set[mod1(x_range[2]-1,2)]-(1/2)*N_occu_set[mod1(x_range[2]-1,2)]+(1/4)*Ident_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#U
    eU=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    eU=eU/Norm_;

    return ex,occu,eU
end


function compute_ob_2x2_left(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);

    AA_LD,AA_LU=prepare_hopping_y_LU_LD(Cdag_set[mod1(x_range[2]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#ty
    ey=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,AA_LU,iPEPS_double_2x2[2,2],AA_LD,iPEPS_double_2x2[2,1]);
    ey=ey/Norm_;

    AA_LD=prepare_LD(N_occu_set[mod1(x_range[1]-1,2)],iPEPS_2x2);#e0
    occu=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,iPEPS_double_2x2[2,1]);
    occu=occu/Norm_;
    
    AA_LD=prepare_LD(n_double_set[mod1(x_range[2]-1,2)]-(1/2)*N_occu_set[mod1(x_range[2]-1,2)]+(1/4)*Ident_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#U
    eU=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,iPEPS_double_2x2[2,1]);
    eU=eU/Norm_;

    return ey,occu,eU
end


function compute_ob_2x2_left_top(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);


    AA_LU=prepare_LU(N_occu_set[mod1(x_range[1]-1,2)],iPEPS_2x2);#e0
    occu=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,AA_LU,iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    occu=occu/Norm_;
    
    AA_LU=prepare_LU(n_double_set[mod1(x_range[2]-1,2)]-(1/2)*N_occu_set[mod1(x_range[2]-1,2)]+(1/4)*Ident_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#U
    eU=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,AA_LU,iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    eU=eU/Norm_;

    return occu,eU
end

##################################################################################################################################################

function compute_ob_2x2_triangle_new(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    # mps_bot=mps_bot_set[y_range[1]-1];
    # mps_top=mps_top_set[y_range[2]+1];

    # VL0=VL_set_set[y_range[1]][x_range[1]-1];
    # VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    #Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    Norm_=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)


    AA_LD,AA_RD=prepare_hopping_x_LD_RD(Cdag_set[mod1(x_range[1]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#tx
    AA_LD=remove_trivial_boundary(AA_LD,x_range[1],y_range[1],Lx,Ly);
    AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    #ex=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,AA_RD);
    ex=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    ex=ex/Norm_;

    AA_RD,AA_RU=prepare_hopping_y_RU_RD(Cdag_set[mod1(x_range[2]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#ty
    AA_RU=remove_trivial_boundary(AA_RU,x_range[2],y_range[2],Lx,Ly);
    AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    #ey=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],AA_RD);
    ey=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    ey=ey/Norm_;

    AA_LD, AA_RU, AA_RD=prepare_hopping_LD_RU(Cdag_set[mod1(x_range[1]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#t_ld_ru
    AA_RU=remove_trivial_boundary(AA_RU,x_range[2],y_range[2],Lx,Ly);
    AA_LD=remove_trivial_boundary(AA_LD,x_range[1],y_range[1],Lx,Ly);
    AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    #e_ld_ru=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,AA_LD,AA_RD);
    e_ld_ru=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    e_ld_ru=e_ld_ru/Norm_;

    AA_RD=prepare_RD(N_occu_set[mod1(x_range[1]-1,2)],iPEPS_2x2);#e0
    AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    #occu=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD);
    occu=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    occu=occu/Norm_;
    
    AA_RD=prepare_RD(n_double_set[mod1(x_range[2]-1,2)]-(1/2)*N_occu_set[mod1(x_range[2]-1,2)]+(1/4)*Ident_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#U
    AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    #eU=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD);
    eU=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    eU=eU/Norm_;

    return ex,ey,e_ld_ru,occu,eU
end

function compute_ob_2x2_triangle_new_test(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    # mps_bot=mps_bot_set[y_range[1]-1];
    # mps_top=mps_top_set[y_range[2]+1];

    # VL0=VL_set_set[y_range[1]][x_range[1]-1];
    # VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    #Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    Norm_=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    return Norm_

    # AA_LD,AA_RD=prepare_hopping_x_LD_RD(Cdag_set[mod1(x_range[1]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#tx
    # AA_LD=remove_trivial_boundary(AA_LD,x_range[1],y_range[1],Lx,Ly);
    # AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    # #ex=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,AA_RD);
    # ex=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    # ex=ex/Norm_;

    # AA_RD,AA_RU=prepare_hopping_y_RU_RD(Cdag_set[mod1(x_range[2]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#ty
    # AA_RU=remove_trivial_boundary(AA_RU,x_range[2],y_range[2],Lx,Ly);
    # AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    # #ey=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],AA_RD);
    # ey=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    # ey=ey/Norm_;

    # AA_LD, AA_RU, AA_RD=prepare_hopping_LD_RU(Cdag_set[mod1(x_range[1]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#t_ld_ru
    # AA_RU=remove_trivial_boundary(AA_RU,x_range[2],y_range[2],Lx,Ly);
    # AA_LD=remove_trivial_boundary(AA_LD,x_range[1],y_range[1],Lx,Ly);
    # AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    # #e_ld_ru=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,AA_LD,AA_RD);
    # e_ld_ru=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    # e_ld_ru=e_ld_ru/Norm_;

    # AA_RD=prepare_RD(N_occu_set[mod1(x_range[1]-1,2)],iPEPS_2x2);#e0
    # AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    # #occu=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD);
    # occu=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    # occu=occu/Norm_;
    
    # AA_RD=prepare_RD(n_double_set[mod1(x_range[2]-1,2)]-(1/2)*N_occu_set[mod1(x_range[2]-1,2)]+(1/4)*Ident_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#U
    # AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);
    # #eU=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD);
    # eU=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    # eU=eU/Norm_;

    # return ex,ey,e_ld_ru,occu,eU
end

function compute_ob_2x2_top_new(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    # mps_bot=mps_bot_set[y_range[1]-1];
    # mps_top=mps_top_set[y_range[2]+1];

    # VL0=VL_set_set[y_range[1]][x_range[1]-1];
    # VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    #Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    Norm_=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)


    AA_LU,AA_RU=prepare_hopping_x_LU_RU(Cdag_set[mod1(x_range[1]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#tx
    AA_LU=remove_trivial_boundary(AA_LU,x_range[1],y_range[2],Lx,Ly);
    AA_RU=remove_trivial_boundary(AA_RU,x_range[2],y_range[2],Lx,Ly);
    #ex=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,AA_LU,AA_RU,iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    ex=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,AA_LU,AA_RU,iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
    ex=ex/Norm_;

    AA_RU=prepare_RU(N_occu_set[mod1(x_range[1]-1,2)],iPEPS_2x2);#e0
    AA_RU=remove_trivial_boundary(AA_RU,x_range[2],y_range[2],Lx,Ly);
    #occu=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    occu=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    occu=occu/Norm_;
    
    AA_RU=prepare_RU(n_double_set[mod1(x_range[2]-1,2)]-(1/2)*N_occu_set[mod1(x_range[2]-1,2)]+(1/4)*Ident_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#U
    AA_RU=remove_trivial_boundary(AA_RU,x_range[2],y_range[2],Lx,Ly);
    #eU=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    eU=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],AA_RU,iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    eU=eU/Norm_;

    return ex,occu,eU
end


function compute_ob_2x2_left_new(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    # mps_bot=mps_bot_set[y_range[1]-1];
    # mps_top=mps_top_set[y_range[2]+1];

    # VL0=VL_set_set[y_range[1]][x_range[1]-1];
    # VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    #Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    Norm_=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)

    AA_LD,AA_LU=prepare_hopping_y_LU_LD(Cdag_set[mod1(x_range[2]-1,2)],C_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#ty
    AA_LU=remove_trivial_boundary(AA_LU,x_range[1],y_range[2],Lx,Ly);
    AA_LD=remove_trivial_boundary(AA_LD,x_range[1],y_range[1],Lx,Ly);
    #ey=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,AA_LU,iPEPS_double_2x2[2,2],AA_LD,iPEPS_double_2x2[2,1]);
    ey=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,AA_LU,iPEPS_double_2x2[2,2],AA_LD,iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    ey=ey/Norm_;

    AA_LD=prepare_LD(N_occu_set[mod1(x_range[1]-1,2)],iPEPS_2x2);#e0
    AA_LD=remove_trivial_boundary(AA_LD,x_range[1],y_range[1],Lx,Ly);
    #occu=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,iPEPS_double_2x2[2,1]);
    occu=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    occu=occu/Norm_;
    
    AA_LD=prepare_LD(n_double_set[mod1(x_range[2]-1,2)]-(1/2)*N_occu_set[mod1(x_range[2]-1,2)]+(1/4)*Ident_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#U
    AA_LD=remove_trivial_boundary(AA_LD,x_range[1],y_range[1],Lx,Ly);
    #eU=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,iPEPS_double_2x2[2,1]);
    eU=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],AA_LD,iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    eU=eU/Norm_;

    return ey,occu,eU
end


function compute_ob_2x2_left_top_new(mps_bot_set,mps_top_set,iPEPS_2x2_, iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    iPEPS_2x2=[iPEPS_2x2_[1,1] iPEPS_2x2_[1,2]; iPEPS_2x2_[2,1] iPEPS_2x2_[2,2]];

    # mps_bot=mps_bot_set[y_range[1]-1];
    # mps_top=mps_top_set[y_range[2]+1];

    # VL0=VL_set_set[y_range[1]][x_range[1]-1];
    # VR0=VR_set_set[y_range[1]][x_range[2]+1];
    ###############################################
    
    if isa(space(psi_double[1,1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        #Hamiltonian_terms=Hamiltonians_spinless_Z2;
        Hamiltonian_terms=Hamiltonians_spinful_Z2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(psi_double[1,1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(psi_double[1,1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end

    Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
    ##########################


    #Norm_=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    Norm_=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2[1,2],iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)


    AA_LU=prepare_LU(N_occu_set[mod1(x_range[1]-1,2)],iPEPS_2x2);#e0
    AA_LU=remove_trivial_boundary(AA_LU,x_range[1],y_range[2],Lx,Ly);
    #occu=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,AA_LU,iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    occu=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,AA_LU,iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    occu=occu/Norm_;
    
    AA_LU=prepare_LU(n_double_set[mod1(x_range[2]-1,2)]-(1/2)*N_occu_set[mod1(x_range[2]-1,2)]+(1/4)*Ident_set[mod1(x_range[2]-1,2)],iPEPS_2x2);#U
    AA_LU=remove_trivial_boundary(AA_LU,x_range[1],y_range[2],Lx,Ly);
    #eU=contract_2x2(VL0,VR0,x_range,y_range,mps_top,mps_bot,AA_LU,iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1]);
    eU=build_fermi_cluster_2x2(mps_bot_set,mps_top_set,AA_LU,iPEPS_double_2x2[2,2],iPEPS_double_2x2[1,1],iPEPS_double_2x2[2,1], VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    eU=eU/Norm_;

    return occu,eU
end
