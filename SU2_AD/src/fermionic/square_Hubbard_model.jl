function Hamiltonians_spinless_Z2()
    

    #Vdummy=ℂ[U1Irrep](-1=>1);
    #V=ℂ[U1Irrep](0=>1,1=>1);

    Vdummy=Rep[ℤ₂](1=>1);
    V=Rep[ℤ₂](0=>1,1=>1);

    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    Ident=TensorMap(Id,  V  ←  V);

    occu=TensorMap(occu,  V ←  V);
    

    Cdag=zeros(1,2,2);
    Cdag[1,:,:]=sp;
    Cdag=TensorMap(Cdag, Vdummy ⊗ V ← V);


    C=zeros(1,2,2);
    C[1,:,:]=sm;
    C=TensorMap(C, Vdummy' ⊗ V ← V);

    Cdag_=zeros(1,2,2);
    Cdag_[1,:,:]=sp;
    Cdag_=TensorMap(Cdag_, Vdummy' ⊗ V ← V);


    return Ident, occu, Cdag, C, Cdag_ 
end

function ob_2x2(CTM,AA_LU,AA_RU,AA_LD,AA_RD)

    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_LU[4,-2,-4,3]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_LD[3,4,-4,-2]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4]; 


    @tensor up[:]:=MM_LU[-1,-2,1,2]*MM_RU[1,2,-3,-4];
    @tensor down[:]:=MM_LD[-1,-2,1,2]*MM_RD[1,2,-3,-4];
    Norm=@tensor up[1,2,3,4]*down[1,2,3,4];

    return Norm
end


function hopping_x(CTM,O1,O2,A,AA,ctm_setting)

    @tensor A_LU[:]:= A[-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RU[:]:= A[-1,-2,-3,-4,1]*O2[-6,-5,1]
   

        
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

    if ctm_setting.grad_checkpoint
        AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_LU);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_RU);
    else
        AA_LU_double,_,_,_,_=build_double_layer_swap(A',A_LU);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A',A_RU);
    end
    


    ob=ob_2x2(CTM,AA_LU_double,AA_RU_double,AA,AA);
    Norm=ob_2x2(CTM,AA,AA,AA,AA);
    ob=ob/Norm;
    return ob
end

function hopping_y(CTM,O1,O2,A,AA,ctm_setting)


    #the first index of O is dummy
    @tensor A_RU[:]:= A[-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RD[:]:= A[-1,-2,-3,-4,1]*O2[-6,-5,1]


    
    gate=@ignore_derivatives parity_gate(A_RU,1); 
    @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,-6]*gate[-1,1];
    gate=@ignore_derivatives parity_gate(A_RU,2); 
    @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,-6]*gate[-2,1];
    gate=@ignore_derivatives parity_gate(A_RU,4); 
    @tensor A_RU[:]:=A_RU[-1,-2,-3,1,-5,-6]*gate[-4,1];


    U1=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
    @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U1[-2,1,2];
    @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,2]*U1'[1,2,-4];

    if ctm_setting.grad_checkpoint
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_RD);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_RU);
    else
        AA_RD_double,_,_,_,_=build_double_layer_swap(A',A_RD);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A',A_RU);
    end

    
    ob=ob_2x2(CTM,AA,AA_RU_double,AA,AA_RD_double);
    Norm=ob_2x2(CTM,AA,AA,AA,AA);
    ob=ob/Norm;
    return ob
end


function hopping_right_top(CTM,O1,O2,A,AA,ctm_setting)

    # @tensor A_LU[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-5,1]
    # @tensor A_RD[:]:= A_cell[pos_RD[1]][pos_RD[2]][-1,-2,-3,-4,1]*O2[-5,1]
    @tensor A_LU[:]:= A[-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RD[:]:= A[-1,-2,-3,-4,1]*O2[-6,-5,1]
    O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));
    


    gate=@ignore_derivatives parity_gate(A_LU,1); 
    @tensor A_LU[:]:=A_LU[1,-2,-3,-4,-5,-6]*gate[-1,1];
    gate=@ignore_derivatives parity_gate(A_LU,2); 
    @tensor A_LU[:]:=A_LU[-1,1,-3,-4,-5,-6]*gate[-2,1];
    gate=@ignore_derivatives parity_gate(A_LU,4); 
    @tensor A_LU[:]:=A_LU[-1,-2,-3,1,-5,-6]*gate[-4,1];

    gate=@ignore_derivatives parity_gate(A_RD,4); 
    @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,-6]*gate[-4,1];


        
    U1=@ignore_derivatives unitary(fuse(space(A_LU,3)⊗space(A_LU,6)), space(A_LU,3)⊗space(A_LU,6)); 
    U2=@ignore_derivatives unitary(fuse(space(A_RD,4)⊗space(A_RD,6)), space(A_RD,4)⊗space(A_RD,6)); 
    @tensor A_LU[:]:=A_LU[-1,-2,1,-4,-5,2]*U1[-3,1,2];
    @tensor A_RU[:]:=A[1,3,-3,-4,-5]*O_string[4,2]*U1'[1,2,-1]*U2'[3,4,-2];
    @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,2]*U2[-4,1,2];



    if ctm_setting.grad_checkpoint
        AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_LU);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_RU);
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_RD);
    else
        AA_LU_double,_,_,_,_=build_double_layer_swap(A',A_LU);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A',A_RU);
        AA_RD_double,_,_,_,_=build_double_layer_swap(A',A_RD);
    end


    ob=ob_2x2(CTM,AA_LU_double,AA_RU_double,AA,AA_RD_double);
    Norm=ob_2x2(CTM,AA,AA,AA,AA);
    ob=ob/Norm;
    return ob    
end


function hopping_right_bot(CTM,O1,O2,A,AA,ctm_setting)

    @tensor A_LD[:]:= A[-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RU[:]:= A[-1,-2,-3,-4,1]*O2[-6,-5,1]
    # @tensor A_LD[:]:= A_cell[pos_LD[1]][pos_LD[2]][-1,-2,-3,-4,1]*O1[-5,1]
    # @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-5,1]
    O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));


    gate=@ignore_derivatives parity_gate(A_LD,1); 
    @tensor A_LD[:]:=A_LD[1,-2,-3,-4,-5,-6]*gate[-1,1];
    gate=@ignore_derivatives parity_gate(A_LD,2); 
    @tensor A_LD[:]:=A_LD[-1,1,-3,-4,-5,-6]*gate[-2,1];
    gate=@ignore_derivatives parity_gate(A_LD,4); 
    @tensor A_LD[:]:=A_LD[-1,-2,-3,1,-5,-6]*gate[-4,1];

    gate=@ignore_derivatives parity_gate(A,2); 
    @tensor A_RD[:]:=A[-1,1,-3,-4,-5]*gate[-2,1];
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
        AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_LD);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_RU);
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A_RD);
    else
        AA_LD_double,_,_,_,_=build_double_layer_swap(A',A_LD);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A',A_RU);
        AA_RD_double,_,_,_,_=build_double_layer_swap(A',A_RD);
    end

    ob=ob_2x2(CTM,AA,AA_RU_double,AA_LD_double,AA_RD_double);
    Norm=ob_2x2(CTM,AA,AA,AA,AA);
    ob=ob/Norm;
    return ob        
end


function ob_onsite(CTM,O1,A,AA,ctm_setting)
    
    @tensor A1[:]:= A[-1,-2,-3,-4,1]*O1[-5,1]

    if ctm_setting.grad_checkpoint
        A1_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A',A1);
    else
        A1_double,_,_,_,_=build_double_layer_swap(A',A1);
    end
    

    ob=ob_2x2(CTM,AA,A1_double,AA,AA);
    Norm=ob_2x2(CTM,AA,AA,AA,AA);
    ob=ob/Norm;
    return ob
end



# function Hamiltonians_spinless()
#     # Heisenberg interaction
#     #Rep[ℤ₂](1=>1)
#     #Vp=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1);
#     Vp=SU2Space(1/2=>1);
#     U_phy=@ignore_derivatives unitary(fuse(Vp*Vp*Vp),Vp*Vp*Vp);
#     Id=I(2);
#     sx=[[0,1] [1,0]]/2; sy=[[0,1] [-1,0]]/2*im; sz=[[1,0] [0,-1]]/2;
#     @tensor H12[:]:=sx[-1,-4]*sx[-2,-5]*Id[-3,-6]+sy[-1,-4]*sy[-2,-5]*Id[-3,-6]+sz[-1,-4]*sz[-2,-5]*Id[-3,-6];
#     @tensor H31[:]:=sx[-1,-4]*Id[-2,-5]*sx[-3,-6]+sy[-1,-4]*Id[-2,-5]*sy[-3,-6]+sz[-1,-4]*Id[-2,-5]*sz[-3,-6];
#     @tensor H23[:]:=Id[-1,-4]*sx[-2,-5]*sx[-3,-6]+Id[-1,-4]*sy[-2,-5]*sy[-3,-6]+Id[-1,-4]*sz[-2,-5]*sz[-3,-6];
#     @tensor H123chiral[:]:=sx[-1,-4]*sy[-2,-5]*sz[-3,-6]-sx[-1,-4]*sz[-2,-5]*sy[-3,-6]+sy[-1,-4]*sz[-2,-5]*sx[-3,-6]-sy[-1,-4]*sx[-2,-5]*sz[-3,-6]+sz[-1,-4]*sx[-2,-5]*sy[-3,-6]-sz[-1,-4]*sy[-2,-5]*sx[-3,-6];
#     H12_tensorkit=TensorMap(H12, domain(U_phy)' ← domain(U_phy)');
#     H31_tensorkit=TensorMap(H31, domain(U_phy)' ← domain(U_phy)');
#     H23_tensorkit=TensorMap(H23, domain(U_phy)' ← domain(U_phy)');
#     H123chiral_tensorkit=TensorMap(H123chiral, domain(U_phy)' ← domain(U_phy)');
#     # @tensor H12_tensorkit[:]:=U_phy'[4,5,6,-1]*H12_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
#     # @tensor H31_tensorkit[:]:=U_phy'[4,5,6,-1]*H31_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
#     # @tensor H23_tensorkit[:]:=U_phy'[4,5,6,-1]*H23_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];
#     #@tensor H123chiral_tensorkit[:]:=U_phy'[4,5,6,-1]*H123chiral_tensorkit[1,2,3,4,5,6]*U_phy[-2,1,2,3];

#     @tensor H_Heisenberg[:]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4];
#     H_Heisenberg=TensorMap(H_Heisenberg, Vp'*Vp' ← Vp'*Vp');
#     H_Heisenberg=permute(H_Heisenberg,(1,2,3,4,));

#     return H_Heisenberg, H123chiral_tensorkit, H12_tensorkit, H31_tensorkit, H23_tensorkit 
# end



# function evaluate_triangle_fpeps(H_triangle, A::TensorMap, AA, U_L,U_D,U_R,U_U, CTM, ctm_setting)

#     AA_open, U_s_s=build_double_layer_swap_open(A',A);
#     U_s_s=U_s_s';
    

#     rho_LU_RU_LD=ob_LU_RU_LD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
#     @tensor rho_LU_RU_LD[:]:=rho_LU_RU_LD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
#     norm_LU_RU_LD=@tensor rho_LU_RU_LD[1,2,3,1,2,3];
#     E_LU_RU_LD=@tensor rho_LU_RU_LD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
#     E_LU_RU_LD=E_LU_RU_LD/norm_LU_RU_LD; 
    

#     rho_LD_RU_RD=ob_LD_RU_RD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
#     rho_LD_RU_RD=permute(rho_LD_RU_RD,(3,1,2,));
#     @tensor rho_LD_RU_RD[:]:=rho_LD_RU_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
#     norm_LD_RU_RD=@tensor rho_LD_RU_RD[1,2,3,1,2,3];
#     E_LD_RU_RD=@tensor rho_LD_RU_RD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
#     E_LD_RU_RD=E_LD_RU_RD/norm_LD_RU_RD ; 
    

#     rho_LU_LD_RD=ob_LU_LD_RD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
#     rho_LU_LD_RD=permute(rho_LU_LD_RD,(2,1,3,));
#     @tensor rho_LU_LD_RD[:]:=rho_LU_LD_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
#     norm_LU_LD_RD=@tensor rho_LU_LD_RD[1,2,3,1,2,3];
#     E_LU_LD_RD=@tensor rho_LU_LD_RD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
#     E_LU_LD_RD=E_LU_LD_RD/norm_LU_LD_RD ; 
    

#     rho_LU_RU_RD=ob_LU_RU_RD(CTM,AA,AA_open,AA_open,AA_open);  #clockwise
#     rho_LU_RU_RD=permute(rho_LU_RU_RD,(2,3,1,));
#     @tensor rho_LU_RU_RD[:]:=rho_LU_RU_RD[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3];#s1',s2',s1,s2
#     norm_LU_RU_RD=@tensor rho_LU_RU_RD[1,2,3,1,2,3];
#     E_LU_RU_RD=@tensor rho_LU_RU_RD[1,2,3,4,5,6]*H_triangle[1,2,3,4,5,6];
#     E_LU_RU_RD=E_LU_RU_RD/norm_LU_RU_RD;
        
#     return E_LU_RU_LD, E_LD_RU_RD, E_LU_LD_RD, E_LU_RU_RD
# end




# function ob_LU_RU_LD(CTM,AA_fused,AA_LU,AA_RU,AA_LD)
#     Cset=CTM.Cset;
#     Tset=CTM.Tset;

#     @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-4]*Tset.T4[-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
#     @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

#     @tensor MM_LD[:]:=Tset.T4[1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset.C4[2,1]*Tset.T3[-4,4,2]; 
#     @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

#     MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
#     MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:= up[-1,1,2,3,4,-2]*down[-3,1,2,3,4];
#     return rho
# end


# function ob_LD_RU_RD(CTM,AA_fused,AA_LD,AA_RU,AA_RD)
#     Cset=CTM.Cset;
#     Tset=CTM.Tset;

#     @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_fused[4,-2,-4,3]; 
#     @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

#     @tensor MM_LD[:]:=Tset.T4[1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset.C4[2,1]*Tset.T3[-4,4,2]; 
#     @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

#     MM_LU=permute(MM_LU,(1,2,),(3,4,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
#     MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:= up[1,2,3,4,-2]*down[-1,1,2,3,4,-3];
#     return rho
# end


# function ob_LU_LD_RD(CTM,AA_fused,AA_LU,AA_LD,AA_RD)
#     Cset=CTM.Cset;
#     Tset=CTM.Tset;

#     @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-4]*Tset.T4[-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
#     @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

#     @tensor MM_LD[:]:=Tset.T4[1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset.C4[2,1]*Tset.T3[-4,4,2]; 
#     @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

#     MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,));
#     MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:= up[-1,1,2,3,4]*down[-2,1,2,3,4,-3];
#     return rho
# end



# function ob_LU_RU_RD(CTM,AA_fused,AA_LU,AA_RU,AA_RD)
#     Cset=CTM.Cset;
#     Tset=CTM.Tset;

#     @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-4]*Tset.T4[-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
#     @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_RU[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

#     @tensor MM_LD[:]:=Tset.T4[1,3,-2]*AA_fused[3,4,-5,-3]*Cset.C4[2,1]*Tset.T3[-4,4,2]; 
#     @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4,-5]; 

#     MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
#     MM_LD=permute(MM_LD,(1,2,),(3,4,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:= up[-1,1,2,3,4,-2]*down[1,2,3,4,-3];
#     return rho
# end