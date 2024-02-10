function Rank(T::TensorMap)
    return length(space(T).domain)+length(space(T).codomain)
end
function Hamiltonians_spinless_U1_SU2_2site(M)
    global U_phy1,U_phy2
    if M==1
        U_phy1=unitary(Rep[U₁ × SU₂]((2, 0)=>1, (4, 0)=>3, (6, 0)=>1, (3, 1/2)=>2, (5, 1/2)=>2, (4, 1)=>1) ← (Rep[U₁ × SU₂]((6, 0)=>1) ⊗ Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)' ⊗ Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'));
        U_phy2=unitary(Rep[U₁ × SU₂]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (1, 1/2)=>2, (-1, 1/2)=>2, (0, 1)=>1) ← (Rep[U₁ × SU₂]((2, 0)=>1, (4, 0)=>3, (6, 0)=>1, (3, 1/2)=>2, (5, 1/2)=>2, (4, 1)=>1) ⊗ Rep[U₁× SU₂]((-2, 0)=>1) ⊗ Rep[U₁ × SU₂]((-2, 0)=>1)));
        Vdummy=Rep[U₁ × SU₂]((1,1/2)=>1)';
        V=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)';
    elseif M==2
        U_phy1=unitary(Rep[U₁](3=>1, 4=>2, 5=>1) ← (Rep[U₁](-5=>1)' ⊗ Rep[U₁](0=>1, 1=>1)' ⊗ Rep[U₁](0=>1, 1=>1)'));
        U_phy2=unitary(Rep[U₁](0=>2, 1=>1, -1=>1) ← (Rep[U₁](3=>1, 4=>2, 5=>1) ⊗ Rep[U₁](1=>1)' ⊗ Rep[U₁](1=>1)' ⊗ Rep[U₁](1=>1)' ⊗ Rep[U₁](1=>1)'));
        Vdummy=ℂ[U1Irrep](-1=>1)';
        V=ℂ[U1Irrep](0=>1,1=>1)';
    end

    
    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order: (0,0), (0,1), (1,0), (1,1)
    
    Ident=kron(Id,Id);
    Ident2=Ident[[1,4,3,2],[1,4,3,2]];
    @tensor Ident4[:]:=Ident2[-1,-3]*Ident2[-2,-4];
    Ident4=TensorMap(Ident4,  V*V  ←  V*V);

    sz_string=kron(sz,sz);
    sz_string=sz_string[[1,4,3,2],[1,4,3,2]];

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=N_occu[[1,4,3,2],[1,4,3,2]];
    @tensor NA[:]:=N_occu[-1,-3]*Ident2[-2,-4];
    @tensor NB[:]:=Ident2[-1,-3]*N_occu[-2,-4];
    NA=TensorMap(NA,  V*V  ←  V*V);
    NB=TensorMap(NB,  V*V  ←  V*V);
    n_double=kron(occu,occu);
    n_double=n_double[[1,4,3,2],[1,4,3,2]];
    @tensor n_double_A[:]:=n_double[-1,-3]*Ident2[-2,-4];
    @tensor n_double_B[:]:=Ident2[-1,-3]*n_double[-2,-4];
    n_double_A=TensorMap(n_double_A,  V*V  ←  V*V);
    n_double_B=TensorMap(n_double_B,  V*V  ←  V*V);


    # method 1
    Cdagup=zeros(4,4,2);
    Cdagup[[1,4,3,2],[1,4,3,2],1]=kron(sp,Id);
    Cdagdn=zeros(4,4,2);
    Cdagdn[[1,4,3,2],[1,4,3,2],2]=kron(sz,sp);
    Cdag=Cdagup+Cdagdn;
    @tensor Cdag_A[:]:=Cdag[-1,-3,-5]*Ident2[-2,-4];
    @tensor Cdag_B[:]:=sz_string[-1,-3]*Cdag[-2,-4,-5];
    Cdag_A=TensorMap(Cdag_A,  V*V ← V*V *Vdummy);
    Cdag_B=TensorMap(Cdag_B,  V*V ← V*V *Vdummy);


    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=Cup+Cdn;
    @tensor C_A[:]:=C[-1,-2,-4]*Ident2[-3,-5];
    @tensor C_B[:]:=sz_string[-2,-4]*C[-1,-3,-5];
    C_A=TensorMap(C_A, Vdummy *V*V ← V*V);
    C_B=TensorMap(C_B, Vdummy *V*V ← V*V);






    ###############################

    if M==1
        @tensor Ident4[:]:=Ident4[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Ident4[:]:=Ident4[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor n_double_A[:]:=n_double_A[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor n_double_A[:]:=n_double_A[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor n_double_B[:]:=n_double_B[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor n_double_B[:]:=n_double_B[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
        


        @tensor Cdag_A[:]:=Cdag_A[1,2,4,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Cdag_A[:]:=Cdag_A[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
        Cdag_A=permute(Cdag_A,(3,1,2,));

        @tensor Cdag_B[:]:=Cdag_B[1,2,4,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Cdag_B[:]:=Cdag_B[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
        Cdag_B=permute(Cdag_B,(3,1,2,));

        @tensor C_A[:]:=C_A[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor C_A[:]:=C_A[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor C_B[:]:=C_B[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor C_B[:]:=C_B[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CdagA_CB[:]:=Cdag_A[2,-1,1]*C_B[2,1,-2];

    elseif M==2
        @tensor Ident[:]:=Ident[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Ident[:]:=Ident[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

        @tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

        @tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

        @tensor NANB[:]:=NANB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NANB[:]:=NANB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

        @tensor CAdag_CB[:]:=CAdag_CB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor CAdag_CB[:]:=CAdag_CB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
        

        @tensor CAdag[:]:=CAdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CAdag[:]:=CAdag[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];

        @tensor CBdag[:]:=CBdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CBdag[:]:=CBdag[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];

        @tensor CA[:]:=CA[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CA[:]:=CA[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];

        @tensor CB[:]:=CB[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CB[:]:=CB[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    end


    return Ident4, NA, NB, n_double_A, CdagA_CB, Cdag_A, C_A, Cdag_B, C_B 
end

function Hamiltonians_spinless_U1_2site(M)
    global U_phy1,U_phy2
    if M==1
        U_phy1=unitary(Rep[U₁](-1=>1, -2=>2, -3=>1) ← (Rep[U₁](-3=>1) ⊗ Rep[U₁](0=>1, 1=>1) ⊗ Rep[U₁](0=>1, 1=>1)));
        U_phy2=unitary(Rep[U₁](0=>2, 1=>1, -1=>1) ← (Rep[U₁](-1=>1, -2=>2, -3=>1) ⊗ Rep[U₁](1=>1) ⊗ Rep[U₁](1=>1)));
        Vdummy=ℂ[U1Irrep](-1=>1);
        V=ℂ[U1Irrep](0=>1,1=>1);
    elseif M==2
        U_phy1=unitary(Rep[U₁](3=>1, 4=>2, 5=>1) ← (Rep[U₁](-5=>1)' ⊗ Rep[U₁](0=>1, 1=>1)' ⊗ Rep[U₁](0=>1, 1=>1)'));
        U_phy2=unitary(Rep[U₁](0=>2, 1=>1, -1=>1) ← (Rep[U₁](3=>1, 4=>2, 5=>1) ⊗ Rep[U₁](1=>1)' ⊗ Rep[U₁](1=>1)' ⊗ Rep[U₁](1=>1)' ⊗ Rep[U₁](1=>1)'));
        Vdummy=ℂ[U1Irrep](-1=>1)';
        V=ℂ[U1Irrep](0=>1,1=>1)';
    end

    Id=[1 0;0 1];
    sm=[0 1;0 0]; sp=[0 0;1 0]; sz=[1 0; 0 -1]; occu=[0 0; 0 1];
    
    @tensor Ident[:]:=Id[-1,-3]*Id[-2,-4];
    Ident=TensorMap(Ident,  V ⊗ V ← V ⊗ V);

    @tensor NA[:]:=occu[-1,-3]*Id[-2,-4];
    NA=TensorMap(NA,  V ⊗ V ← V ⊗ V);
    
    @tensor NB[:]:=Id[-1,-3]*occu[-2,-4];
    NB=TensorMap(NB,  V ⊗ V ← V ⊗ V);

    @tensor NANB[:]:=occu[-1,-3]*occu[-2,-4];
    NANB=TensorMap(NANB,  V ⊗ V ← V ⊗ V);

    @tensor CAdag_CB[:]:=(sp*sz)[-1,-3]*sm[-2,-4];
    CAdag_CB=TensorMap(CAdag_CB, V ⊗ V ← V ⊗ V);



    @tensor cAdag[:]:=sp[-1,-3]*Id[-2,-4];
    CAdag=zeros(1,2,2,2,2);
    CAdag[1,:,:,:,:]=cAdag;
    CAdag=TensorMap(CAdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);

    @tensor cBdag[:]:=sz[-1,-3]*sp[-2,-4];
    CBdag=zeros(1,2,2,2,2);
    CBdag[1,:,:,:,:]=cBdag;
    CBdag=TensorMap(CBdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);

    @tensor cA[:]:=sm[-1,-3]*Id[-2,-4];
    CA=zeros(1,2,2,2,2);
    CA[1,:,:,:,:]=cA;
    CA=TensorMap(CA, Vdummy' ⊗ V ⊗ V ← V ⊗ V);

    @tensor cB[:]:=sz[-1,-3]*sm[-2,-4];
    CB=zeros(1,2,2,2,2);
    CB[1,:,:,:,:]=cB;
    CB=TensorMap(CB, Vdummy' ⊗ V ⊗ V ← V ⊗ V);






    ###############################

    if M==1
        @tensor Ident[:]:=Ident[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Ident[:]:=Ident[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NANB[:]:=NANB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NANB[:]:=NANB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor CAdag_CB[:]:=CAdag_CB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor CAdag_CB[:]:=CAdag_CB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
        

        @tensor CAdag[:]:=CAdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CAdag[:]:=CAdag[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CBdag[:]:=CBdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CBdag[:]:=CBdag[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CA[:]:=CA[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CA[:]:=CA[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CB[:]:=CB[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CB[:]:=CB[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

    elseif M==2
        @tensor Ident[:]:=Ident[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Ident[:]:=Ident[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

        @tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

        @tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

        @tensor NANB[:]:=NANB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NANB[:]:=NANB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];

        @tensor CAdag_CB[:]:=CAdag_CB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor CAdag_CB[:]:=CAdag_CB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
        

        @tensor CAdag[:]:=CAdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CAdag[:]:=CAdag[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];

        @tensor CBdag[:]:=CBdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CBdag[:]:=CBdag[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];

        @tensor CA[:]:=CA[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CA[:]:=CA[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];

        @tensor CB[:]:=CB[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CB[:]:=CB[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    end


    return Ident, NA, NB, NANB, CAdag_CB, CAdag, CA, CBdag, CB 
end

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

    if (Rank(O1)==2)&(Rank(O2)==2)
        @tensor A_LU[:]:= A[-1,-2,-3,-4,1]*O1[-5,1]
        @tensor A_RU[:]:= A[-1,-2,-3,-4,1]*O2[-5,1]
    elseif (Rank(O1)==3)&(Rank(O2)==3)
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
    end

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
    if (Rank(O1)==2)&(Rank(O2)==2)
        @tensor A_RU[:]:= A[-1,-2,-3,-4,1]*O1[-5,1]
        @tensor A_RD[:]:= A[-1,-2,-3,-4,1]*O2[-5,1]
    elseif (Rank(O1)==3)&(Rank(O2)==3)

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
    end

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

    if (Rank(O1)==2)&(Rank(O2)==2)
        @tensor A_LU[:]:= A[-1,-2,-3,-4,1]*O1[-5,1]
        @tensor A_RD[:]:= A[-1,-2,-3,-4,1]*O2[-5,1]
    elseif (Rank(O1)==3)&(Rank(O2)==3)
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
    end


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

    if (Rank(O1)==2)&(Rank(O2)==2)
        @tensor A_LD[:]:= A[-1,-2,-3,-4,1]*O1[-5,1]
        @tensor A_RU[:]:= A[-1,-2,-3,-4,1]*O2[-5,1]
    elseif (Rank(O1)==3)&(Rank(O2)==3)
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
    end


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



