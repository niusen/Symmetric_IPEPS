function Rank(T::TensorMap)
    return length(domain(T))+length(codomain(T))
end

function Gutzwiller_Z2(coe)

    V=Rep[ℤ₂](0=>2,1=>2);


    PG=zeros(4,4);
    PG[1,1]=coe;
    PG[2,2]=coe;
    PG[3,3]=1;
    PG[4,4]=1;
    PG=TensorMap(PG,  V ← V);
    
    return (PG,PG,)
end
function Gutzwiller_SU2(coe)


    V=Rep[SU₂](0=>2,1/2=>1);

    PG=zeros(4,4);
    PG[1,1]=coe;
    PG[2,2]=coe;
    PG[3,3]=1;
    PG[4,4]=1;
    PG=TensorMap(PG,  V ← V);

############################################


    return PG
end


function Gutzwiller_U1_SU2(coe)
    global VDummy_set
    VDummy1=VDummy_set[1];
    VDummy2=VDummy_set[2];

    V=Rep[U₁ × SU₂]((0,0)=>1,(2,0)=>1,(1, 1/2)=>1);

    PG=zeros(4,4);
    PG[1,1]=coe;
    PG[2,2]=coe;
    PG[3,3]=1;
    PG[4,4]=1;
    PG=TensorMap(PG,  V ← V);

############################################
    U_phy1=unitary(fuse(VDummy1*V), VDummy1*V);
    U_phy2=unitary(fuse(VDummy2*V), VDummy2*V);

    @tensor PG1[:]:=U_phy1[-1,1,2]*PG[2,3]*U_phy1'[1,3,-2];
    if Rank(U_phy2)==3
        @tensor PG2[:]:=U_phy2[-1,1,2]*PG[2,3]*U_phy2'[1,3,-2];
    elseif Rank(U_phy2)==2
        @tensor PG2[:]:=U_phy2[-1,2]*PG[2,3]*U_phy2'[3,-2];
    end
    
    return (PG1,PG2,)
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

    #return Ident, occu, Cdag, C, Cdag_ 
    n_double=Ident*0;#no hubbard interaction for spinless model
    return (Ident,Ident), (occu,occu), (n_double,n_double), (Cdag,Cdag), (C,C)
end


function Hamiltonians_spinless_U1()
    

    #Vdummy=ℂ[U1Irrep](-1=>1);
    #V=ℂ[U1Irrep](0=>1,1=>1);

    Vdummy=Rep[U₁](1=>1);
    V=Rep[U₁](-1/2=>1,1/2=>1);

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

    Cdag_=nothing


    return Ident, occu, Cdag, C, Cdag_ 
end



function Hamiltonians_spinful_U1_SU2()
    global VDummy_set
    VDummy1=VDummy_set[1];
    VDummy2=VDummy_set[2];

    Vdummy=Rep[U₁ × SU₂]((1, 1/2)=>1);
    V=Rep[U₁ × SU₂]((0,0)=>1,(2,0)=>1,(1, 1/2)=>1);


    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order: (0,0), (0,1), (1,0), (1,1)
    
    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[[1,4,3,2],[1,4,3,2]],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[[1,4,3,2],[1,4,3,2]],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[[1,4,3,2],[1,4,3,2]],  V ←  V);


    Cdagup=zeros(4,4,2);
    Cdagup[[1,4,3,2],[1,4,3,2],1]=kron(sp,Id);
    Cdagdn=zeros(4,4,2);
    Cdagdn[[1,4,3,2],[1,4,3,2],2]=kron(sz,sp);
    Cdag=TensorMap(Cdagup+Cdagdn,  V ← V ⊗Vdummy);
    Cdag=permute(Cdag,(3,1,),(2,))

    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=TensorMap(Cup+Cdn, Vdummy ⊗ V ← V);

############################################
    U_phy1=unitary(fuse(VDummy1*V), VDummy1*V);
    U_phy2=unitary(fuse(VDummy2*V), VDummy2*V);

    @tensor Ident1[:]:=U_phy1[-1,1,2]*Ident[2,3]*U_phy1'[1,3,-2];
    if Rank(U_phy2)==3
        @tensor Ident2[:]:=U_phy2[-1,1,2]*Ident[2,3]*U_phy2'[1,3,-2];
    elseif Rank(U_phy2)==2
        @tensor Ident2[:]:=U_phy2[-1,2]*Ident[2,3]*U_phy2'[3,-2];
    end

    @tensor N_occu1[:]:=U_phy1[-1,1,2]*N_occu[2,3]*U_phy1'[1,3,-2];
    if Rank(U_phy2)==3
        @tensor N_occu2[:]:=U_phy2[-1,1,2]*N_occu[2,3]*U_phy2'[1,3,-2];
    elseif Rank(U_phy2)==2
        @tensor N_occu2[:]:=U_phy2[-1,2]*N_occu[2,3]*U_phy2'[3,-2];
    end

    @tensor n_double1[:]:=U_phy1[-1,1,2]*n_double[2,3]*U_phy1'[1,3,-2];
    if Rank(U_phy2)==3
        @tensor n_double2[:]:=U_phy2[-1,1,2]*n_double[2,3]*U_phy2'[1,3,-2];
    elseif Rank(U_phy2)==2
        @tensor n_double2[:]:=U_phy2[-1,2]*n_double[2,3]*U_phy2'[3,-2];
    end

    @tensor Cdag1[:]:=U_phy1[-2,1,2]*Cdag[-1,2,3]*U_phy1'[1,3,-3];
    if Rank(U_phy2)==3
        @tensor Cdag2[:]:=U_phy2[-2,1,2]*Cdag[-1,2,3]*U_phy2'[1,3,-3];
    elseif Rank(U_phy2)==2
        @tensor Cdag2[:]:=U_phy2[-2,2]*Cdag[-1,2,3]*U_phy2'[3,-3];
    end

    @tensor C1[:]:=U_phy1[-2,1,2]*C[-1,2,3]*U_phy1'[1,3,-3];
    if Rank(U_phy2)==3
        @tensor C2[:]:=U_phy2[-2,1,2]*C[-1,2,3]*U_phy2'[1,3,-3];
    elseif Rank(U_phy2)==2
        @tensor C2[:]:=U_phy2[-2,2]*C[-1,2,3]*U_phy2'[3,-3];
    end
 


    
    return (Ident1,Ident2,), (N_occu1,N_occu2), (n_double1,n_double2,), (Cdag1,Cdag2,), (C1,C2,)
end



function special_Hamiltonians_spinful_U1_SU2()
    global VDummy_set
    VDummy1=VDummy_set[1];
    VDummy2=VDummy_set[2];

    Vdummy=Rep[U₁ × SU₂]((1, 1/2)=>1);
    V=Rep[U₁ × SU₂]((0,0)=>1,(2,0)=>1,(1, 1/2)=>1);


    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order: (0,0), (0,1), (1,0), (1,1)
    
    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[[1,4,3,2],[1,4,3,2]],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[[1,4,3,2],[1,4,3,2]],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[[1,4,3,2],[1,4,3,2]],  V ←  V);


    Cdagup=zeros(4,4,2);
    Cdagup[[1,4,3,2],[1,4,3,2],1]=kron(sp,sz);
    Cdagdn=zeros(4,4,2);
    Cdagdn[[1,4,3,2],[1,4,3,2],2]=kron(Id,sp);
    Cdag=TensorMap(Cdagup+Cdagdn,  V ← V ⊗Vdummy);
    Cdag=permute(Cdag,(3,1,),(2,))

    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=TensorMap(Cup+Cdn, Vdummy ⊗ V ← V);

############################################
    U_phy1=unitary(fuse(VDummy1*V), VDummy1*V);
    U_phy2=unitary(fuse(VDummy2*V), VDummy2*V);

    @tensor Ident1[:]:=U_phy1[-1,1,2]*Ident[2,3]*U_phy1'[1,3,-2];
    if Rank(U_phy2)==3
        @tensor Ident2[:]:=U_phy2[-1,1,2]*Ident[2,3]*U_phy2'[1,3,-2];
    elseif Rank(U_phy2)==2
        @tensor Ident2[:]:=U_phy2[-1,2]*Ident[2,3]*U_phy2'[3,-2];
    end

    @tensor N_occu1[:]:=U_phy1[-1,1,2]*N_occu[2,3]*U_phy1'[1,3,-2];
    if Rank(U_phy2)==3
        @tensor N_occu2[:]:=U_phy2[-1,1,2]*N_occu[2,3]*U_phy2'[1,3,-2];
    elseif Rank(U_phy2)==2
        @tensor N_occu2[:]:=U_phy2[-1,2]*N_occu[2,3]*U_phy2'[3,-2];
    end

    @tensor n_double1[:]:=U_phy1[-1,1,2]*n_double[2,3]*U_phy1'[1,3,-2];
    if Rank(U_phy2)==3
        @tensor n_double2[:]:=U_phy2[-1,1,2]*n_double[2,3]*U_phy2'[1,3,-2];
    elseif Rank(U_phy2)==2
        @tensor n_double2[:]:=U_phy2[-1,2]*n_double[2,3]*U_phy2'[3,-2];
    end

    @tensor Cdag1[:]:=U_phy1[-2,1,2]*Cdag[-1,2,3]*U_phy1'[1,3,-3];
    if Rank(U_phy2)==3
        @tensor Cdag2[:]:=U_phy2[-2,1,2]*Cdag[-1,2,3]*U_phy2'[1,3,-3];
    elseif Rank(U_phy2)==2
        @tensor Cdag2[:]:=U_phy2[-2,2]*Cdag[-1,2,3]*U_phy2'[3,-3];
    end

    @tensor C1[:]:=U_phy1[-2,1,2]*C[-1,2,3]*U_phy1'[1,3,-3];
    if Rank(U_phy2)==3
        @tensor C2[:]:=U_phy2[-2,1,2]*C[-1,2,3]*U_phy2'[1,3,-3];
    elseif Rank(U_phy2)==2
        @tensor C2[:]:=U_phy2[-2,2]*C[-1,2,3]*U_phy2'[3,-3];
    end
 


    
    return (Ident1,Ident2,), (N_occu1,N_occu2), (n_double1,n_double2,), (Cdag1,Cdag2,), (C1,C2,)
end

function Hamiltonians_spinful_SU2()
    

    Vdummy=SU2Space(1/2=>1);
    V=SU2Space(0=>2,1/2=>1);


    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order: (0,0), (0,1), (1,0), (1,1)
    
    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[[1,4,3,2],[1,4,3,2]],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[[1,4,3,2],[1,4,3,2]],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[[1,4,3,2],[1,4,3,2]],  V ←  V);

    # method 1
    Cdagup=zeros(4,4,2);
    Cdagup[[1,4,3,2],[1,4,3,2],1]=kron(sp,Id);
    Cdagdn=zeros(4,4,2);
    Cdagdn[[1,4,3,2],[1,4,3,2],2]=kron(sz,sp);
    Cdag=TensorMap(Cdagup+Cdagdn,  V ← V ⊗Vdummy);
    Cdag=permute(Cdag,(3,1,),(2,))

    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=TensorMap(Cup+Cdn, Vdummy ⊗ V ← V);
   
    return (Ident,Ident,), (N_occu,N_occu,), (n_double,n_double,), (Cdag,Cdag,), (C,C,)
end

function Operators_spinful_SU2()
    

    Vdummy=SU2Space(1/2=>1);
    V=SU2Space(0=>2,1/2=>1);


    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order: (0,0), (0,1), (1,0), (1,1)
    
    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[[1,4,3,2],[1,4,3,2]],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[[1,4,3,2],[1,4,3,2]],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[[1,4,3,2],[1,4,3,2]],  V ←  V);


    n_hole=kron(Id-occu,Id-occu);
    n_hole=TensorMap(n_hole[[1,4,3,2],[1,4,3,2]],  V ←  V);

    # method 1
    Cdagup=zeros(4,4,2);
    Cdagup[[1,4,3,2],[1,4,3,2],1]=kron(sp,Id);
    Cdagdn=zeros(4,4,2);
    Cdagdn[[1,4,3,2],[1,4,3,2],2]=kron(sz,sp);
    Cdag=TensorMap(Cdagup+Cdagdn,  V ← V ⊗Vdummy);
    Cdag=permute(Cdag,(3,1,),(2,))

    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=TensorMap(Cup+Cdn, Vdummy ⊗ V ← V);


    CdagupCdagdn=zeros(4,4);
    CdagupCdagdn[[1,4,3,2],[1,4,3,2]]=kron(sp,sp);
    CdagupCdagdn=TensorMap(CdagupCdagdn,  V ← V );

    Vdummy0=SU2Space(0=>1);
    Pairinga=zeros(4,4,1);#CdagupCdagdn
    Pairinga[[1,4,3,2],[1,4,3,2],1]=kron(sp,sp);
    Pairinga=TensorMap(Pairinga,  V ← V ⊗Vdummy0);
    Pairinga=permute(Pairinga,(3,1,),(2,))

    Pairingb=zeros(1,4,4);#CupCdn
    Pairingb[1,[1,4,3,2],[1,4,3,2]]=kron(sm,sm);
    Pairingb=TensorMap(Pairingb, Vdummy0 ⊗ V ← V);

    #Spin operator
    Sp=zeros(4,4);
    Sp[[1,4,3,2],[1,4,3,2]]=kron(sp,sm);
    Sm=zeros(4,4);
    Sm[[1,4,3,2],[1,4,3,2]]=kron(sm,sp);
    Sz=zeros(4,4);
    Sz[[1,4,3,2],[1,4,3,2]]=kron(occu,Id)/2-kron(Id,occu)/2;
    @tensor SpSm[:]:=Sp[-1,-2]*Sm[-3,-4];
    @tensor SmSp[:]:=Sm[-1,-2]*Sp[-3,-4];
    @tensor SzSz[:]:=Sz[-1,-2]*Sz[-3,-4];
    SS=SpSm/2+SmSp/2+SzSz;
    SS=permutedims(SS,(1,3,2,4));#s1's2's1s2
    SS=TensorMap(SS, V ⊗ V ← V ⊗ V);
    SS=permute(SS,(1,3,),(2,4,));#V,s1',s1,s2',s2
    u0,s0,v0=tsvd(SS; trunc=truncerr(1e-12));
    @assert norm(u0*s0*v0-SS)<1e-14;
    Sa=permute(u0*s0,(3,1,),(2,));
    Sb=permute(v0,(1,2,),(3,));


    Sx=(Sp+Sm)/2;
    Sy=(Sp-Sm)/(2*im);
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

    return (Ident,Ident,), (N_occu,N_occu,), (n_hole,n_hole), (n_double,n_double,), (Cdag,Cdag,), (C,C,), (CdagupCdagdn,CdagupCdagdn), (Pairinga,Pairinga), (Pairingb,Pairingb), (Sa,Sa), (Sb,Sb), chirality_S1,chirality_S2,chirality_S3
end

function Operators_spinful_Z2()
    
    Vdummy=Rep[ℤ₂](1=>2);
    V=Rep[ℤ₂](0=>2,1=>2);



    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order: (0,0), (0,1), (1,0), (1,1)
    
    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[[1,4,3,2],[1,4,3,2]],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[[1,4,3,2],[1,4,3,2]],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[[1,4,3,2],[1,4,3,2]],  V ←  V);


    n_hole=kron(Id-occu,Id-occu);
    n_hole=TensorMap(n_hole[[1,4,3,2],[1,4,3,2]],  V ←  V);

    # method 1
    Cdagup=zeros(4,4,2);
    Cdagup[[1,4,3,2],[1,4,3,2],1]=kron(sp,Id);
    Cdagdn=zeros(4,4,2);
    Cdagdn[[1,4,3,2],[1,4,3,2],2]=kron(sz,sp);
    Cdag=TensorMap(Cdagup+Cdagdn,  V ← V ⊗Vdummy);
    Cdag=permute(Cdag,(3,1,),(2,))

    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=TensorMap(Cup+Cdn, Vdummy ⊗ V ← V);


    CdagupCdagdn=zeros(4,4);
    CdagupCdagdn[[1,4,3,2],[1,4,3,2]]=kron(sp,sp);
    CdagupCdagdn=TensorMap(CdagupCdagdn,  V ← V );

    Vdummy0=Rep[ℤ₂](0=>1);
    Pairinga=zeros(4,4,1);#CdagupCdagdn
    Pairinga[[1,4,3,2],[1,4,3,2],1]=kron(sp,sp);
    Pairinga=TensorMap(Pairinga,  V ← V ⊗Vdummy0);
    Pairinga=permute(Pairinga,(3,1,),(2,))

    Pairingb=zeros(1,4,4);#CupCdn
    Pairingb[1,[1,4,3,2],[1,4,3,2]]=kron(sm,sm);
    Pairingb=TensorMap(Pairingb, Vdummy0 ⊗ V ← V);

    #Spin operator
    Sp=zeros(4,4);
    Sp[[1,4,3,2],[1,4,3,2]]=kron(sp,sm);
    Sm=zeros(4,4);
    Sm[[1,4,3,2],[1,4,3,2]]=kron(sm,sp);
    Sz=zeros(4,4);
    Sz[[1,4,3,2],[1,4,3,2]]=kron(occu,Id)/2-kron(Id,occu)/2;
    @tensor SpSm[:]:=Sp[-1,-2]*Sm[-3,-4];
    @tensor SmSp[:]:=Sm[-1,-2]*Sp[-3,-4];
    @tensor SzSz[:]:=Sz[-1,-2]*Sz[-3,-4];
    SS=SpSm/2+SmSp/2+SzSz;
    SS=permutedims(SS,(1,3,2,4));#s1's2's1s2
    SS=TensorMap(SS, V ⊗ V ← V ⊗ V);
    SS=permute(SS,(1,3,),(2,4,));#V,s1',s1,s2',s2
    u0,s0,v0=tsvd(SS; trunc=truncerr(1e-12));
    @assert norm(u0*s0*v0-SS)<1e-14;
    
    Sa=permute(u0*s0,(3,1,),(2,));
    Sb=permute(v0,(1,2,),(3,));

    Sx=(Sp+Sm)/2;
    Sy=(Sp-Sm)/(2*im);
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

    return (Ident,Ident,), (N_occu,N_occu,), (n_hole,n_hole), (n_double,n_double,), (Cdag,Cdag,), (C,C,), (CdagupCdagdn,CdagupCdagdn), (Pairinga,Pairinga), (Pairingb,Pairingb), (Sa,Sa), (Sb,Sb), chirality_S1,chirality_S2,chirality_S3
end


function special_Hamiltonians_spinful_SU2()
    

    Vdummy=SU2Space(1/2=>1);
    V=SU2Space(0=>2,1/2=>1);


    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order: (0,0), (0,1), (1,0), (1,1)
    
    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[[1,4,3,2],[1,4,3,2]],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[[1,4,3,2],[1,4,3,2]],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[[1,4,3,2],[1,4,3,2]],  V ←  V);

    # method 1
    Cdagup=zeros(4,4,2);
    Cdagup[[1,4,3,2],[1,4,3,2],1]=kron(sp,sz);
    Cdagdn=zeros(4,4,2);
    Cdagdn[[1,4,3,2],[1,4,3,2],2]=kron(Id,sp);
    Cdag=TensorMap(Cdagup+Cdagdn,  V ← V ⊗Vdummy);
    Cdag=permute(Cdag,(3,1,),(2,))

    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=TensorMap(Cup+Cdn, Vdummy ⊗ V ← V);


    return (Ident,Ident,), (N_occu,N_occu,), (n_double,n_double,), (Cdag,Cdag,), (C,C,)
end

function Hamiltonians_spinful_Z2()
    

    # Vdummy=Rep[ℤ₂](1=>1);
    Vdummy=Rep[ℤ₂](1=>2);
    V=Rep[ℤ₂](0=>2,1=>2);


    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order of kron() command: (0,0), (0,1), (1,0), (1,1)
    order=[1,4,3,2];

    # Ident=kron(Id,Id);
    # Ident=TensorMap(Ident[order,order],  V  ←  V);

    # N_occu=kron(occu,Id)+kron(Id,occu);
    # N_occu=TensorMap(N_occu[order,order],  V ←  V);
    # n_double=kron(occu,occu)
    # n_double=TensorMap(n_double[order,order],  V ←  V);

    # # method 1
    # Cdagup=zeros(4,4,1);
    # Cdagup[order,order,1]=kron(sp,Id);
    # Cdagdn=zeros(4,4,1);
    # Cdagdn[order,order,1]=kron(sz,sp);
    # Cdagup=TensorMap(Cdagup,  V ← V ⊗Vdummy);Cdagup=permute(Cdagup,(3,1,),(2,))
    # Cdagdn=TensorMap(Cdagdn,  V ← V ⊗Vdummy);Cdagdn=permute(Cdagdn,(3,1,),(2,))

    # Cup=zeros(1,4,4);
    # Cup[1,order,order]=kron(sm,Id);
    # Cdn=zeros(1,4,4);
    # Cdn[1,order,order]=kron(sz,sm);
    # Cup=TensorMap(Cup, Vdummy ⊗ V ← V);
    # Cdn=TensorMap(Cdn, Vdummy ⊗ V ← V);

    # return Ident, N_occu, n_double, Cdagup, Cup, Cdagdn, Cdn

    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[[1,4,3,2],[1,4,3,2]],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[[1,4,3,2],[1,4,3,2]],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[[1,4,3,2],[1,4,3,2]],  V ←  V);

    # method 1
    Cdagup=zeros(4,4,2);
    Cdagup[[1,4,3,2],[1,4,3,2],1]=kron(sp,Id);
    Cdagdn=zeros(4,4,2);
    Cdagdn[[1,4,3,2],[1,4,3,2],2]=kron(sz,sp);
    Cdag=TensorMap(Cdagup+Cdagdn,  V ← V ⊗Vdummy);
    Cdag=permute(Cdag,(3,1,),(2,))

    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=TensorMap(Cup+Cdn, Vdummy ⊗ V ← V);
   
    return (Ident,Ident,), (N_occu,N_occu,), (n_double,n_double,), (Cdag,Cdag,), (C,C,)
end

function spin_operator_Z2()
    
    V=Rep[ℤ₂](0=>2,1=>2);


    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order of kron() command: (0,0), (0,1), (1,0), (1,1)
    order=[1,4,3,2];



    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[[1,4,3,2],[1,4,3,2]],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[[1,4,3,2],[1,4,3,2]],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[[1,4,3,2],[1,4,3,2]],  V ←  V);

    
    Cdagup_Cup=zeros(4,4);
    Cdagup_Cup[[1,4,3,2],[1,4,3,2]]=kron(sp*sm,Id);
    Cdagup_Cup=TensorMap(Cdagup_Cup,  V ← V);

    Cdagdn_Cdn=zeros(4,4);
    Cdagdn_Cdn[[1,4,3,2],[1,4,3,2]]=kron(Id,sp*sm);
    Cdagdn_Cdn=TensorMap(Cdagdn_Cdn,  V ← V);

    Cdagup_Cdn=zeros(4,4);
    Cdagup_Cdn[[1,4,3,2],[1,4,3,2]]=kron(sp,sm);
    Cdagup_Cdn=TensorMap(Cdagup_Cdn,  V ← V);

    Cdagdn_Cup=zeros(4,4);
    Cdagdn_Cup[[1,4,3,2],[1,4,3,2]]=kron(sm,sp);
    Cdagdn_Cup=TensorMap(Cdagdn_Cup,  V ← V);


    sx=Cdagup_Cdn+Cdagdn_Cup;
    sy=-im*Cdagup_Cdn+im*Cdagdn_Cup;
    sz=Cdagup_Cup-Cdagdn_Cdn;


    return sx,sy,sz
end

function special_Hamiltonians_spinful_Z2()
    

    # Vdummy=Rep[ℤ₂](1=>1);
    Vdummy=Rep[ℤ₂](1=>2);
    V=Rep[ℤ₂](0=>2,1=>2);


    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order of kron() command: (0,0), (0,1), (1,0), (1,1)
    order=[1,4,3,2];

    # Ident=kron(Id,Id);
    # Ident=TensorMap(Ident[order,order],  V  ←  V);

    # N_occu=kron(occu,Id)+kron(Id,occu);
    # N_occu=TensorMap(N_occu[order,order],  V ←  V);
    # n_double=kron(occu,occu)
    # n_double=TensorMap(n_double[order,order],  V ←  V);

    # # method 1
    # Cdagup=zeros(4,4,1);
    # Cdagup[order,order,1]=kron(sp,sz);
    # Cdagdn=zeros(4,4,1);
    # Cdagdn[order,order,1]=kron(Id,sp);
    # Cdagup=TensorMap(Cdagup,  V ← V ⊗Vdummy);Cdagup=permute(Cdagup,(3,1,),(2,))
    # Cdagdn=TensorMap(Cdagdn,  V ← V ⊗Vdummy);Cdagdn=permute(Cdagdn,(3,1,),(2,))

    # Cup=zeros(1,4,4);
    # Cup[1,order,order]=kron(sm,Id);
    # Cdn=zeros(1,4,4);
    # Cdn[1,order,order]=kron(sz,sm);
    # Cup=TensorMap(Cup, Vdummy ⊗ V ← V);
    # Cdn=TensorMap(Cdn, Vdummy ⊗ V ← V);



    # return Ident, N_occu, n_double, Cdagup, Cup, Cdagdn, Cdn


    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[[1,4,3,2],[1,4,3,2]],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[[1,4,3,2],[1,4,3,2]],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[[1,4,3,2],[1,4,3,2]],  V ←  V);

    # method 1
    Cdagup=zeros(4,4,2);
    Cdagup[[1,4,3,2],[1,4,3,2],1]=kron(sp,sz);
    Cdagdn=zeros(4,4,2);
    Cdagdn[[1,4,3,2],[1,4,3,2],2]=kron(Id,sp);
    Cdag=TensorMap(Cdagup+Cdagdn,  V ← V ⊗Vdummy);
    Cdag=permute(Cdag,(3,1,),(2,))

    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=TensorMap(Cup+Cdn, Vdummy ⊗ V ← V);


    return (Ident,Ident,), (N_occu,N_occu,), (n_double,n_double,), (Cdag,Cdag,), (C,C,)
end


function B_field_fun(Lx,Ly,coord)
    #120 degree magnetic order
    @assert mod(Lx,3)==0;
    @assert mod(Ly,3)==0;
    #coordnate:
    # (1,1)(2,1)(3,2)
    # (1,2)(2,2)(3,2)
    # (1,3)(2,3)(3,3)

    #A triangle:
    #      (2,0)
    # (1,1)(2,1)

    px=coord[1];
    py=coord[2];
    #reshape square lattice to triangle lattice
    #step1: 
    px1= px+py/2;
    #step2: 
    py1->py*sqrt(3)/2;



    return [sin(-2*pi/3*(px-py)) cos(-2*pi/3*(px-py)) 0]
end



function ob_2x2(CTM,AA_LU_,AA_RU_,AA_LD_,AA_RD_,cx,cy)
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




function hopping_x(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
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


    if ctm_setting.grad_checkpoint
        AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    else
        AA_LU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    end


    

    ob=ob_2x2(CTM,AA_LU_double,AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
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
        AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    else
        AA_LU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    end


    

    ob=ob_2x2(CTM,AA_LU_double,AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
end

function hopping_y(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
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


    if ctm_setting.grad_checkpoint
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    else
        AA_RD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    end


    
    ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_RD_double,cx,cy);
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
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    else
        AA_RD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    end


    
    ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_RD_double,cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
end



function ob_onsite(CTM,O1,A_cell,AA_cell,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];


    @tensor A1[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-5,1]

    if ctm_setting.grad_checkpoint
        A1_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LU[1]][pos_LU[2]]',A1);
    else
        A1_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LU[1]][pos_LU[2]]',A1);
    end

    ob=ob_2x2(CTM,A1_double, AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
end

function hopping_diagonalb(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    # @tensor A_LU[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-5,1]
    # @tensor A_RD[:]:= A_cell[pos_RD[1]][pos_RD[2]][-1,-2,-3,-4,1]*O2[-5,1]
    @tensor A_LU[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
    @tensor A_RD[:]:= A_cell[pos_RD[1]][pos_RD[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
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
    @tensor A_RU[:]:=A_cell[pos_RU[1]][pos_RU[2]][1,3,-3,-4,-5]*O_string[4,2]*U1'[1,2,-1]*U2'[3,4,-2];
    @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,2]*U2[-4,1,2];



    if ctm_setting.grad_checkpoint
        AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
    else
        AA_LU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_RD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
    end


    ob=ob_2x2(CTM,AA_LU_double,AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_RD_double,cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob    
end



function hopping_diagonala(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
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

    ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_LD_double,AA_RD_double,cx,cy);
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
        AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
    else
        AA_LD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_RD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
    end

    ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_LD_double,AA_RD_double,cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob        
end


function hopping_diagonala_split(CTM,O1,O2,A_cell,AA_cell,cx,cy,ctm_setting)
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

    p1a,p1b=@ignore_derivatives projector_parity(space(A_RD,1));
    p4a,p4b=@ignore_derivatives projector_parity(space(A_RD,4));
    Pset1=(p1a,p1b,);
    Pset4=(p4a,p4b,);

    ob_total=0;
    for cp1=1:2
        for cp2=1:2
            @tensor A_LD_tem[:]:=A_LD[-1,-2,3,-4,-5]*Pset1[cp1]'[3,-3];
            @tensor A_RD_tem[:]:=A_RD[1,-2,-3,4,-5]*Pset1[cp1][-1,1]*Pset4[cp2][-4,4];
            @tensor A_RU_tem[:]:=A_RU[-1,2,-3,-4,-5]*Pset4[cp2]'[2,-2];
            if ctm_setting.grad_checkpoint
                AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LD[1]][pos_LD[2]]',A_LD_tem);
                AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU_tem);
                AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1]][pos_RD[2]]',A_RD_tem);
            else
                AA_LD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LD[1]][pos_LD[2]]',A_LD_tem);
                AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU_tem);
                AA_RD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD_tem);
            end

            ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_LD_double,AA_RD_double,cx,cy);
            Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
            ob_total=ob_total+ob/Norm;
        end
    end
    return ob_total        
end




function evaluate_spin_cell(A_cell::Tuple, AA_cell, CTM_cell, ctm_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    
    global Lx,Ly
    if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
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

function evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    

    global Lx,Ly

    if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        if energy_setting.model in  ("Triangle_Hofstadter_Hubbard", "spinful_triangle_lattice", "standard_triangle_Hubbard","standard_triangle_Hubbard_spiral","standard_triangle_Hubbard_Bfield")
            Hamiltonian_terms=Hamiltonians_spinful_Z2;
            operator_terms=Operators_spinful_Z2;
        elseif (energy_setting.model == "Triangle_Hofstadter_spinless")
            Hamiltonian_terms=Hamiltonians_spinless_Z2;
        end
    elseif isa(space(A_cell[1][1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(A_cell[1][1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
        operator_terms=Operators_spinful_SU2;
    elseif isa(space(A_cell[1][1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        if mod(energy_setting.Magnetic_cell,2)==1 #odd number of sites in unitcell
            @assert mod(Ly,2)==0;
            #if use U1 symmetry, use different dummy physical space along y direction along Ly, where Ly should be even number
        end
        Hamiltonian_terms=Hamiltonians_spinful_U1_SU2;
    end


    if energy_setting.model=="spinless_Hubbard";
        Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonian_terms();
        t1=parameters["t1"];
        γ=parameters["γ"];
        μ=parameters["μ"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        px_set=zeros(Lx,Ly)*im;
        py_set=zeros(Lx,Ly)*im;
        e0_set=zeros(Lx,Ly)*im;

        E_total=0;
        for cx=1:Lx
            for cy=1:Ly
                
                ex=hopping_x(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
                ey=hopping_y(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
                e0=ob_onsite(CTM_cell,occu,A_cell,AA_cell,cx,cy,ctm_setting);
        
                @ignore_derivatives ex_set[cx,cy]=ex;
                @ignore_derivatives ey_set[cx,cy]=ey;
                @ignore_derivatives e0_set[cx,cy]=e0;
                
                E_total=E_total+real(t1*ex+t1'*ex'+t1*ey+t1'*ey' -2*μ*e0);
                
            end
        end
        E_total=E_total/(Lx*Ly);

        # println(E_LU_RU_LD_set)
        # println(E_LD_RU_RD_set)
        # println(E_LU_LD_RD_set)
        # println(E_LU_RU_RD_set)
        return E_total,  ex_set, ey_set, e0_set
    elseif energy_setting.model=="spinless_Hubbard_pairing";
        Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonian_terms();
        t1=parameters["t1"];
        γ=parameters["γ"];
        μ=parameters["μ"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        px_set=zeros(Lx,Ly)*im;
        py_set=zeros(Lx,Ly)*im;
        e0_set=zeros(Lx,Ly)*im;

        E_total=0;
        for cx=1:Lx
            for cy=1:Ly
                
                ex=hopping_x(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
                ey=hopping_y(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
                px=hopping_x(CTM_cell,Cdag,Cdag_,A_cell,AA_cell,cx,cy,ctm_setting);
                py=hopping_y(CTM_cell,Cdag,Cdag_,A_cell,AA_cell,cx,cy,ctm_setting);
                e0=ob_onsite(CTM_cell,occu,A_cell,AA_cell,cx,cy,ctm_setting);
        
                @ignore_derivatives ex_set[cx,cy]=ex;
                @ignore_derivatives ey_set[cx,cy]=ey;
                @ignore_derivatives px_set[cx,cy]=px;
                @ignore_derivatives py_set[cx,cy]=py;
                @ignore_derivatives e0_set[cx,cy]=e0;
                
                E_total=E_total+real(t1*ex+t1'*ex'+t1*ey+t1'*ey'+γ*px+γ'*px'+γ*py+γ'*py' -2*μ*e0);
                
            end
        end
        E_total=E_total/(Lx*Ly);

        # println(E_LU_RU_LD_set)
        # println(E_LD_RU_RD_set)
        # println(E_LU_LD_RD_set)
        # println(E_LU_RU_RD_set)
        return E_total,  ex_set, ey_set, px_set, py_set, e0_set
    elseif energy_setting.model=="spinless_t1_t2";
        Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonian_terms();
        t1=parameters["t1"];
        t2=parameters["t2"];
        μ=parameters["μ"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonalb_set=zeros(Lx,Ly)*im;
        e_diagonala_set=zeros(Lx,Ly)*im;
        e0_set=zeros(Lx,Ly)*im;

        E_total=0;
        for cx=1:Lx
            for cy=1:Ly
                
                ex=hopping_x(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
                ey=hopping_y(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
                e_diagonalb=hopping_diagonalb(CTM_cell,Cdag,-1*C,A_cell,AA_cell,cx,cy,ctm_setting);#compared with exact result, here a minus sign to ensure correct result
                e_diagonala=hopping_diagonala(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
                e0=ob_onsite(CTM_cell,occu,A_cell,AA_cell,cx,cy,ctm_setting);
        
                @ignore_derivatives ex_set[cx,cy]=ex;
                @ignore_derivatives ey_set[cx,cy]=ey;
                @ignore_derivatives e_diagonalb_set[cx,cy]=e_diagonalb;
                @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
                @ignore_derivatives e0_set[cx,cy]=e0;
                
                E_total=E_total+real(t1*ex+t1'*ex'+t1*ey+t1'*ey'+t2*e_diagonalb+t2'*e_diagonalb'+t2*e_diagonala+t2'*e_diagonala' -2*μ*e0);
                
            end
        end
        E_total=E_total/(Lx*Ly);

        # println(E_LU_RU_LD_set)
        # println(E_LD_RU_RD_set)
        # println(E_LU_LD_RD_set)
        # println(E_LU_RU_RD_set)
        return E_total,  ex_set, ey_set, e_diagonalb_set, e_diagonala_set, e0_set
    elseif energy_setting.model=="spinless_triangle_lattice"
        if (Lx==2) & (Ly==1) #2x1 cell
            Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonian_terms();
            t1=parameters["t1"];
            t2=parameters["t2"];
            ϕ=parameters["ϕ"];
            μ=parameters["μ"];

            ex_set=zeros(Lx,Ly)*im;
            ey_set=zeros(Lx,Ly)*im;
            e_diagonala_set=zeros(Lx,Ly)*im;
            e0_set=zeros(Lx,Ly)*im;

            
            E_total=0;

            cx=1;cy=1;
            ex=hopping_x(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
            ey=hopping_y(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
            e_diagonala=hopping_diagonala(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
            e0=ob_onsite(CTM_cell,occu,A_cell,AA_cell,cx,cy,ctm_setting);
            @ignore_derivatives ex_set[cx,cy]=ex;
            @ignore_derivatives ey_set[cx,cy]=ey;
            @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
            @ignore_derivatives e0_set[cx,cy]=e0;
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala') -μ*e0);

            cx=2;cy=1;
            ex=hopping_x(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
            ey=hopping_y(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
            e_diagonala=hopping_diagonala(CTM_cell,Cdag,C,A_cell,AA_cell,cx,cy,ctm_setting);
            e0=ob_onsite(CTM_cell,occu,A_cell,AA_cell,cx,cy,ctm_setting);
            @ignore_derivatives ex_set[cx,cy]=ex;
            @ignore_derivatives ey_set[cx,cy]=ey;
            @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
            @ignore_derivatives e0_set[cx,cy]=e0;
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')+t1*(ey+ey')+t2*(e_diagonala+e_diagonala') -μ*e0);
            
                
            
            E_total=E_total/(Lx*Ly);
            return E_total,  ex_set, ey_set, e_diagonala_set, e0_set
            
        end
    elseif energy_setting.model=="spinful_triangle_lattice"
        if (Lx==2) & (Ly==1) #2x1 cell
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

            cx=1;cy=1;
            ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e0=ob_onsite(CTM_cell,N_occu_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            eU=ob_onsite(CTM_cell,n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            
            @ignore_derivatives ex_set[cx,cy]=ex;
            @ignore_derivatives ey_set[cx,cy]=ey;
            @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
            @ignore_derivatives e0_set[cx,cy]=e0;
            @ignore_derivatives eU_set[cx,cy]=eU;
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala')  +U*eU -μ*e0);

            cx=2;cy=1;
            ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e0=ob_onsite(CTM_cell,N_occu_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            eU=ob_onsite(CTM_cell,n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);

            @ignore_derivatives ex_set[cx,cy]=ex;
            @ignore_derivatives ey_set[cx,cy]=ey;
            @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
            @ignore_derivatives e0_set[cx,cy]=e0;
            @ignore_derivatives eU_set[cx,cy]=eU;
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')+t1*(ey+ey')+t2*(e_diagonala+e_diagonala')  +U*eU -μ*e0);
            
                
            
            E_total=E_total/(Lx*Ly);
            return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set
        elseif (Lx==2) & (Ly==2) 
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

            cx=1;cy=1;
            ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e0=ob_onsite(CTM_cell,N_occu_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            eU=ob_onsite(CTM_cell,n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            @ignore_derivatives ex_set[cx,cy]=ex;
            @ignore_derivatives ey_set[cx,cy]=ey;
            @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
            @ignore_derivatives e0_set[cx,cy]=e0;
            @ignore_derivatives eU_set[cx,cy]=eU;
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala')  +U*eU -μ*e0);

            cx=1;cy=2;
            ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e0=ob_onsite(CTM_cell,N_occu_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            eU=ob_onsite(CTM_cell,n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            @ignore_derivatives ex_set[cx,cy]=ex;
            @ignore_derivatives ey_set[cx,cy]=ey;
            @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
            @ignore_derivatives e0_set[cx,cy]=e0;
            @ignore_derivatives eU_set[cx,cy]=eU;
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala')  +U*eU -μ*e0);

            cx=2;cy=1;
            ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e0=ob_onsite(CTM_cell,N_occu_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            eU=ob_onsite(CTM_cell,n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            @ignore_derivatives ex_set[cx,cy]=ex;
            @ignore_derivatives ey_set[cx,cy]=ey;
            @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
            @ignore_derivatives e0_set[cx,cy]=e0;
            @ignore_derivatives eU_set[cx,cy]=eU;
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')+t1*(ey+ey')+t2*(e_diagonala+e_diagonala')  +U*eU -μ*e0);

            cx=2;cy=2;
            ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(cx+1,Lx)],C_set[mod1(cx+2,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            e0=ob_onsite(CTM_cell,N_occu_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            eU=ob_onsite(CTM_cell,n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)],A_cell,AA_cell,cx,cy,ctm_setting);
            @ignore_derivatives ex_set[cx,cy]=ex;
            @ignore_derivatives ey_set[cx,cy]=ey;
            @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
            @ignore_derivatives e0_set[cx,cy]=e0;
            @ignore_derivatives eU_set[cx,cy]=eU;
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')+t1*(ey+ey')+t2*(e_diagonala+e_diagonala')  +U*eU -μ*e0);
             
            
            E_total=E_total/(Lx*Ly);
            return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set
        else#if (Lx==6) & (Ly==3)
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
                    
                    ex=hopping_x(CTM_cell,Cdag_set[mod1(cx+1,2)],C_set[mod1(cx+2,2)],A_cell,AA_cell,cx,cy,ctm_setting);
                    ey=hopping_y(CTM_cell,Cdag_set[mod1(cx+2,2)],C_set[mod1(cx+2,2)],A_cell,AA_cell,cx,cy,ctm_setting);
                    e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(cx+1,2)],C_set[mod1(cx+2,2)],A_cell,AA_cell,cx,cy,ctm_setting);
                    e0=ob_onsite(CTM_cell,N_occu_set[mod1(cx+1,2)],A_cell,AA_cell,cx,cy,ctm_setting);
                    eU=ob_onsite(CTM_cell,n_double_set[mod1(cx+1,2)]-(1/2)*N_occu_set[mod1(cx+1,2)]+(1/4)*Ident_set[mod1(cx+1,2)],A_cell,AA_cell,cx,cy,ctm_setting);
                    @ignore_derivatives ex_set[cx,cy]=ex;
                    @ignore_derivatives ey_set[cx,cy]=ey;
                    @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
                    @ignore_derivatives e0_set[cx,cy]=e0;
                    @ignore_derivatives eU_set[cx,cy]=eU;
                    if mod(cx,2)==1
                        E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala')  +U*eU -μ*e0);
                    else
                        E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')+t1*(ey+ey')+t2*(e_diagonala+e_diagonala')  +U*eU -μ*e0);
                    end
                end
            end

            E_total=E_total/(Lx*Ly);
            return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set
        end

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
                ex=hopping_x(CTM_cell,Cdag_set[mod1(py,Ly)],C_set[mod1(py,Ly)],A_cell,AA_cell,cx,cy,ctm_setting);
                ey=hopping_y(CTM_cell,Cdag_set[mod1(py,Ly)],C_set[mod1(py+1,Ly)],A_cell,AA_cell,cx,cy,ctm_setting);
                e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(py+1,Ly)],C_set[mod1(py,Ly)],A_cell,AA_cell,cx,cy,ctm_setting);
                e0=ob_onsite(CTM_cell,N_occu_set[mod1(py,Ly)],A_cell,AA_cell,cx,cy,ctm_setting);
                eU=ob_onsite(CTM_cell,n_double_set[mod1(py,Ly)]-(1/2)*N_occu_set[mod1(py,Ly)]+(1/4)*Ident_set[mod1(py,Ly)],A_cell,AA_cell,cx,cy,ctm_setting);
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
        (Ident,Ident,), (N_occu,N_occu,), (n_hole,n_hole), (n_double,n_double,), (Cdag,Cdag,), (C,C,), (CdagupCdagdn,CdagupCdagdn), (Pairinga,Pairinga), (Pairingb,Pairingb), (Sa,Sa), (Sb,Sb), chirality_S1,chirality_S2,chirality_S3 =@ignore_derivatives  operator_terms();
        
        t1=parameters["t1"];
        t2=parameters["t2"];
        μ=parameters["μ"];
        U=parameters["U"];
        Chi_up_triangle=parameters["Chi_up_triangle"];
        Chi_dn_triangle=parameters["Chi_dn_triangle"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonala_set=zeros(Lx,Ly)*im;
        e0_set=zeros(Lx,Ly)*im;
        eU_set=zeros(Lx,Ly)*im;
        triangle_up_set=zeros(Lx,Ly)*im;
        triangle_dn_set=zeros(Lx,Ly)*im;
        
        E_total=0;
        for px=1:Lx
            for py=1:Ly
                #(cx,cy): coordinate of left-top C1 tensor
                cx=mod1(px-1,Lx);
                cy=mod1(py-1,Ly);
                ex=hopping_x(CTM_cell,Cdag_set[mod1(py,2)],C_set[mod1(py,2)],A_cell,AA_cell,cx,cy,ctm_setting);
                ey=hopping_y(CTM_cell,Cdag_set[mod1(py,2)],C_set[mod1(py+1,2)],A_cell,AA_cell,cx,cy,ctm_setting);
                e_diagonala=hopping_diagonala(CTM_cell,Cdag_set[mod1(py+1,2)],C_set[mod1(py,2)],A_cell,AA_cell,cx,cy,ctm_setting);
                e0=ob_onsite(CTM_cell,N_occu_set[mod1(py,2)],A_cell,AA_cell,cx,cy,ctm_setting);
                eU=ob_onsite(CTM_cell,n_double_set[mod1(py,2)]-(1/2)*N_occu_set[mod1(py,2)]+(1/4)*Ident_set[mod1(py,2)],A_cell,AA_cell,cx,cy,ctm_setting);

                up_triangle=ob_up_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,RD,RU
                dn_triangle=-ob_dn_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,LU,RU


                @ignore_derivatives ex_set[px,py]=ex;
                @ignore_derivatives ey_set[px,py]=ey;
                @ignore_derivatives e_diagonala_set[px,py]=e_diagonala;
                @ignore_derivatives e0_set[px,py]=e0;
                @ignore_derivatives eU_set[px,py]=eU;
                @ignore_derivatives triangle_up_set[px,py]=up_triangle;
                @ignore_derivatives triangle_dn_set[px,py]=dn_triangle;

                E_temp=-t1*ex -t1*ey -t2*e_diagonala -μ*e0/2  +U*eU/2;
                #E_temp=-t1*ex -t1*ey -t2*e_diagonala  +U*eU/2; # do not include chemical potential
                E_total=E_total+real(E_temp+E_temp');

                if abs(Chi_up_triangle)>0
                    E_total=E_total+Chi_up_triangle*real(up_triangle);
                end
                if abs(Chi_dn_triangle)>0
                    E_total=E_total+Chi_dn_triangle*real(dn_triangle);
                end
                
            end
        end 
        E_total=E_total/(Lx*Ly);
        return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set, triangle_up_set, triangle_dn_set

    elseif energy_setting.model =="standard_triangle_Hubbard_spiral" 
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
        (Ident,Ident,), (N_occu,N_occu,), (n_hole,n_hole), (n_double,n_double,), (Cdag,Cdag,), (C,C,), (CdagupCdagdn,CdagupCdagdn), (Pairinga,Pairinga), (Pairingb,Pairingb), (Sa,Sa), (Sb,Sb), chirality_S1,chirality_S2,chirality_S3 =@ignore_derivatives  operator_terms();
        @assert Lx==1;
        @assert Ly==1;

        t1=parameters["t1"];
        t2=parameters["t2"];
        μ=parameters["μ"];
        U=parameters["U"];
        J=parameters["J"];
        # Chi_up_triangle=parameters["Chi_up_triangle"];
        # Chi_dn_triangle=parameters["Chi_dn_triangle"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonala_set=zeros(Lx,Ly)*im;
        e0_set=zeros(Lx,Ly)*im;
        eU_set=zeros(Lx,Ly)*im;
        SSx_set=zeros(Lx,Ly)*im;
        SSy_set=zeros(Lx,Ly)*im;
        SSdiagonal_set=zeros(Lx,Ly)*im;
        # triangle_up_set=zeros(Lx,Ly)*im;
        # triangle_dn_set=zeros(Lx,Ly)*im;
        
        E_total=0;
        px=1;
        py=1;

        #(cx,cy): coordinate of left-top C1 tensor
        cx=mod1(px-1,Lx);
        cy=mod1(py-1,Ly);
        cdag_origin=Cdag_set[1];
        c_origin=C_set[1];

        coord_LU=[1,1];
        coord_LD=[1,2];
        coord_RD=[2,2];
        coord_RU=[2,1];

        sx,sy,sz=@ignore_derivatives spin_operator_Z2();
        sx=sx/2;
        sy=sy/2;
        sz=sz/2;

        op_LU=@ignore_derivatives exp(-im*2*pi/3*(coord_LU[1]-coord_LU[2])*sz);
        op_LD=@ignore_derivatives exp(-im*2*pi/3*(coord_LD[1]-coord_LD[2])*sz);
        op_RD=@ignore_derivatives exp(-im*2*pi/3*(coord_RD[1]-coord_RD[2])*sz);
        op_RU=@ignore_derivatives exp(-im*2*pi/3*(coord_RU[1]-coord_RU[2])*sz);

        @tensor Cdag_LU[:]:=op_LU'[-2,1]*cdag_origin[-1,1,2]*op_LU[2,-3];
        @tensor C_LU[:]:=op_LU'[-2,1]*c_origin[-1,1,2]*op_LU[2,-3];
        @tensor Sa_LU[:]:=op_LU'[-2,1]*Sa[-1,1,2]*op_LU[2,-3];
        @tensor Sb_LU[:]:=op_LU'[-2,1]*Sb[-1,1,2]*op_LU[2,-3];

        @tensor Cdag_LD[:]:=op_LD'[-2,1]*cdag_origin[-1,1,2]*op_LD[2,-3];
        @tensor C_LD[:]:=op_LD'[-2,1]*c_origin[-1,1,2]*op_LD[2,-3];
        @tensor Sa_LD[:]:=op_LD'[-2,1]*Sa[-1,1,2]*op_LD[2,-3];
        @tensor Sb_LD[:]:=op_LD'[-2,1]*Sb[-1,1,2]*op_LD[2,-3];

        @tensor Cdag_RD[:]:=op_RD'[-2,1]*cdag_origin[-1,1,2]*op_RD[2,-3];
        @tensor C_RD[:]:=op_RD'[-2,1]*c_origin[-1,1,2]*op_RD[2,-3];
        @tensor Sa_RD[:]:=op_RD'[-2,1]*Sa[-1,1,2]*op_RD[2,-3];
        @tensor Sb_RD[:]:=op_RD'[-2,1]*Sb[-1,1,2]*op_RD[2,-3];

        @tensor Cdag_RU[:]:=op_RU'[-2,1]*cdag_origin[-1,1,2]*op_RU[2,-3];
        @tensor C_RU[:]:=op_RU'[-2,1]*c_origin[-1,1,2]*op_RU[2,-3];
        @tensor Sa_RU[:]:=op_RU'[-2,1]*Sa[-1,1,2]*op_RU[2,-3];
        @tensor Sb_RU[:]:=op_RU'[-2,1]*Sb[-1,1,2]*op_RU[2,-3];

        ex=hopping_x(CTM_cell,Cdag_LU,C_RU,A_cell,AA_cell,cx,cy,ctm_setting);
        ey=hopping_y(CTM_cell,Cdag_RU,C_RD,A_cell,AA_cell,cx,cy,ctm_setting);
        e_diagonala=hopping_diagonala(CTM_cell,Cdag_LD,C_RU,A_cell,AA_cell,cx,cy,ctm_setting);
        e0=ob_onsite(CTM_cell,N_occu_set[1],A_cell,AA_cell,cx,cy,ctm_setting);
        eU=ob_onsite(CTM_cell,n_double_set[1]-(1/2)*N_occu_set[1]+(1/4)*Ident_set[1],A_cell,AA_cell,cx,cy,ctm_setting);

        # up_triangle=ob_up_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,RD,RU
        # dn_triangle=-ob_dn_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,LU,RU

        # if abs(J)>0
            SS_x=hopping_x_no_sign(CTM_cell,Sa_LU,Sb_RU,A_cell,AA_cell,cx,cy,ctm_setting);
            SS_y=hopping_y_no_sign(CTM_cell,Sa_RU,Sb_RD,A_cell,AA_cell,cx,cy,ctm_setting);
            SS_diagonal=hopping_diagonala_no_sign(CTM_cell,Sa_LD,Sb_RU,A_cell,AA_cell,cx,cy,ctm_setting);
        # end


        @ignore_derivatives ex_set[cx,cy]=ex;
        @ignore_derivatives ey_set[cx,cy]=ey;
        @ignore_derivatives e_diagonala_set[cx,cy]=e_diagonala;
        @ignore_derivatives e0_set[cx,cy]=e0;
        @ignore_derivatives eU_set[cx,cy]=eU;
        @ignore_derivatives SSx_set[cx,cy]=SS_x;
        @ignore_derivatives SSy_set[cx,cy]=SS_y;
        @ignore_derivatives SSdiagonal_set[cx,cy]=SS_diagonal;
        # @ignore_derivatives triangle_up_set[px,py]=up_triangle;
        # @ignore_derivatives triangle_dn_set[px,py]=dn_triangle;

        E_temp=-t1*ex -t1*ey -t2*e_diagonala -μ*e0/2  +U*eU/2;  
        if abs(J)>0
            E_temp=E_temp +J*(SS_x+SS_y+SS_diagonal)/2;
        end
        #E_temp=-t1*ex -t1*ey -t2*e_diagonala  +U*eU/2; # do not include chemical potential
        E_total=E_total+real(E_temp+E_temp');

        # if abs(Chi_up_triangle)>0
        #     E_total=E_total+Chi_up_triangle*real(up_triangle);
        # end
        # if abs(Chi_dn_triangle)>0
        #     E_total=E_total+Chi_dn_triangle*real(dn_triangle);
        # end
                
        E_total=E_total/(Lx*Ly);
        return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set, SSx_set, SSy_set, SSdiagonal_set

    elseif energy_setting.model =="standard_triangle_Hubbard_Bfield" 
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set =@ignore_derivatives Hamiltonian_terms();
        #(Ident,Ident,), (N_occu,N_occu,), (n_hole,n_hole), (n_double,n_double,), (Cdag,Cdag,), (C,C,), (CdagupCdagdn,CdagupCdagdn), (Pairinga,Pairinga), (Pairingb,Pairingb), (Sa,Sa), (Sb,Sb), chirality_S1,chirality_S2,chirality_S3 =@ignore_derivatives  operator_terms();
        @assert Lx==1;
        @assert Ly==1;

        t1=parameters["t1"];
        t2=parameters["t2"];
        μ=parameters["μ"];
        U=parameters["U"];
        Bx=parameters["Bx"];
        By=parameters["By"];
        Bz=parameters["Bz"];
        # Chi_up_triangle=parameters["Chi_up_triangle"];
        # Chi_dn_triangle=parameters["Chi_dn_triangle"];

        ex_set=zeros(Lx,Ly)*im;
        ey_set=zeros(Lx,Ly)*im;
        e_diagonala_set=zeros(Lx,Ly)*im;
        e0_set=zeros(Lx,Ly)*im;
        eU_set=zeros(Lx,Ly)*im;
        em_set=zeros(Lx,Ly)*im;
        # triangle_up_set=zeros(Lx,Ly)*im;
        # triangle_dn_set=zeros(Lx,Ly)*im;
        
        E_total=0;
        px=1;
        py=1;

        #(cx,cy): coordinate of left-top C1 tensor
        cx=mod1(px-1,Lx);
        cy=mod1(py-1,Ly);
        cdag_origin=Cdag_set[1];
        c_origin=C_set[1];



        sx,sy,sz=@ignore_derivatives spin_operator_Z2();
        sx=sx/2;
        sy=sy/2;
        sz=sz/2;

        Bfield=sx*Bx+sy*By+sz*Bz;





        ex=hopping_x(CTM_cell,cdag_origin,c_origin,A_cell,AA_cell,cx,cy,ctm_setting);
        ey=hopping_y(CTM_cell,cdag_origin,c_origin,A_cell,AA_cell,cx,cy,ctm_setting);
        e_diagonala=hopping_diagonala(CTM_cell,cdag_origin,c_origin,A_cell,AA_cell,cx,cy,ctm_setting);
        e0=ob_onsite(CTM_cell,N_occu_set[1],A_cell,AA_cell,cx,cy,ctm_setting);
        eU=ob_onsite(CTM_cell,n_double_set[1]-(1/2)*N_occu_set[1]+(1/4)*Ident_set[1],A_cell,AA_cell,cx,cy,ctm_setting);
        em=ob_onsite(CTM_cell,Bfield,A_cell,AA_cell,cx,cy,ctm_setting);

        # up_triangle=ob_up_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,RD,RU
        # dn_triangle=-ob_dn_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,LU,RU


        @ignore_derivatives ex_set[px,py]=ex;
        @ignore_derivatives ey_set[px,py]=ey;
        @ignore_derivatives e_diagonala_set[px,py]=e_diagonala;
        @ignore_derivatives e0_set[px,py]=e0;
        @ignore_derivatives eU_set[px,py]=eU;
        @ignore_derivatives em_set[px,py]=em;
        # @ignore_derivatives triangle_up_set[px,py]=up_triangle;
        # @ignore_derivatives triangle_dn_set[px,py]=dn_triangle;

        E_temp=-t1*ex -t1*ey -t2*e_diagonala -μ*e0/2  +U*eU/2 +em/2;
        #E_temp=-t1*ex -t1*ey -t2*e_diagonala  +U*eU/2; # do not include chemical potential
        E_total=E_total+real(E_temp+E_temp');

        # if abs(Chi_up_triangle)>0
        #     E_total=E_total+Chi_up_triangle*real(up_triangle);
        # end
        # if abs(Chi_dn_triangle)>0
        #     E_total=E_total+Chi_dn_triangle*real(dn_triangle);
        # end
                
        E_total=E_total/(Lx*Ly);
        return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set,em_set
    end
end


##########################
function ob_up_triangle(CTM,S1,S2,S3,A_cell,AA_cell,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    @tensor A_LD[:]:= A_cell[pos_LD[1]][pos_LD[2]][-1,-2,-3,-4,1]*S1[-5,1,-6]
    @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*S3[-6,-5,1]

    A_RD=A_cell[pos_RD[1]][pos_RD[2]];

    U1=@ignore_derivatives unitary(fuse(space(A_LD,3)⊗space(A_LD,6)), space(A_LD,3)⊗space(A_LD,6)); 
    U2=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
    @tensor A_LD[:]:=A_LD[-1,-2,1,-4,-5,2]*U1[-3,1,2];
    @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U2[-2,1,2];
    @tensor A_RD[:]:=A_RD[1,-2,-3,4,3]*S2[2,-5,3,5]*U1'[1,2,-1]*U2'[4,5,-4];


    if ctm_setting.grad_checkpoint
        AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_RD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
    else
        AA_LD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_RD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
    end

    ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_LD_double,AA_RD_double,cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob        
end

function ob_dn_triangle(CTM,S1,S2,S3,A_cell,AA_cell,cx,cy,ctm_setting)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    @tensor A_LD[:]:= A_cell[pos_LD[1]][pos_LD[2]][-1,-2,-3,-4,1]*S1[-5,1,-6]
    @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*S3[-6,-5,1]

    A_LU=A_cell[pos_LU[1]][pos_LU[2]];

    U1=@ignore_derivatives unitary(fuse(space(A_LD,4)⊗space(A_LD,6)), space(A_LD,4)⊗space(A_LD,6)); 
    U2=@ignore_derivatives unitary(fuse(space(A_RU,1)⊗space(A_RU,6)), space(A_RU,1)⊗space(A_RU,6)); 
    @tensor A_LD[:]:=A_LD[-1,-2,-3,1,-5,2]*U1[-4,1,2];
    @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,2]*U2[-1,1,2];
    @tensor A_LU[:]:=A_LU[-1,1,4,-4,3]*S2[2,-5,3,5]*U1'[1,2,-2]*U2'[4,5,-3];


    if ctm_setting.grad_checkpoint
        AA_LD_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
        AA_RU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_LU_double,_,_,_,_=Zygote.checkpointed(build_double_layer_swap, A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
    else
        AA_LD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
        AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
        AA_LU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
    end

    ob=ob_2x2(CTM,AA_LU_double,AA_RU_double,AA_LD_double,AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob        
end

function evaluate_spin_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    

    global Lx,Ly

    if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}}) 
        Hamiltonian_terms=Operators_spinful_Z2;
    elseif isa(space(A_cell[1][1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Operators_spinful_SU2;
    end

    (Ident,Ident,), (N_occu,N_occu,), (n_hole,n_hole), (n_double,n_double,), (Cdag,Cdag,), (C,C,), (CdagupCdagdn,CdagupCdagdn), (Pairinga,Pairinga), (Pairingb,Pairingb), (Sa,Sa), (Sb,Sb), chirality_S1,chirality_S2,chirality_S3 =@ignore_derivatives  Hamiltonian_terms();

    if energy_setting.model=="spinful_triangle_lattice"

        @assert mod(Lx,2)==0
        #for 120 degree magnetic order in the Hofstadter M2 model. Unit-cell for 120 degree order should be at least 3x3.  
        

        triangle_up_set=zeros(Lx,Ly)*im;
        triangle_dn_set=zeros(Lx,Ly)*im;

        SS_x_set=zeros(Lx,Ly)*im;
        SS_y_set=zeros(Lx,Ly)*im;
        SS_diagonal_set=zeros(Lx,Ly)*im;

        
        

        for cx=1:Lx
            for cy=1:Ly
                
                #expectation value for chirality operator
                up_triangle=ob_up_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,RD,RU
                dn_triangle=ob_dn_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,LU,RU
                dn_triangle=-dn_triangle;

                #expectation value for Heisenberg operator
                SS_x=hopping_x_no_sign(CTM_cell,Sa,Sb,A_cell,AA_cell,cx,cy,ctm_setting);
                SS_y=hopping_y_no_sign(CTM_cell,Sa,Sb,A_cell,AA_cell,cx,cy,ctm_setting);
                SS_diagonal=hopping_diagonala_no_sign(CTM_cell,Sa,Sb,A_cell,AA_cell,cx,cy,ctm_setting);

                @ignore_derivatives triangle_up_set[cx,cy]=up_triangle;
                @ignore_derivatives triangle_dn_set[cx,cy]=dn_triangle;
                @ignore_derivatives SS_x_set[cx,cy]=SS_x;
                @ignore_derivatives SS_y_set[cx,cy]=SS_y;
                @ignore_derivatives SS_diagonal_set[cx,cy]=SS_diagonal;


            end
        end


        return triangle_up_set,triangle_dn_set,SS_x_set,SS_y_set,SS_diagonal_set
        
    elseif energy_setting.model =="standard_triangle_Hubbard" 
        
        #for 120 degree magnetic order in the Hofstadter M2 model. Unit-cell for 120 degree order should be at least 3x3.  
        

        triangle_up_set=zeros(Lx,Ly)*im;
        triangle_dn_set=zeros(Lx,Ly)*im;

        SS_x_set=zeros(Lx,Ly)*im;
        SS_y_set=zeros(Lx,Ly)*im;
        SS_diagonal_set=zeros(Lx,Ly)*im;

        
        

        for cx=1:Lx
            for cy=1:Ly
                
                #expectation value for chirality operator
                up_triangle=ob_up_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,RD,RU
                dn_triangle=-ob_dn_triangle(CTM_cell,chirality_S1,chirality_S2,chirality_S3,A_cell,AA_cell,cx,cy,ctm_setting);#LD,LU,RU
                

                #expectation value for Heisenberg operator
                SS_x=hopping_x_no_sign(CTM_cell,Sa,Sb,A_cell,AA_cell,cx,cy,ctm_setting);
                SS_y=hopping_y_no_sign(CTM_cell,Sa,Sb,A_cell,AA_cell,cx,cy,ctm_setting);
                SS_diagonal=hopping_diagonala_no_sign(CTM_cell,Sa,Sb,A_cell,AA_cell,cx,cy,ctm_setting);

                @ignore_derivatives triangle_up_set[cx,cy]=up_triangle;
                @ignore_derivatives triangle_dn_set[cx,cy]=dn_triangle;
                @ignore_derivatives SS_x_set[cx,cy]=SS_x;
                @ignore_derivatives SS_y_set[cx,cy]=SS_y;
                @ignore_derivatives SS_diagonal_set[cx,cy]=SS_diagonal;


            end
        end


        return triangle_up_set,triangle_dn_set,SS_x_set,SS_y_set,SS_diagonal_set

    end
end




function get_Hofstadter_coefficients(Lx,Ly,parameters,energy_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)=(1-1,1-1)
    defining terms: (px,py)=(1,1)
    """    
    @assert mod(Lx,energy_setting.Magnetic_cell)==0;
    t1=parameters["t1"];
    t2=parameters["t2"];
    μ=parameters["μ"];
    U=parameters["U"];
    tx_coe_set=Matrix{ComplexF64}(undef,Lx,Ly);
    ty_coe_set=Matrix{ComplexF64}(undef,Lx,Ly);
    t2_coe_set=Matrix{ComplexF64}(undef,Lx,Ly);
    U_coe_set=Matrix{ComplexF64}(undef,Lx,Ly);
    μ_coe_set=Matrix{ComplexF64}(undef,Lx,Ly);
    phi=2*pi/(energy_setting.Magnetic_cell);
    for px=1:Lx
        for py=1:Ly
            tx_coe_set[px,py]=-t1';#(px,py+1) <- (px+1,py+1)
            ty_coe_set[px,py]=-t1*exp(im*(px+1)*phi)'; #(px+1,py+1) <- (px+1,py)
            t2_coe_set[px,py]=t2*exp(im*((px+1)*phi-phi/2)); #(px,py+1) <- (px+1,py)
            U_coe_set[px,py]=U;
            μ_coe_set[px,py]=μ;
        end
    end
    parameters_site=Dict([("tx_coe_set", tx_coe_set),("ty_coe_set", ty_coe_set), ("t2_coe_set",  t2_coe_set), ("U_coe_set",  U_coe_set), ("μ_coe_set", μ_coe_set)]);
    return parameters_site
end









