function Rank(T::TensorMap)
    return length(domain(T))+length(codomain(T))
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


    return Ident, occu, Cdag, C, Cdag_ 
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

    Cup=zeros(2,4,4);
    Cup[1,[1,4,3,2],[1,4,3,2]]=kron(sm,Id);
    Cdn=zeros(2,4,4);
    Cdn[2,[1,4,3,2],[1,4,3,2]]=kron(sz,sm);
    C=TensorMap(Cup+Cdn, Vdummy ⊗ V ← V);



    # # method 2
    # order=[1,4,3,2];
    # Cdagup=kron(sp,Id);
    # Cdagup=Cdagup[order,order];
    # Cup=kron(sm,Id);
    # Cup=Cup[order,order];
    # Cdagdn=kron(sz,sp);
    # Cdagdn=Cdagdn[order,order];
    # Cdn=kron(sz,sm);
    # Cdn=Cdn[order,order];
    # @tensor hop_up[:]:=Cdagup[-1,-2]*Cup[-3,-4];
    # @tensor hop_dn[:]:=Cdagdn[-1,-2]*Cdn[-3,-4];
    # Hop=TensorMap(hop_up+hop_dn,V*V' ← V'*V);
    # u,s,v=tsvd(Hop,trunc=truncdim(2));
    # u=u*s;
    # @assert norm(u*v-Hop)/norm(Hop)<1e-12;

    # #compare two methods
    # @tensor hop1[:]:=Cdag[-1,-2,1]*C[1,-3,-4];
    # hop1=permute(hop1,(1,2,),(3,4,));
    # @assert norm(hop1-Hop)<1e-12;


    Cdag=permute(Cdag,(3,1,),(2,))
    return Ident, N_occu, n_double, Cdag, C
end

function Hamiltonians_spinful_Z2()
    

    Vdummy=Rep[ℤ₂](1=>1);
    V=Rep[ℤ₂](0=>2,1=>2);


    Id=[1.0 0;0 1.0];
    sm=[0 1.0;0 0]; sp=[0 0;1.0 0]; sz=[1.0 0; 0 -1.0]; occu=[0 0; 0 1.0];
    
    #order of kron() command: (0,0), (0,1), (1,0), (1,1)
    order=[1,4,3,2];

    Ident=kron(Id,Id);
    Ident=TensorMap(Ident[order,order],  V  ←  V);

    N_occu=kron(occu,Id)+kron(Id,occu);
    N_occu=TensorMap(N_occu[order,order],  V ←  V);
    n_double=kron(occu,occu)
    n_double=TensorMap(n_double[order,order],  V ←  V);

    # method 1
    Cdagup=zeros(4,4,1);
    Cdagup[order,order,1]=kron(sp,Id);
    Cdagdn=zeros(4,4,1);
    Cdagdn[order,order,1]=kron(sz,sp);
    Cdagup=TensorMap(Cdagup,  V ← V ⊗Vdummy);Cdagup=permute(Cdagup,(3,1,),(2,))
    Cdagdn=TensorMap(Cdagdn,  V ← V ⊗Vdummy);Cdagdn=permute(Cdagdn,(3,1,),(2,))

    Cup=zeros(1,4,4);
    Cup[1,order,order]=kron(sm,Id);
    Cdn=zeros(1,4,4);
    Cdn[1,order,order]=kron(sz,sm);
    Cup=TensorMap(Cup, Vdummy ⊗ V ← V);
    Cdn=TensorMap(Cdn, Vdummy ⊗ V ← V);



    return Ident, N_occu, n_double, Cdagup, Cup, Cdagdn, Cdn
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




function evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, ctm_setting, energy_setting)
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """    

    global Lx,Ly

    if isa(space(A_cell[1][1],1),GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_Z2;
    elseif isa(space(A_cell[1][1],1),GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinless_U1;
    elseif isa(space(A_cell[1][1],1),GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Hamiltonian_terms=Hamiltonians_spinful_SU2;
    elseif isa(space(A_cell[1][1],1),GradedSpace{ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
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
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala')  +U*eU);

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
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')+t1*(ey+ey')+t2*(e_diagonala+e_diagonala')  +U*eU);
            
                
            
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
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala')  +U*eU);

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
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')-t1*(ey+ey')-t2*(e_diagonala+e_diagonala')  +U*eU);

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
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')+t1*(ey+ey')+t2*(e_diagonala+e_diagonala')  +U*eU);

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
            E_total=E_total+real(t1*(exp(im*ϕ)*ex+exp(-im*ϕ)*ex')+t1*(ey+ey')+t2*(e_diagonala+e_diagonala')  +U*eU);
            
                
            
            E_total=E_total/(Lx*Ly);
            return E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set
        end
    end
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



# function ob_LU_RU_LD_cell(cx,cy,CTM,AA_,AA_LU_,AA_RU_,AA_LD_)
#     global Lx,Ly
#     Cset=CTM.Cset;
#     Tset=CTM.Tset;

#     @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-4]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-2,4,1]*AA_LU_[4,-3,-5,3,-1]; 
#     @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

#     @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_LD_[3,4,-5,-3,-1]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
#     @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_[-2,1,2,-4]; 

#     MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
#     MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:= up[-1,1,2,3,4,-2]*down[-3,1,2,3,4];
#     return rho
# end

# function ob_LD_RU_RD_cell(cx,cy,CTM,AA_,AA_LD_,AA_RU_,AA_RD_)
#     global Lx,Ly
#     Cset=CTM.Cset;
#     Tset=CTM.Tset;

#     @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-3]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-1,4,1]*AA_[4,-2,-4,3]; 
#     @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

#     @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_LD_[3,4,-5,-3,-1]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
#     @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4,-5]; 

#     MM_LU=permute(MM_LU,(1,2,),(3,4,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
#     MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:= up[1,2,3,4,-2]*down[-1,1,2,3,4,-3];
#     return rho
# end

# function ob_LU_LD_RD_cell(cx,cy,CTM,AA_,AA_LU_,AA_LD_,AA_RD_)
#     global Lx,Ly
#     Cset=CTM.Cset;
#     Tset=CTM.Tset;

#     @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-4]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-2,4,1]*AA_LU_[4,-3,-5,3,-1]; 
#     @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_[-2,-4,4,3]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

#     @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_LD_[3,4,-5,-3,-1]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
#     @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4,-5]; 

#     MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,));
#     MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:= up[-1,1,2,3,4]*down[-2,1,2,3,4,-3];
#     return rho
# end

# function ob_LU_RU_RD_cell(cx,cy,CTM,AA_,AA_LU_,AA_RU_,AA_RD_)
#     global Lx,Ly
#     Cset=CTM.Cset;
#     Tset=CTM.Tset;

#     @tensor MM_LU[:]:=Cset[mod1(cx,Lx)][mod1(cy,Ly)].C1[1,2]*Tset[mod1(cx+1,Lx)][mod1(cy,Ly)].T1[2,3,-4]*Tset[mod1(cx,Lx)][mod1(cy+1,Ly)].T4[-2,4,1]*AA_LU_[4,-3,-5,3,-1]; 
#     @tensor MM_RU[:]:=Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T1[-1,3,1]* Cset[mod1(cx+3,Lx)][mod1(cy,Ly)].C2[1,2]* AA_RU_[-2,-4,4,3,-5]* Tset[mod1(cx+3,Lx)][mod1(cy+1,Ly)].T2[2,4,-3];

#     @tensor MM_LD[:]:=Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T4[1,3,-2]*AA_[3,4,-5,-3]*Cset[mod1(cx,Lx)][mod1(cy+3,Ly)].C4[2,1]*Tset[mod1(cx+1,Lx)][mod1(cy+3,Ly)].T3[-4,4,2]; 
#     @tensor MM_RD[:]:=Tset[mod1(cx+3,Lx)][mod1(cy+2,Ly)].T2[-4,-3,2]*Tset[mod1(cx+2,Lx)][mod1(cy+3,Ly)].T3[1,-2,-1]*Cset[mod1(cx+3,Lx)][mod1(cy+3,Ly)].C3[2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD_[-2,1,2,-4,-5]; 

#     MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
#     MM_LD=permute(MM_LD,(1,2,),(3,4,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,5,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:= up[-1,1,2,3,4,-2]*down[1,2,3,4,-3];
#     return rho
# end
