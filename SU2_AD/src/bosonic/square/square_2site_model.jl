function spin_op_SU2()
    Pm=zeros(4,4)*im;Pm[1,1]=1;Pm[2,4]=1;Pm[3,2]=1;Pm[4,3]=1;
    #V=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(1,1/2)=>1,(2,0)=>1);#element order after converting to dense: <0,0>, <up,down>, <up,0>, <0,down>, 
    V=SU2Space(0=>2,1/2=>1);

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
    #Vspin=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1);
    Vspin=SU2Space(1/2=>1)
    P_G=TensorMap(P, Vspin'  ←  V');

    @tensor SS_op_S[:]:=P_G[-1,1]*P_G[-2,2]*SS_op_F[1,2,3,4]*P_G'[3,-3]*P_G'[4,-4];
    SS_op_S=permute(SS_op_S,(1,2,),(3,4,))

    @tensor Schiral_op_S[:]:=P_G[-1,1]*P_G[-2,2]*P_G[-3,3]*Schiral_op_F[1,2,3,4,5,6]*P_G'[4,-4]*P_G'[5,-5]*P_G'[6,-6];
    Schiral_op_S=permute(Schiral_op_S,(1,2,3,),(4,5,6,));
    return SS_op_S,Schiral_op_S
end

function Hamiltonian_terms_2site_SU2()
    SS_op,Schiral_op=spin_op_SU2();

    #v1=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((6,0)=>1)*GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1)'*GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1,1/2)=>1)';
    v1=SU2Space(0=>1)*SU2Space(1/2=>1)'*SU2Space(1/2=>1)';
    U_phy1=unitary(fuse(v1),v1);
    #v2=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((4,0)=>1,(4,1)=>1)*GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,0)=>1)*GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,0)=>1);
    v2=SU2Space(0=>1,1=>1)*SU2Space(0=>1)*SU2Space(0=>1);
    U_phy2=unitary(fuse(v2),v2);



    
    #println("construct physical operators");flush(stdout);
    #spin-spin operator act on a single site
    um,sm,vm=@ignore_derivatives tsvd(permute(SS_op,(1,3,),(2,4,)));
    vm=sm*vm;vm=permute(vm,(2,3,),(1,));

    @tensor SS_cell[:]:=SS_op[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];#spin-spin operator inside a unitcell

    @tensor S_R_a[:]:=um[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];#second physical site
    @tensor S_L_a[:]:=um[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];#first physical site
    @tensor S_R_b[:]:=vm[1,4,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,4,2,-2];#second physical site
    @tensor S_L_b[:]:=vm[2,5,-3]*U_phy1[-1,3,1,2]*U_phy1'[3,1,5,-2];#first physical site
    
       
    @tensor SS_cell[:]:=SS_cell[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_L_a[:]:=S_L_a[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_R_a[:]:=S_R_a[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_L_b[:]:=S_L_b[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];
    @tensor S_R_b[:]:=S_R_b[3,4,-3]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];




    ####################################
    #chiral operator act on a single site: Si Sj Sk
    um,sm,vm=@ignore_derivatives tsvd(Schiral_op,(1,4,),(2,3,5,6,));
    vm=sm*vm;
    Si=permute(um,(1,2,3,));#P1,P1',D1
    um,sm,vm=@ignore_derivatives tsvd(vm,(1,2,4,),(3,5,));#D1,P2,P2',P3,P3'
    vm=sm*vm;
    Sj=permute(um,(2,3,1,4,));#P2,P2', D1,D2
    Sk=permute(vm,(2,3,1,))#P3,P3',D2
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


function spin_op_U1_SU2()
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

function Hamiltonian_terms_2site_U1_SU2()
    SS_op,Schiral_op=spin_op_U1_SU2();

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




function build_double_layer_extra_leg(Ap,A)
    #The last index of A tensor is an extra virtual index, such as that comes from decomposition of Heisenberg interaction


    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5,6));

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
    vM=permute(vM,(1,2,3,4,5,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5,-6];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2,-6];

    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))


    double_RU=permute(double_RU,(1,4,5,6,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,5,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,4,3,));
    AA_fused=double_LD*double_RU;


    return AA_fused, U_L,U_D,U_R,U_U
end

function build_double_layer_op(A1,O1,has_extra_leg)

    if has_extra_leg
        @tensor A1_new[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1,-6]#the last index is extra
        A1_double,_,_,_,_=build_double_layer_extra_leg(A1',A1_new)
    else
        A1_double,_,_,_,_=build_double_layer(A1,permute(O1,(2,),(1,)));
    end

    return A1_double
end

function energy_2site(parameter,A_fused,AA_fused,CTM)

    if isa(space(A_fused,1),  GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        SS_cell, S_L_a, S_R_a, S_L_b, S_R_b, Si_left, Si_right, Sj_left, Sj_right, Sk_left, Sk_right, SiSj_op, SjSi_op, SjSk_op, SkSj_op,U_Schiral=@ignore_derivatives Hamiltonian_terms_2site_SU2();
    elseif isa(space(A_fused,1), GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
        SS_cell, S_L_a, S_R_a, S_L_b, S_R_b, Si_left, Si_right, Sj_left, Sj_right, Sk_left, Sk_right, SiSj_op, SjSi_op, SjSk_op, SkSj_op,U_Schiral=@ignore_derivatives Hamiltonian_terms_2site_U1_SU2();
    end

    #println("construct double layer tensor with operator");flush(stdout);
    AA_SS=build_double_layer_op(A_fused,SS_cell,false);
    AA_SLa=build_double_layer_op(A_fused,S_L_a,true);
    AA_SRa=build_double_layer_op(A_fused,S_R_a,true);
    AA_SLb=build_double_layer_op(A_fused,S_L_b,true);
    AA_SRb=build_double_layer_op(A_fused,S_R_b,true);

    AA_SiL=build_double_layer_op(A_fused,Si_left,true);
    AA_SiR=build_double_layer_op(A_fused,Si_right,true);
    AA_SjL=build_double_layer_op(A_fused,Sj_left,true);
    AA_SjR=build_double_layer_op(A_fused,Sj_right,true);
    AA_SkL=build_double_layer_op(A_fused,Sk_left,true);
    AA_SkR=build_double_layer_op(A_fused,Sk_right,true);
    AA_SiSj=build_double_layer_op(A_fused,SiSj_op,true);
    AA_SjSi=build_double_layer_op(A_fused,SjSi_op,true);
    AA_SjSk=build_double_layer_op(A_fused,SjSk_op,true);
    AA_SkSj=build_double_layer_op(A_fused,SkSj_op,true);


    #println("Calculate energy terms:");flush(stdout);

    #############################################################
    # Norm_1=ob_1site_closed(CTM,AA_fused)
    # Norm_2x=norm_2sites_x(CTM,AA_fused)
    # Norm_2y=norm_2sites_y(CTM,AA_fused)
    # Norm_4=norm_2x2(CTM,AA_fused);

    # #J1 term
    # E_1_a=ob_1site_closed(CTM,AA_SS)/Norm_1
    # E_1_b=ob_2sites_x(CTM,AA_SRa,AA_SLb)/Norm_2x
    # E_1_c=ob_2sites_y(CTM,AA_SLa,AA_SLb)/Norm_2y
    # E_1_d=ob_2sites_y(CTM,AA_SRa,AA_SRb)/Norm_2y

    # #J2 term
    # E_2_a=ob_2sites_y(CTM,AA_SLa,AA_SRb)/Norm_2y
    # E_2_b=ob_2sites_y(CTM,AA_SRa,AA_SLb)/Norm_2y
    # E_2_c=ob_LU_RD(CTM,AA_fused,AA_SRa,AA_SLb)/Norm_4
    # E_2_d=ob_RU_LD(CTM,AA_fused,AA_SLa,AA_SRb)/Norm_4



    # #chiral term
    # E_C_a=ob_2sites_y(CTM,AA_SiSj,AA_SkR)/Norm_2y
    # E_C_b=ob_2sites_y(CTM,AA_SiR,AA_SkSj)/Norm_2y
    # E_C_c=ob_2sites_y(CTM,AA_SjSk,AA_SiL)/Norm_2y
    # E_C_d=ob_2sites_y(CTM,AA_SiSj,AA_SkL)/Norm_2y

    # E_C_e=ob_LD_LU_RU(CTM,AA_fused,AA_SiR,AA_SjR,AA_SkL,U_Schiral)/Norm_4
    # E_C_f=ob_LU_RU_RD(CTM,AA_fused,AA_SiR,AA_SjL,AA_SkL,U_Schiral)/Norm_4
    # E_C_g=ob_RU_RD_LD(CTM,AA_fused,AA_SiL,AA_SjL,AA_SkR,U_Schiral)/Norm_4
    # E_C_h=ob_RD_LD_LU(CTM,AA_fused,AA_SiL,AA_SjR,AA_SkR,U_Schiral)/Norm_4

    #############################################################

    Norm_4=norm_2x2(CTM,AA_fused);

    #J1 term
    E_1_a=ob_1site_closed(CTM,AA_fused,AA_SS)/Norm_4
    E_1_b=ob_2sites_x(CTM,AA_fused,AA_SRa,AA_SLb)/Norm_4
    E_1_c=ob_2sites_y(CTM,AA_fused,AA_SLa,AA_SLb)/Norm_4
    E_1_d=ob_2sites_y(CTM,AA_fused,AA_SRa,AA_SRb)/Norm_4

    #J2 term
    E_2_a=ob_2sites_y(CTM,AA_fused,AA_SLa,AA_SRb)/Norm_4
    E_2_b=ob_2sites_y(CTM,AA_fused,AA_SRa,AA_SLb)/Norm_4
    E_2_c=ob_LU_RD(CTM,AA_fused,AA_SRa,AA_SLb)/Norm_4
    E_2_d=ob_RU_LD(CTM,AA_fused,AA_SLa,AA_SRb)/Norm_4



    #chiral term
    E_C_a=ob_2sites_y(CTM,AA_fused,AA_SiSj,AA_SkR)/Norm_4
    E_C_b=ob_2sites_y(CTM,AA_fused,AA_SiR,AA_SkSj)/Norm_4
    E_C_c=ob_2sites_y(CTM,AA_fused,AA_SiSj,AA_SkL)/Norm_4
    E_C_d=ob_2sites_y(CTM,AA_fused,AA_SiL,AA_SkSj)/Norm_4

    E_C_e=ob_LD_LU_RU(CTM,AA_fused,AA_SiR,AA_SjR,AA_SkL,U_Schiral)/Norm_4
    E_C_f=ob_LU_RU_RD(CTM,AA_fused,AA_SiR,AA_SjL,AA_SkL,U_Schiral)/Norm_4
    E_C_g=ob_RU_RD_LD(CTM,AA_fused,AA_SiL,AA_SjL,AA_SkR,U_Schiral)/Norm_4
    E_C_h=ob_RD_LD_LU(CTM,AA_fused,AA_SiL,AA_SjR,AA_SkR,U_Schiral)/Norm_4

    #############################################################
    global energy_setting
    if energy_setting.print_all_terms
        println("J1 terms:")
        println(E_1_a)
        println(E_1_b)
        println(E_1_c)
        println(E_1_d)
        println("J2 terms:")
        println(E_2_a)
        println(E_2_b)
        println(E_2_c)
        println(E_2_d)
        println("Jchi terms:")
        println(E_C_a)
        println(E_C_b)
        println(E_C_c)
        println(E_C_d)
        println(E_C_e)
        println(E_C_f)
        println(E_C_g)
        println(E_C_h)
    end


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




function ob_1site_closed(CTM,AA_fused,AA_op)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA_op[4,-2,-4,3]; 
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



function ob_2sites_x(CTM,AA_fused,AA1,AA2)

    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA1[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA2[-2,-4,4,3,-5]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA_fused[3,4,-4,-2]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(1,2,),(3,4,5,));
    MM_RU=permute(MM_RU,(1,2,5,),(3,4,));
    MM_LD=permute(MM_LD,(1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    ob=@tensor up[1,2,3,4]*down[1,2,3,4];
    return ob
end



function ob_2sites_y(CTM,AA_fused,AA1,AA2)
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset.C1[1,2]*Tset.T1[2,3,-3]*Tset.T4[-1,4,1]*AA1[4,-2,-4,3,-5]; 
    @tensor MM_RU[:]:=Tset.T1[-1,3,1]* Cset.C2[1,2]* AA_fused[-2,-4,4,3]* Tset.T2[2,4,-3];

    @tensor MM_LD[:]:=Tset.T4[1,3,-1]*AA2[3,4,-4,-2,-5]*Cset.C4[2,1]*Tset.T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset.T2[-4,-3,2]*Tset.T3[1,-2,-1]*Cset.C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

    MM_LU=permute(MM_LU,(5,1,2,),(3,4,));
    MM_RU=permute(MM_RU,(1,2,),(3,4,));
    MM_LD=permute(MM_LD,(5,1,2,),(3,4,));
    MM_RD=permute(MM_RD,(1,2,),(3,4,));

    up=MM_LU*MM_RU;
    down=MM_LD*MM_RD;
    ob=@tensor up[1,2,3,4,5]*down[1,2,3,4,5];
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

