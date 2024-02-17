function projector_virtual(V::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})


    if V==Rep[SU₂](0=>2, 1/2=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1
        T=TensorMap(M,Rep[SU₂](0=>2),V);
        P_even[1]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1),V);
        P_odd[1]=T;
    elseif  V==Rep[SU₂](0=>2, 1/2=>1)'
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1
        T=TensorMap(M,Rep[SU₂](0=>2)',V);
        P_even[1]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1)',V);
        P_odd[1]=T;
    end


    return P_odd,P_even
end

function projector_physical(V::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})

    if V==Rep[SU₂](1/2=>1)


        P_even=[];

        M=zeros(2,2)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1),Rep[SU₂](1/2=>1));
        P_odd=T;
    elseif V==Rep[SU₂](1/2=>1)'

        P_even=[];

        M=zeros(2,2)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1)',Rep[SU₂](1/2=>1)');
        P_odd=T;
    elseif V==Rep[SU₂](0=>2,1/2=>1)
        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[SU₂](0=>2),Rep[SU₂](0=>2,1/2=>1));
        P_even=T;

        M=zeros(2,4)*im;
        M[1,2+1]=1;
        M[2,2+2]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1),Rep[SU₂](0=>2,1/2=>1));
        P_odd=T;
    elseif V==Rep[SU₂](0=>2,1/2=>1)'
        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[SU₂](0=>2)',Rep[SU₂](0=>2,1/2=>1)');
        P_even=T;

        M=zeros(2,4)*im;
        M[1,2+1]=1;
        M[2,2+2]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1)',Rep[SU₂](0=>2,1/2=>1)');
        P_odd=T;
    end

    

    return P_odd,P_even

end

function projector_virtual(V) #U1 or U1xSU2
    VV1=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'
    VV2=Rep[U₁ × SU₂]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1)
    
    if V==Rep[U₁](0=>1, 1=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(1,2)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁](0=>1),V);
        P_even[1]=T;

        M=zeros(1,2)*im;
        M[1,2]=1;
        T=TensorMap(M,Rep[U₁](1=>1),V);
        P_odd[1]=T;
    elseif V==Rep[U₁](0=>1, -1=>2, -2=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,2);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁](0=>1),V);
        P_even[1]=T;

        M=zeros(1,4)*im;
        M[1,4]=1;
        T=TensorMap(M,Rep[U₁](-2=>1),V);
        P_even[2]=T;

        M=zeros(2,4)*im;
        M[1,2]=1;
        M[2,3]=1;
        T=TensorMap(M,Rep[U₁](-1=>2),V);
        P_odd[1]=T;
    elseif V==VV1
        P_odd=Vector(undef,1);
        P_even=Vector(undef,2);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>1)',V);
        P_even[1]=T;

        M=zeros(1,4)*im;
        M[1,2]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((2, 0)=>1)',V);
        P_even[2]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((1, 1/2)=>1)',V);
        P_odd[1]=T;
    elseif V==VV2
        P_odd=Vector(undef,2);
        P_even=Vector(undef,4);

        M=zeros(1,16)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>1),V);
        P_even[1]=T;

        M=zeros(3,16)*im;
        M[1,2]=1;
        M[2,3]=1;
        M[3,4]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-2, 0)=>3),V);
        P_even[2]=T;

        M=zeros(1,16)*im;
        M[1,5]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-4, 0)=>1),V);
        P_even[3]=T;

        M=zeros(3,16)*im;
        M[1,14]=1;
        M[2,15]=1;
        M[3,16]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-2, 1)=>1),V);
        P_even[4]=T;

        M=zeros(4,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-1, 1/2)=>2),V);
        P_odd[1]=T;

        M=zeros(4,16)*im;
        M[1,10]=1;
        M[2,11]=1;
        M[3,12]=1;
        M[4,13]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-3, 1/2)=>2),V);
        P_odd[2]=T;

    end


    return P_odd,P_even
end

function projector_virtual_devided(V)
    VV1=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'
    VV2=Rep[U₁ × SU₂]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1)
    
    if V==VV1
        Ps=Vector(undef,3);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>1)',V);
        Ps[1]=T;

        M=zeros(1,4)*im;
        M[1,2]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((2, 0)=>1)',V);
        Ps[2]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((1, 1/2)=>1)',V);
        Ps[3]=T;
    elseif V==VV2
        Ps=Vector(undef,6);

        M=zeros(1,16)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>1),V);
        Ps[1]=T;

        M=zeros(3,16)*im;
        M[1,2]=1;
        M[2,3]=1;
        M[3,4]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-2, 0)=>3),V);
        Ps[2]=T;

        M=zeros(1,16)*im;
        M[1,5]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-4, 0)=>1),V);
        Ps[3]=T;

        M=zeros(3,16)*im;
        M[1,14]=1;
        M[2,15]=1;
        M[3,16]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-2, 1)=>1),V);
        Ps[4]=T;

        M=zeros(4,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-1, 1/2)=>2),V);
        Ps[5]=T;

        M=zeros(4,16)*im;
        M[1,10]=1;
        M[2,11]=1;
        M[3,12]=1;
        M[4,13]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-3, 1/2)=>2),V);
        Ps[6]=T;

    end


    return Ps
end


function projector_physical(V)#U1 or U1xSU2
    VV1=Rep[U₁ × SU₂]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (1, 1/2)=>2, (-1, 1/2)=>2, (0, 1)=>1)
    if V==Rep[U₁](0=>2, 1=>1, -1=>1)

        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[U₁](0=>2),V);
        P_even=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[U₁](1=>1,-1=>1),V);
        P_odd=T;
    elseif V==VV1
        M=zeros(8,16)*im;
        M[1,1]=1;
        M[2,2]=1;
        M[3,3]=1;
        M[4,4]=1;
        M[5,5]=1;
        M[6,14]=1;
        M[7,15]=1;
        M[8,16]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (0, 1)=>1),V);
        P_even=T;

        M=zeros(8,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        M[5,10]=1;
        M[6,11]=1;
        M[7,12]=1;
        M[8,13]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((1, 1/2)=>2, (-1, 1/2)=>2),V);
        P_odd=T;

    end

    

    return P_odd,P_even

end

function projector_general_SU2_U1(V1)
    Prime=false;
    if string(V1)[end]=='\''
        Prime=true;
    end


    Qnlist=[];
    Spinlist=[];
    
    for s in sectors(V1)
        # println(s)
        # println(dim(V1,s))
        st=replace(string(s), "Irrep[U₁]" => "a");
        st=replace(st, "⊠ Irrep[SU₂]" => "a");
        #println(st)
        left_pos,right_pos,slash_pos=QN_str_search(string(st));

        Qn=parse(Int64, st[left_pos[2]+1:right_pos[1]-1])
        if length(slash_pos)>0
            @assert length(slash_pos)==1
            Numerator=parse(Int64, st[left_pos[3]+1:slash_pos[1]-1])
            Denominator=parse(Int64, st[slash_pos[1]+1:right_pos[2]-1])
            Spin=Numerator/Denominator
        else
            Spin=Numerator=parse(Int64, st[left_pos[3]+1:right_pos[2]-1])
        end
        #println(Spin)
        Dim=dim(V1, s)
        #Dim=Int(Dim*(2*Spin+1))
        for cc=1:Dim
            Qnlist=vcat(Qnlist,Int(Qn));
            Spinlist=vcat(Spinlist,Spin);

        end
        #Qnlist=vcat(Qnlist,Int.(ones(Dim))*Qn);
        
    end
    # println(Qnlist)
    # println(Spinlist)

    L=length(Qnlist);
    Ps=Vector(undef,L);
    total_dim=Int(sum(Spinlist*2 .+1));
    posit=0;
    for cc=1:L
        S=Spinlist[cc];
        Qn=Qnlist[cc];
        M=zeros(Int(2*S+1),total_dim)
        for dd=1:Int(2*S+1)
            M[dd,posit+dd]=1;
        end
        posit=posit+Int(2*S+1);
        if Prime
            T=TensorMap(M,(Rep[U₁ × SU₂]((-Qn, S)=>1))', V1);
        else
            T=TensorMap(M,Rep[U₁ × SU₂]((Qn, S)=>1),V1);
        end
        Ps[cc]=T;
    end

    #check
    @tensor T[:]:=Ps[1]'[-1,1]*Ps[1][1,-2];
    for cc=2:length(Ps);
        @tensor TT[:]:=Ps[cc]'[-1,1]*Ps[cc][1,-2];
        T=T+TT;
    end
    @assert norm(permute(T,(1,),(2,))-unitary(V1,V1))<1e-10

    return Ps
end


function build_double_layer_NoSwap(Ap,A)
    #display(space(A))



    




    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5));
    
    # U_L=unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    # U_D=unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    # U_R=inv(U_L);
    # U_U=inv(U_D);

    # U_Lp=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    # U_Dp=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    # U_Rp=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    # U_Up=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # println(norm(U_R-U_Rp)/norm(U_R))
    # println(norm(U_L-U_Lp)/norm(U_L))
    # println(norm(U_D-U_Dp)/norm(U_D))
    # println(norm(U_U-U_Up)/norm(U_U))

    U_L=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    U_D=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    U_R=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    U_U=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uMp,sMp,vMp=tsvd(Ap);
    uMp=uMp*sMp;
    uM,sM,vM=tsvd(A);
    uM=uM*sM;

    uMp=permute(uMp,(1,2,3,),())
    uM=permute(uM,(1,2,3,),())
    Vp=space(uMp,3);
    V=space(vM,1);
    U=unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

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

