function projector_physical(V::GradedSpace{Z2Irrep, Tuple{Int64, Int64}})

    if V==Rep[ℤ₂](0=>1, 1=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(1,2)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[ℤ₂](0=>1),V);
        P_even[1]=T;

        M=zeros(1,2)*im;
        M[1,2]=1;
        T=TensorMap(M,Rep[ℤ₂](1=>1),V);
        P_odd[1]=T;
    elseif  V==Rep[ℤ₂](0=>2, 1=>2)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[ℤ₂](0=>2),V);
        P_even[1]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[ℤ₂](1=>2),V);
        P_odd[1]=T;
    end

    

    return P_odd,P_even

end

function projector_virtual(V::GradedSpace{Z2Irrep, Tuple{Int64, Int64}})


    if V==Rep[ℤ₂](0=>1, 1=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(1,2)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[ℤ₂](0=>1),V);
        P_even[1]=T;

        M=zeros(1,2)*im;
        M[1,2]=1;
        T=TensorMap(M,Rep[ℤ₂](1=>1),V);
        P_odd[1]=T;
    elseif  V==Rep[ℤ₂](0=>2, 1=>2)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[ℤ₂](0=>2),V);
        P_even[1]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[ℤ₂](1=>2),V);
        P_odd[1]=T;
    elseif  V==Rep[ℤ₂](0=>8, 1=>8)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(8,16)*im;
        M[1,1]=1;
        M[2,2]=1;
        M[3,3]=1;
        M[4,4]=1;
        M[5,5]=1;
        M[6,6]=1;
        M[7,7]=1;
        M[8,8]=1;
        T=TensorMap(M,Rep[ℤ₂](0=>8),V);
        P_even[1]=T;

        M=zeros(8,16)*im;
        M[1,1+8]=1;
        M[2,2+8]=1;
        M[3,3+8]=1;
        M[4,4+8]=1;
        M[5,5+8]=1;
        M[6,6+8]=1;
        M[7,7+8]=1;
        M[8,8+8]=1;
        T=TensorMap(M,Rep[ℤ₂](1=>8),V);
        P_odd[1]=T;
    end


    return P_odd,P_even
end

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

    end

    

    return P_odd,P_even

end

function projector_virtual(V::GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
    VV1=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'
    VV2=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1)
    
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
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1)',V);
        P_even[1]=T;

        M=zeros(1,4)*im;
        M[1,2]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2, 0)=>1)',V);
        P_even[2]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1, 1/2)=>1)',V);
        P_odd[1]=T;
    elseif V==VV2
        P_odd=Vector(undef,2);
        P_even=Vector(undef,4);

        M=zeros(1,16)*im;
        M[1,1]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1),V);
        P_even[1]=T;

        M=zeros(3,16)*im;
        M[1,2]=1;
        M[2,3]=1;
        M[3,4]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2, 0)=>3),V);
        P_even[2]=T;

        M=zeros(1,16)*im;
        M[1,5]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-4, 0)=>1),V);
        P_even[3]=T;

        M=zeros(3,16)*im;
        M[1,14]=1;
        M[2,15]=1;
        M[3,16]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2, 1)=>1),V);
        P_even[4]=T;

        M=zeros(4,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-1, 1/2)=>2),V);
        P_odd[1]=T;

        M=zeros(4,16)*im;
        M[1,10]=1;
        M[2,11]=1;
        M[3,12]=1;
        M[4,13]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-3, 1/2)=>2),V);
        P_odd[2]=T;

    end


    return P_odd,P_even
end

function projector_virtual_devided(V)
    VV1=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'
    VV2=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1)
    
    if V==VV1
        Ps=Vector(undef,3);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1)',V);
        Ps[1]=T;

        M=zeros(1,4)*im;
        M[1,2]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((2, 0)=>1)',V);
        Ps[2]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1, 1/2)=>1)',V);
        Ps[3]=T;
    elseif V==VV2
        Ps=Vector(undef,6);

        M=zeros(1,16)*im;
        M[1,1]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1),V);
        Ps[1]=T;

        M=zeros(3,16)*im;
        M[1,2]=1;
        M[2,3]=1;
        M[3,4]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2, 0)=>3),V);
        Ps[2]=T;

        M=zeros(1,16)*im;
        M[1,5]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-4, 0)=>1),V);
        Ps[3]=T;

        M=zeros(3,16)*im;
        M[1,14]=1;
        M[2,15]=1;
        M[3,16]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2, 1)=>1),V);
        Ps[4]=T;

        M=zeros(4,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-1, 1/2)=>2),V);
        Ps[5]=T;

        M=zeros(4,16)*im;
        M[1,10]=1;
        M[2,11]=1;
        M[3,12]=1;
        M[4,13]=1;
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-3, 1/2)=>2),V);
        Ps[6]=T;

    end


    return Ps
end


function projector_physical(V)
    VV1=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (1, 1/2)=>2, (-1, 1/2)=>2, (0, 1)=>1)
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
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (0, 1)=>1),V);
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
        T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((1, 1/2)=>2, (-1, 1/2)=>2),V);
        P_odd=T;

    end

    

    return P_odd,P_even

end

function projector_general(V1::GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}})
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
            T=TensorMap(M,(GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-Qn, S)=>1))', V1);
        else
            T=TensorMap(M,GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((Qn, S)=>1),V1);
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




function projector_general(V1::GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
    Prime=V1.dual;

    Qnlist=[];
    
    for cc=1:2

        Qn=cc-1;
        Dim=V1.dims[cc];
        for cc=1:Dim
            Qnlist=vcat(Qnlist,Int(Qn));
        end

    end


    L=2;
    Ps=Vector(undef,L);
    total_dim=dim(V1);
    posit=0;
    for cc=1:2
        Qn=Qnlist[cc];
        M=zeros(V1.dims[cc],total_dim)
        for dd=1:(V1.dims[cc])
            M[dd,posit+dd]=1;
        end
        posit=posit+(V1.dims[cc]);
        if Prime
            T=TensorMap(M,Rep[ℤ₂](cc-1=>(V1.dims[cc]))', V1);
        else
            T=TensorMap(M,Rep[ℤ₂](cc-1=>(V1.dims[cc])), V1);
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