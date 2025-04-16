
function canonical_right_to_left(mps_set,pos)
    mps_set=deepcopy(mps_set);
    Lx=length(mps_set);
    @assert pos>1;
    cx=pos;
    if pos==Lx
        u,s,v=tsvd(mps_set[cx],(1,),(2,));
        u=u*s;
        mps_set[cx]=permute(v,(1,2,));
        @tensor A[:]:=mps_set[cx-1][-1,1,-3]*u[1,-2];
        mps_set[cx-1]=A;
    elseif 2<pos<Lx
        u,s,v=tsvd(mps_set[cx],(1,),(2,3,));
        u=u*s;
        mps_set[cx]=permute(v,(1,2,3,));
        @tensor A[:]:=mps_set[cx-1][-1,1,-3]*u[1,-2];
        mps_set[cx-1]=permute(A,(1,2,3,));
    elseif pos==2
        u,s,v=tsvd(mps_set[cx],(1,),(2,3,));
        u=u*s;
        mps_set[cx]=permute(v,(1,2,3,));
        @tensor A[:]:=mps_set[cx-1][1,-2]*u[1,-1];
        mps_set[cx-1]=permute(A,(1,2,));
    end
    return mps_set
end

function canonical_left_to_right(mps_set,pos)
    mps_set=deepcopy(mps_set);
    Lx=length(mps_set);
    @assert pos<Lx;
    Lx=length(mps_set);
    cx=pos;
    if pos==1
        u,s,v=tsvd(mps_set[cx],(2,),(1,));
        v=s*v;
        mps_set[cx]=permute(u,(2,1,));
        @tensor A[:]:=v[-1,1]*mps_set[cx+1][1,-2,-3];
        mps_set[cx+1]=permute(A,(1,2,3,));
    elseif 1<pos<Lx-1
        u,s,v=tsvd(mps_set[cx],(1,3,),(2,)); 
        v=s*v;
        mps_set[cx]=permute(u,(1,3,2,));
        @tensor A[:]:=v[-1,1]*mps_set[cx+1][1,-2,-3];
        mps_set[cx+1]=permute(A,(1,2,3,));
    elseif pos==Lx-1
        u,s,v=tsvd(mps_set[cx],(1,3,),(2,)); 
        v=s*v;
        mps_set[cx]=permute(u,(1,3,2,));
        @tensor A[:]:=v[-1,1]*mps_set[cx+1][1,-2];
        mps_set[cx+1]=permute(A,(1,2,));
    end
    return mps_set
end

function sweep_right_move(Cleft,mpo,mps0,mpstrun,posx)
    Lx=length(mps0);
    @assert 1<posx<Lx
    @tensor Cleft[:]:=Cleft[1,3,5]*mpstrun[posx]'[1,-1,2]*mpo[posx][3,4,-2,2]*mps0[posx][5,-3,4];
    return Cleft
end

function sweep_left_move(Cright,mpo,mps0,mpstrun,posx)
    Lx=length(mps0);
    @assert 1<posx<Lx
    @tensor Cright[:]:=Cright[1,3,5]*mpstrun[posx]'[-1,1,2]*mpo[posx][-2,4,3,2]*mps0[posx][-3,5,4];
    return Cright
end

function compress_sweep(mpo,mps0,mpstrun)
    mpstrun=deepcopy(mpstrun);
    Lx=length(mpo);

    # for cc=Lx:-1:2
    #     mpstrun=canonical_right_to_left(mpstrun,cc);
    # end

    Cright_set=Vector{TensorMap}(undef,Lx);
    Cleft_set=Vector{TensorMap}(undef,Lx);
    cp=Lx;
    # println(space(mpstrun[cp]))
    # println(space(mpo[cp]))
    # println(space(mps0[cp]))
    @tensor Cright[:]:=mpstrun[cp]'[-1,1]*mpo[cp][-2,2,1]*mps0[cp][-3,2];
    Cright_set[cp]=Cright;
    for cp=Lx-1:-1:2
        Cright=sweep_left_move(Cright,mpo,mps0,mpstrun,cp);
        Cright_set[cp]=Cright;
    end

    @assert n_mps_sweep>=1;
    for css=1:n_mps_sweep
        #################################
        #sweep from left to right
        cp=1;
        @tensor Tnew[:]:=mpo[cp][1,2,-2]*mps0[cp][3,1]*Cright_set[cp+1][-1,2,3];
        mpstrun[cp]=Tnew;#update mps
        mpstrun=canonical_left_to_right(mpstrun,cp);#shift canonical center
        @tensor Cleft[:]:=mpstrun[cp]'[-1,1]*mpo[cp][2,-2,1]*mps0[cp][-3,2];
        Cleft_set[cp]=Cleft;#update environment
        for cp=2:Lx-1
            @tensor Tnew[:]:=Cleft_set[cp-1][-1,2,1]*mpo[cp][2,3,4,-3]*mps0[cp][1,5,3]*Cright_set[cp+1][-2,4,5];
            mpstrun[cp]=Tnew;#update mps
            mpstrun=canonical_left_to_right(mpstrun,cp);#shift canonical center
            Cleft=sweep_right_move(Cleft,mpo,mps0,mpstrun,cp);
            Cleft_set[cp]=Cleft;#update environment
        end

        #################################
        #sweep from right to left
        cp=Lx;
        @tensor Tnew[:]:=Cleft_set[cp-1][-1,2,3]*mpo[cp][2,1,-2]*mps0[cp][3,1];
        mpstrun[cp]=Tnew;#update mps
        mpstrun=canonical_right_to_left(mpstrun,cp);#shift canonical center
        @tensor Cright[:]:=mpstrun[cp]'[-1,1]*mpo[cp][-2,2,1]*mps0[cp][-3,2];
        Cright_set[cp]=Cright;#update environment
        for cp=Lx-1:-1:2
            @tensor Tnew[:]:=Cleft_set[cp-1][-1,2,1]*mpo[cp][2,3,4,-3]*mps0[cp][1,5,3]*Cright_set[cp+1][-2,4,5];
            mpstrun[cp]=Tnew;#update mps
            mpstrun=canonical_right_to_left(mpstrun,cp);#shift canonical center
            Cright=sweep_left_move(Cright,mpo,mps0,mpstrun,cp);
            Cright_set[cp]=Cright;#update environment
        end
    end

    #An extra step: determine the coefficient for global state
    cp=1;
    @tensor Tnew[:]:=mpo[cp][1,2,-2]*mps0[cp][3,1]*Cright_set[cp+1][-1,2,3];
    mpstrun[cp]=Tnew;#update mps

    return mpstrun
end




function overlap_mps(mps_set1,mps_set2)
    Lx=length(mps_set1);

    cx=1
    @tensor env[:]:=mps_set1[cx]'[-1,1]*mps_set2[cx][-2,1];
    for cx=2:Lx-1
        @tensor env[:]:=env[1,3]*mps_set1[cx]'[1,-1,2]*mps_set2[cx][3,-2,2];
    end
    cx=Lx;
    Norm=@tensor env[1,2]*mps_set1[cx]'[1,3]*mps_set2[cx][2,3];
    return Norm
end
function mps_diff(mps1,mps2)
    y=overlap_mps(mps1,mps1)+overlap_mps(mps2,mps2)-overlap_mps(mps1,mps2)-overlap_mps(mps2,mps1)
    return sqrt(y)
end


function apply_mpo(mpo_set,mps_set)
    mps_set_new=deepcopy(mpo_set);
    Lx=length(mps_set);
    UR_set=Vector{TensorMap}(undef,Lx);
    UL_set=Vector{TensorMap}(undef,Lx);

    cx=1;
    A=mps_set[cx];
    M=mpo_set[cx];

    UR=unitary(fuse(space(M,2)*space(A,1)), space(M,2)*space(A,1));
    UR_set[cx]=UR;
    @tensor A_new[:]:=M[2,3,-2]*A[1,2]*UR[-1,3,1];
    mps_set_new[cx]=A_new;

    for cx=2:Lx-1
        A=mps_set[cx];
        M=mpo_set[cx];
        UL=deepcopy(UR)';
        UL_set[cx]=UL;
        UR=unitary(fuse(space(M,3)*space(A,2)), space(M,3)*space(A,2));
        UR_set[cx]=UR;
        

        @tensor A_new[:]:=M[2,1,4,-3]*A[3,5,1]*UL[2,3,-1]*UR[-2,4,5];
        mps_set_new[cx]=A_new;
    end

    cx=Lx;
    A=mps_set[cx];
    M=mpo_set[cx];
    UL=deepcopy(UR)';
    UL_set[cx]=UL;
    @tensor A_new[:]:=M[2,1,-2]*A[3,1]*UL[2,3,-1];
    mps_set_new[cx]=A_new;

    return mps_set_new,UR_set,UL_set
end