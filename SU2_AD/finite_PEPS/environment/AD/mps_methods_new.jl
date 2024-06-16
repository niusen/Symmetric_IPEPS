


function truncate_right_to_left(mps_set,pos)
    global chi, multiplet_tol,use_AD;
    mps_set=deepcopy(mps_set);
    Lx=length(mps_set);
    @assert pos<Lx;
    trun_errs=[];

    cx=Lx;
    u,s,v=my_tsvd(permute(mps_set[cx],(1,),(2,)); trunc=truncdim(chi+20));

    trun_err=@ignore_derivatives 1-dot(s,s)/dot(T,T);
    trun_errs=vcat(trun_errs,trun_err);
    u=u*s;
    mps_set=mps_update(mps_set,permute(v,(1,2,)),cx);
    A=mps_set[cx-1];
    @tensor A[:]:=A[-1,1,-3]*u[1,-2];
    mps_set=mps_update(mps_set,A,cx-1);

    for cx=Lx-1:-1:pos
        u,s,v=my_tsvd(permute(mps_set[cx],(1,),(2,3,)); trunc=truncdim(chi+20));

        trun_err=@ignore_derivatives 1-dot(s,s)/dot(T,T);
        trun_errs=vcat(trun_errs,trun_err);
        u=u*s;
        mps_set=mps_update(mps_set,permute(v,(1,2,3,)),cx);
        A=mps_set[cx-1];
        if cx-1>1
            @tensor A[:]:=A[-1,1,-3]*u[1,-2];
            mps_set=mps_update(mps_set,permute(A,(1,2,3,)),cx-1);
        elseif cx-1==1
            @tensor A[:]:=A[1,-2]*u[1,-1];
            mps_set=mps_update(mps_set,permute(A,(1,2,)),cx-1);
        end
        
    end
    return mps_set,trun_errs
end

function truncate_left_to_right(mps_set,pos)
    @assert pos>1;
    global chi, multiplet_tol,use_AD;
    Lx=length(mps_set);
    trun_errs=[];

    cx=1;
    T=permute(mps_set[cx],(2,),(1,));
    u_trun,s_trun,v_trun=my_tsvd(T; trunc=truncdim(chi+20));

    trun_err=@ignore_derivatives 1-dot(s_trun,s_trun)/dot(T,T);
    v_trun=s_trun*v_trun;
    mps_set=mps_update(mps_set,permute(u_trun,(2,1,)),cx);
    A=mps_set[cx+1];
    @tensor A[:]:=v_trun[-1,1]*A[1,-2,-3];
    mps_set=mps_update(mps_set,permute(A,(1,2,3,)),cx+1);
    trun_errs=vcat(trun_errs,trun_err);

    for cx=2:pos
        #println(cx)
        T=permute(mps_set[cx],(1,3,),(2,));
        u_trun,s_trun,v_trun=my_tsvd(T; trunc=truncdim(chi+20)); 

        trun_err=@ignore_derivatives 1-dot(s_trun,s_trun)/dot(T,T);
        v_trun=s_trun*v_trun;
        mps_set=mps_update(mps_set,permute(u_trun,(1,3,2,)),cx);
        A=mps_set[cx+1];
        if cx+1<Lx
            @tensor A[:]:=v_trun[-1,1]*A[1,-2,-3];
            mps_set=mps_update(mps_set,permute(A,(1,2,3,)),cx+1);
            trun_errs=vcat(trun_errs,trun_err);
        elseif cx+1==Lx
            @tensor A[:]:=v_trun[-1,1]*A[1,-2];
            mps_set=mps_update(mps_set,permute(A,(1,2,)),cx+1);
            trun_errs=vcat(trun_errs,trun_err);
        end
    end
    return mps_set,trun_errs
end




function simple_truncate_to_moddle(mpo_set, mps_set)
    #without canonical form, so should be less accurate
    Lx=length(mpo_set);
    mps_origin=deepcopy(mps_set);

    global chi, multiplet_tol,use_AD;
    mps_set=deepcopy(mps_set);
    trun_errs=[];

    ######################
    #from right to left
    cx=Lx;
    @tensor T_tem[:]:= mpo_set[cx][-1,1,-3]*mps_set[cx][-2,1]
    u,s,v=my_tsvd(permute(T_tem,(1,2,),(3,)); trunc=truncdim(chi+20));

    trun_err=@ignore_derivatives 1-dot(s,s)/dot(T_tem,T_tem);
    trun_errs=vcat(trun_errs,trun_err);
    u=u*s;
    mps_set=mps_update(mps_set,permute(v,(1,2,)),cx);
    @tensor A[:]:=mpo_set[cx-1][-1,2,3,-4]*mps_set[cx-1][-2,1,2]*u[3,1,-3];
    mps_set=mps_update(mps_set,A,cx-1);
    for cx=Lx-1:-1:3
        u,s,v=my_tsvd(permute(mps_set[cx],(1,2,),(3,4,)); trunc=truncdim(chi+20));

        trun_err=@ignore_derivatives 1-dot(s,s)/dot(mps_set[cx],mps_set[cx]);
        trun_errs=vcat(trun_errs,trun_err);
        u=u*s;
        mps_set=mps_update(mps_set,permute(v,(1,2,3,)),cx);
        @tensor A[:]:=mpo_set[cx-1][-1,2,3,-4]*mps_set[cx-1][-2,1,2]*u[3,1,-3];
        mps_set=mps_update(mps_set,A,cx-1);
    end
    cx=2;
    u,s,v=my_tsvd(permute(mps_set[cx],(1,2,),(3,4,)); trunc=truncdim(chi+20));

    trun_err=@ignore_derivatives 1-dot(s,s)/dot(mps_set[cx],mps_set[cx]);
    trun_errs=vcat(trun_errs,trun_err);
    u=u*s;
    mps_set=mps_update(mps_set,permute(v,(1,2,3,)),cx);
    @tensor A[:]:=mpo_set[cx-1][2,3,-4]*mps_set[cx-1][1,2]*u[3,1,-3];
    mps_set=mps_update(mps_set,A,cx-1);


    ######################################
    #from left to right
    cx=1;
    u_trun,s_trun,v_trun=my_tsvd(permute(mps_set[cx],(2,),(1,)); trunc=truncdim(chi+20));

    trun_err=@ignore_derivatives 1-dot(s_trun,s_trun)/dot(mps_set[cx],mps_set[cx]);
    v_trun=s_trun*v_trun;
    mps_set=mps_update(mps_set,permute(u_trun,(2,1,)),cx);
    @tensor A[:]:=v_trun[-1,1]*mps_set[cx+1][1,-2,-3];
    mps_set=mps_update(mps_set,A,cx+1);
    trun_errs=vcat(trun_errs,trun_err);
    for cx=2:Lx-2
        T=permute(mps_set[cx],(1,3,),(2,));
        u_trun,s_trun,v_trun=my_tsvd(T; trunc=truncdim(chi+20)); 
   
        trun_err=@ignore_derivatives 1-dot(s_trun,s_trun)/dot(T,T);
        v_trun=s_trun*v_trun;
        mps_set=mps_update(mps_set,permute(u_trun,(1,3,2,)),cx);
        @tensor A[:]:=v_trun[-1,1]*mps_set[cx+1][1,-2,-3];
        mps_set=mps_update(mps_set,A,cx+1);
        trun_errs=vcat(trun_errs,trun_err);
    end
    cx=Lx-1;
    T=permute(mps_set[cx],(1,3,),(2,));
    u_trun,s_trun,v_trun=my_tsvd(T; trunc=truncdim(chi+20)); 
   
    trun_err=@ignore_derivatives 1-dot(s_trun,s_trun)/dot(T,T);
    v_trun=s_trun*v_trun;
    mps_set=mps_update(mps_set,permute(u_trun,(1,3,2,)),cx);
    @tensor A[:]:=v_trun[-1,1]*mps_set[cx+1][1,-2];
    mps_set=mps_update(mps_set,A,cx+1);
    trun_errs=vcat(trun_errs,trun_err);

    global n_mps_sweep
    # if n_mps_sweep>0
    #     println("sweep")
    # elseif n_mps_sweep==0
    #     println("no sweep")
    # end
    if n_mps_sweep>0
        mps_set=compress_sweep(mpo_set,mps_origin,mps_set);
    end


    #normalization
    A=mps_set[1];
    norm_A=norm(A);
    A=A/norm_A;
    mps_set=mps_update(mps_set,A,1);

    return mps_set,trun_errs,norm_A
end

function truncate_mpo_mps_new(mpo,mps_old)
    Lx=length(mpo);
    if mod(Lx,2)==0
        pos1=Lx/2+1;
        pos2=Lx/2;
    elseif mod(Lx,2)==1
        pos1=(Lx+1)/2+1;
        pos2=(Lx+1)/2;
    end
    mps_set,_=apply_mpo(mpo,mps_old);
    mps_set,trun_errs2=truncate_right_to_left(mps_set,pos2);
    mps_set,trun_errs1=truncate_left_to_right(mps_set,pos1);
    
    #mps_set,trun_errs=simple_truncate_to_moddle(mpo_set, mps_old);
    #trun_errs=vcat(trun_errs1,trun_errs2);

    return mps_set,trun_errs
end

function show_dimension(mps_set)
    println("show dim:")
    for cc=1:length(mps_set)
        A=mps_set[cc];
        println([dim(space(A,1)), dim(space(A,2))])
    end
end



function split_vl_or_vr(Vl)
    global use_AD,chi;
    u,s,v=my_tsvd(permute(Vl,(1,2,),(3,4,)); trunc=truncdim(chi+20));

    vl_up=u*s;
    vl_dn=v;
    return vl_up,vl_dn
end






function canonical_right_to_left(mps_set,pos)
    mps_set=deepcopy(mps_set);
    Lx=length(mps_set);
    @assert pos>1;
    cx=pos;
    if pos==Lx
        u,s,v=my_tsvd(permute(mps_set[cx],(1,),(2,)));
        u=u*s;
        mps_set=mps_update(mps_set,permute(v,(1,2,)),cx);
        A=mps_set[cx-1];
        @tensor A[:]:=A[-1,1,-3]*u[1,-2];
        mps_set=mps_update(mps_set,A,cx-1);
    elseif 2<pos<Lx
        u,s,v=my_tsvd(permute(mps_set[cx],(1,),(2,3,)));
        u=u*s;
        mps_set=mps_update(mps_set,permute(v,(1,2,3,)),cx);
        A=mps_set[cx-1];
        @tensor A[:]:=A[-1,1,-3]*u[1,-2];
        mps_set=mps_update(mps_set,permute(A,(1,2,3,)),cx-1);
    elseif pos==2
        u,s,v=my_tsvd(permute(mps_set[cx],(1,),(2,3,)));
        u=u*s;
        mps_set=mps_update(mps_set,permute(v,(1,2,3,)),cx);
        A=mps_set[cx-1];
        @tensor A[:]:=A[1,-2]*u[1,-1];
        mps_set=mps_update(mps_set,permute(A,(1,2,)),cx-1);
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
        T=permute(mps_set[cx],(2,),(1,));
        u,s,v=my_tsvd(T);
        v=s*v;
        mps_set=mps_update(mps_set,permute(u,(2,1,)),cx);
        A=mps_set[cx+1];
        @tensor A[:]:=v[-1,1]*A[1,-2,-3];
        mps_set=mps_update(mps_set,permute(A,(1,2,3,)),cx+1);
    elseif 1<pos<Lx-1
        T=permute(mps_set[cx],(1,3,),(2,));
        u,s,v=my_tsvd(T); 
        v=s*v;
        mps_set=mps_update(mps_set,permute(u,(1,3,2,)),cx);
        A=mps_set[cx+1];
        @tensor A[:]:=v[-1,1]*A[1,-2,-3];
        mps_set=mps_update(mps_set,permute(A,(1,2,3,)),cx+1);
    elseif pos==Lx-1
        T=permute(mps_set[cx],(1,3,),(2,));
        u,s,v=my_tsvd(T); 
        v=s*v;
        mps_set=mps_update(mps_set,permute(u,(1,3,2,)),cx);
        A=mps_set[cx+1];
        @tensor A[:]:=v[-1,1]*A[1,-2];
        mps_set=mps_update(mps_set,permute(A,(1,2,)),cx+1);
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

function single_sweep(mpo,mps0,mpstrun)
    Lx=length(mpo);
    Cright_set=Vector{TensorMap}(undef,Lx);
    Cleft_set=Vector{TensorMap}(undef,Lx);
    cp=Lx;
    @tensor Cright[:]:=mpstrun[cp]'[-1,1]*mpo[cp][-2,2,1]*mps0[cp][-3,2];
    Cright_set[cp]=Cright;
    for cp=Lx-1:-1:2
        Cright=sweep_left_move(Cright,mpo,mps0,mpstrun,cp);
        Cright_set[cp]=Cright;
    end

    #################################
    #sweep from left to right
    cp=1;
    @tensor Tnew[:]:=mpo[cp][1,2,-2]*mps0[cp][3,1]*Cright_set[cp+1][-1,2,3];
    mpstrun=mps_update(mpstrun,Tnew,cp);#update mps
    mpstrun=canonical_left_to_right(mpstrun,cp);#shift canonical center
    @tensor Cleft[:]:=mpstrun[cp]'[-1,1]*mpo[cp][2,-2,1]*mps0[cp][-3,2];
    Cleft_set[cp]=Cleft;#update environment
    for cp=2:Lx-1
        @tensor Tnew[:]:=Cleft_set[cp-1][-1,2,1]*mpo[cp][2,3,4,-3]*mps0[cp][1,5,3]*Cright_set[cp+1][-2,4,5];
        mpstrun=mps_update(mpstrun,Tnew,cp);#update mps
        mpstrun=canonical_left_to_right(mpstrun,cp);#shift canonical center
        Cleft=sweep_right_move(Cleft,mpo,mps0,mpstrun,cp);
        Cleft_set[cp]=Cleft;#update environment
    end

    #################################
    #sweep from right to left
    cp=Lx;
    @tensor Tnew[:]:=Cleft_set[cp-1][-1,2,3]*mpo[cp][2,1,-2]*mps0[cp][3,1];
    mpstrun=mps_update(mpstrun,Tnew,cp);#update mps
    mpstrun=canonical_right_to_left(mpstrun,cp);#shift canonical center
    @tensor Cright[:]:=mpstrun[cp]'[-1,1]*mpo[cp][-2,2,1]*mps0[cp][-3,2];
    Cright_set[cp]=Cright;#update environment
    for cp=Lx-1:-1:2
        @tensor Tnew[:]:=Cleft_set[cp-1][-1,2,1]*mpo[cp][2,3,4,-3]*mps0[cp][1,5,3]*Cright_set[cp+1][-2,4,5];
        mpstrun=mps_update(mpstrun,Tnew,cp);#update mps
        mpstrun=canonical_right_to_left(mpstrun,cp);#shift canonical center
        Cright=sweep_left_move(Cright,mpo,mps0,mpstrun,cp);
        Cright_set[cp]=Cright;#update environment
    end



    for css=1:n_mps_sweep-1
        #################################
        #sweep from left to right
        cp=1;
        @tensor Tnew[:]:=mpo[cp][1,2,-2]*mps0[cp][3,1]*Cright_set[cp+1][-1,2,3];
        mpstrun=mps_update(mpstrun,Tnew,cp);#update mps
        mpstrun=canonical_left_to_right(mpstrun,cp);#shift canonical center
        @tensor Cleft[:]:=mpstrun[cp]'[-1,1]*mpo[cp][2,-2,1]*mps0[cp][-3,2];
        Cleft_set[cp]=Cleft;#update environment
        for cp=2:Lx-1
            @tensor Tnew[:]:=Cleft_set[cp-1][-1,2,1]*mpo[cp][2,3,4,-3]*mps0[cp][1,5,3]*Cright_set[cp+1][-2,4,5];
            mpstrun=mps_update(mpstrun,Tnew,cp);#update mps
            mpstrun=canonical_left_to_right(mpstrun,cp);#shift canonical center
            Cleft=sweep_right_move(Cleft,mpo,mps0,mpstrun,cp);
            Cleft_set[cp]=Cleft;#update environment
        end

        #################################
        #sweep from right to left
        cp=Lx;
        @tensor Tnew[:]:=Cleft_set[cp-1][-1,2,3]*mpo[cp][2,1,-2]*mps0[cp][3,1];
        mpstrun=mps_update(mpstrun,Tnew,cp);#update mps
        mpstrun=canonical_right_to_left(mpstrun,cp);#shift canonical center
        @tensor Cright[:]:=mpstrun[cp]'[-1,1]*mpo[cp][-2,2,1]*mps0[cp][-3,2];
        Cright_set[cp]=Cright;#update environment
        for cp=Lx-1:-1:2
            @tensor Tnew[:]:=Cleft_set[cp-1][-1,2,1]*mpo[cp][2,3,4,-3]*mps0[cp][1,5,3]*Cright_set[cp+1][-2,4,5];
            mpstrun=mps_update(mpstrun,Tnew,cp);#update mps
            mpstrun=canonical_right_to_left(mpstrun,cp);#shift canonical center
            Cright=sweep_left_move(Cright,mpo,mps0,mpstrun,cp);
            Cright_set[cp]=Cright;#update environment
        end
    end


    return mpstrun
end

function compress_sweep(mpo,mps0,mps_trun)
    mpstrun=deepcopy(mps_trun);
    Lx=length(mps_trun);
    for cc=Lx:-1:2
        mpstrun=canonical_right_to_left(mpstrun,cc);
        #println(norm_1D(mpstrun,mps_trun)/sqrt(norm_1D(mps_trun,mps_trun)*norm_1D(mpstrun,mpstrun)))
    end

    mpstrun=single_sweep(mpo,mps0,mpstrun);

    # for cc=1:Lx-1
    #     mpstrun=canonical_left_to_right(mpstrun,cc);
    #     println(norm_1D(mpstrun,mps_trun)/sqrt(norm_1D(mps_trun,mps_trun)*norm_1D(mpstrun,mpstrun)))
    # end

    return mpstrun
end