


function combine_two_rows_method2(mpo1,mpo2,chi)
    function build_projector(Tup_L,Tup_R,Tdn_L,Tdn_R,chi)

        function right_decompose(Tup_R,Tdn_R)
            Tup_R=permute(Tup_R,(1,2,3,4,));
            Tdn_R=permute(Tdn_R,(1,2,3,4,));
        
            @tensor TTTT[:]:=Tup_R[-1,5,2,1]*Tup_R'[-3,6,2,1]*Tdn_R[-2,4,3,5]*Tdn_R'[-4,4,3,6];
            TTTT=permute(TTTT,(1,2,),(3,4,));
            eu,ev= eigh(TTTT);
            
            #decomposition: V=pinv(sqrt(eu))*ev*TT
            t0=ev*sqrt(eu);#U*S
            t_recover=mypinv(sqrt(eu))*ev';
        
            #verification, should be commented later
            @assert norm(ev*eu*ev'-TTTT)/norm(TTTT)<1e-10;
            @tensor TT[:]:=Tup_R[-1,1,-4,-6]*Tdn_R[-2,-3,-5,1]; #expensive, can be avoided
            #U,S,V=TT
            @tensor V[:]:=t_recover[-1,1,2]*TT[1,2,-2,-3,-4,-5];#verify U is isometry
            V=permute(V,(1,),(2,3,4,5,));
            VVd=V*V';
            Id_=unitary(space(VVd,1),space(VVd,1));
            @assert norm(VVd-Id_)/norm(Id_)<1e-8;
        
            return t0,t_recover
        end
        
        function left_decompose(Tup_L,Tdn_L)
            Tup_L=permute(Tup_L,(1,2,3,4,));
            Tdn_L=permute(Tdn_L,(1,2,3,4,));
        
            @tensor TTTT[:]:=Tup_L[2,6,-3,1]*Tup_L'[2,5,-1,1]*Tdn_L[3,4,-4,6]*Tdn_L'[3,4,-2,5];
            TTTT=permute(TTTT,(1,2,),(3,4,));
            eu,ev= eigh(TTTT);
            
            #decomposition: U=TT*ev*pinv(sqrt(eu))
            t0=sqrt(eu)*ev';#S*V
            t_recover=ev*mypinv(sqrt(eu));
        
            #verification, should be commented later
            @assert norm(ev*eu*ev'-TTTT)/norm(TTTT)<1e-10;
            @tensor TT[:]:=Tup_L[-1,1,-4,-6]*Tdn_L[-2,-3,-5,1]; #expensive, can be avoided
            #U,S,V=TT
            @tensor U[:]:=TT[-1,-2,-3,1,2,-5]*t_recover[1,2,-4];#verify U is isometry
            U=permute(U,(1,2,3,5),(4,));
            UdU=U'*U;
            Id_=unitary(space(UdU,1),space(UdU,1));
            @assert norm(UdU-Id_)/norm(Id_)<1e-8;
        
            return t0,t_recover
        end
        
        
        t0r,tr_recover=right_decompose(Tup_R,Tdn_R);
        t0l,tl_recover=left_decompose(Tup_L,Tdn_L);
        # println(err)
        # println(eu)
        M=t0l*t0r;
        uM,sM,vM=tsvd(M; trunc=truncdim(chi))
        err=norm(uM*sM*vM-M)/norm(M);
    
        PL=sqrt(sM)*vM*tr_recover;
        PR=tl_recover*uM*sqrt(sM);
     
        return PR,PL,err
    end
    trun_err=[];
    L=length(mpo1);
    PL_set=Vector{TensorMap}(undef,L);
    PR_set=Vector{TensorMap}(undef,L);
    for cc=1:L
        PR,PL,err=build_projector(mpo1[mod1(cc-1,L)],mpo1[cc],mpo2[mod1(cc-1,L)],mpo2[cc],chi);#projector at left side
        trun_err=vcat(trun_err,err);
        PL_set[cc]=PL;
        PR_set[mod1(cc-1,L)]=PR;
    end
    mpo_new=Vector{TensorMap}(undef,L);
    for cc=1:L
        Tup=mpo1[cc];
        Tdn=mpo2[cc];
        PL=PL_set[cc];
        PR=PR_set[cc];
        @tensor Tnew[:]:=PL[-1,1,3]*Tup[1,2,4,-4]*Tdn[3,-2,5,2]*PR[4,5,-3]; #the cost of this step may be D^7 or chi^7
        mpo_new[cc]=Tnew;
    end
    return mpo_new,trun_err
end





function combine_two_rows_method1(mpo1,mpo2,chi)
    function build_projector(Tup,Tdn,chi)
        Tup=permute(Tup,(1,2,3,4,));
        Tdn=permute(Tdn,(1,2,3,4,));
        # @tensor TTTT[:]:=Tup[-1,1,-4,-6]*Tdn[-2,-3,-5,1];
        @tensor TTTT[:]:=Tup[-1,5,2,1]*Tup'[-3,6,2,1]*Tdn[-2,4,3,5]*Tdn'[-4,4,3,6];
        TTTT=permute(TTTT,(1,2,),(3,4,));
        uM,sM,vM = tsvd(TTTT; trunc=truncdim(chi))
        err=norm(uM*sM*vM-TTTT)/norm(TTTT);
        # println(err)
        # println(eu)
    
    
        PR=uM;
        PL=uM';
        return PR,PL,err
    end
    trun_err=[];
    L=length(mpo1);
    PL_set=Vector{TensorMap}(undef,L);
    PR_set=Vector{TensorMap}(undef,L);
    for cc=1:L
        PR,PL,err=build_projector(mpo1[cc],mpo2[cc],chi);#projector at left side
        trun_err=vcat(trun_err,err);
        PL_set[cc]=PL;
        PR_set[mod1(cc-1,L)]=PR;
    end
    mpo_new=Vector{TensorMap}(undef,L);
    for cc=1:L
        Tup=mpo1[cc];
        Tdn=mpo2[cc];
        PL=PL_set[cc];
        PR=PR_set[cc];
        @tensor Tnew[:]:=PL[-1,1,3]*Tup[1,2,4,-4]*Tdn[3,-2,5,2]*PR[4,5,-3]; #the cost of this step may be D^7 or chi^7
        mpo_new[cc]=Tnew;
    end
    return mpo_new,trun_err
end

function contract_rows(finite_PEPS,chi)
    global projector_method
    if projector_method=="1"
        combine_two_rows=combine_two_rows_method1
    elseif projector_method=="2"
        combine_two_rows=combine_two_rows_method2
    end
    Lx,Ly=size(finite_PEPS);
    finite_PEPS=deepcopy(finite_PEPS);
    if mod(Ly,2)==0
        Ly_new=Int(Ly/2);
    else
        Ly_new=Int((Ly+1)/2);
    end
    trun_err=[];

    PEPS_new=Matrix{TensorMap}(undef,Lx,Ly_new);
    for cy=1:2:Ly
        if cy<Ly
            mpo1=finite_PEPS[:,cy+1];
            mpo2=finite_PEPS[:,cy];
            mpo_new,err=combine_two_rows(mpo1,mpo2,chi);
            trun_err=vcat(trun_err,err);
            PEPS_new[:,Int((cy+1)/2)]=mpo_new;
        elseif cy==Ly #Ly odd
            PEPS_new[:,Ly_new]=finite_PEPS[:,cy];
        end
    end
    return PEPS_new,trun_err
end


function contract_whole_torus(finite_PEPS,chi)
    finite_PEPS=deepcopy(finite_PEPS);
    trun_err=[];

    for ct=1:100
        finite_PEPS,err=contract_rows(finite_PEPS,chi)
        trun_err=vcat(trun_err,err);
        Lx,Ly_=size(finite_PEPS);
        if Ly_==2
            break;
        end
    end

    Lx,Ly_=size(finite_PEPS);
    mps_final=Vector{TensorMap}(undef,Lx);
    for cc=1:Lx
        @tensor T[:]:=finite_PEPS[cc,2][-1,1,-3,2]*finite_PEPS[cc,1][-2,2,-4,1];
        mps_final[cc]=permute(T,(1,2,),(3,4,));
    end

    T=mps_final[1];
    for cc=2:Lx-1
        T=T*mps_final[cc];
    end
    ov=@tensor T[1,2,3,4]*mps_final[Lx][3,4,1,2]


    return ov,trun_err

end