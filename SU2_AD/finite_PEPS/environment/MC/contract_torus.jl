# function build_projector(Tup,Tdn,chi)
#     Tup=permute(Tup,(1,2,3,4,));
#     Tdn=permute(Tdn,(1,2,3,4,));
#     @tensor TTTT[:]:=Tup[-1,5,2,1]*Tup'[-3,6,2,1]*Tdn[-2,4,3,5]*Tdn'[-4,4,3,6];
#     TTTT=permute(TTTT,(1,2,),(3,4,));
#     eu,ev=eigh(TTTT);
#     @assert norm(ev*eu*ev'-TTTT)/norm(TTTT)<1e-12;
#     s_set=Vector{Array}(undef,length(eu.data.values));
#     u_set=Vector{Array}(undef,length(ev.data.values));
#     v_set=Vector{Array}(undef,length(ev.data.values));
#     for cc=1:length(eu.data.values)
#         s_set[cc]=eu.data.values[cc];
#         u_set[cc]=ev.data.values[cc];
#         v_set[cc]=(ev.data.values[cc])';
#     end
#     uM,sM,vM = truncate_block_svd(u_set,s_set,v_set,TTTT,chi);
#     err=norm(uM*sM*vM-TTTT)/norm(TTTT);
#     # println(err)
#     # println(eu)


#     PR=uM;
#     PL=vM;
#     return PR,PL,err
# end
function build_projector(Tup,Tdn,chi)
    Tup=permute(Tup,(1,2,3,4,));
    Tdn=permute(Tdn,(1,2,3,4,));
    @tensor TTTT[:]:=Tup[-1,1,-4,-6]*Tdn[-2,-3,-5,1];
    @tensor TTTT[:]:=Tup[-1,5,2,1]*Tup'[-3,6,2,1]*Tdn[-2,4,3,5]*Tdn'[-4,4,3,6];
    TTTT=permute(TTTT,(1,2,),(3,4,));
    uM,sM,vM = my_tsvd(TTTT; trunc=truncdim(chi))
    err=norm(uM*sM*vM-TTTT)/norm(TTTT);
    # println(err)
    # println(eu)


    PR=uM;
    PL=uM';
    return PR,PL,err
end
function combine_two_rows(mpo1,mpo2,chi)
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