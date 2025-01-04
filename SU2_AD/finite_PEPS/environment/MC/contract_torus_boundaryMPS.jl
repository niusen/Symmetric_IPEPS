
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