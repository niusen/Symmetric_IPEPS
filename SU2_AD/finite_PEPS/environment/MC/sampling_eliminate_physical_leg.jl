function split_pleg_to_right(T::TensorMap)
    @assert Rank(T)==5;
    u,s,v=tsvd(permute(T,(1,2,4,),(3,5)));
    u=u*s;
    u=permute(u,(1,2,4,3,));#four leg tensor
    #v:(L,R,d)
    return u,v
end
function absorb_left_physical(Tp::TensorMap,T::TensorMap)
    @tensor T[:]:=Tp[-1,1,-5]*T[1,-2,-3,-4,-6];
    U=unitary(fuse(space(T,5)*space(T,6)), space(T,5)*space(T,6));
    @tensor T[:]:=T[-1,-2,-3,-4,1,2]*U[-5,1,2];
    return T
end

function transform_row(mpo_set::Vector{TensorMap})
    #mpo_set=deepcopy(mpo_set);
    L=length(mpo_set);

    cc=1;
    Tnew,Tp=split_pleg_to_right(mpo_set[cc]);
    mpo_set[cc]=Tnew;

    for cc=2:L-1
        Tcombined=absorb_left_physical(Tp,mpo_set[cc]);
        Tnew,Tp=split_pleg_to_right(Tcombined);
        mpo_set[cc]=Tnew;
    end

    cc=L;
    Tcombined=absorb_left_physical(Tp,mpo_set[cc]);
    mpo_set[cc]=Tcombined;

    return mpo_set;
end



function split_pleg_to_top(T::TensorMap)
    @assert Rank(T)==5;
    u,s,v=tsvd(permute(T,(1,2,3,),(4,5)));
    u=u*s;
    u=permute(u,(1,2,3,4,));#four leg tensor
    #v:(D,U,d)
    return u,v
end
function absorb_bot_physical(Tp::TensorMap,T::TensorMap)
    @tensor T[:]:=Tp[-2,1,-5]*T[-1,1,-3,-4,-6];
    U=unitary(fuse(space(T,5)*space(T,6)), space(T,5)*space(T,6));
    @tensor T[:]:=T[-1,-2,-3,-4,1,2]*U[-5,1,2];
    return T
end

function transform_column(mpo_set::Vector{TensorMap})
    #mpo_set=deepcopy(mpo_set);
    L=length(mpo_set);

    cc=1;
    Tnew,Tp=split_pleg_to_top(mpo_set[cc]);
    mpo_set[cc]=Tnew;

    for cc=2:L-1
        Tcombined=absorb_bot_physical(Tp,mpo_set[cc]);
        Tnew,Tp=split_pleg_to_top(Tcombined);
        mpo_set[cc]=Tnew;
    end

    cc=L;
    Tcombined=absorb_bot_physical(Tp,mpo_set[cc]);
    mpo_set[cc]=Tcombined;

    return mpo_set;
end


function shift_pleg(fPEPS::Matrix{TensorMap})
    #combine all sampled d=1 physical leg to right-top corner;
    # fPEPS=deepcopy(fPEPS);
    Lx,Ly=size(fPEPS);
    for cy=1:Ly
        fPEPS[:,cy]=transform_row(fPEPS[:,cy]);
    end

    fPEPS[Lx,:]=transform_column(fPEPS[Lx,:]);

    T=fPEPS[Lx,Ly];
    @assert space(T,5)==Rep[U₁](0=>1);

    U=unitary(space(T,4),space(T,4)*space(T,5));
    @tensor T[:]:=T[-1,-2,-3,1,2]*U[-4,1,2];
    fPEPS[Lx,Ly]=T;

    return fPEPS

end



##########################

function add_trivial_physical_leg(psi_network::Matrix{TensorMap})
    psi_network=deepcopy(psi_network);
    Lx,Ly=size(psi_network);
    #add trivial physical leg
    if isa(space(psi_network[1],1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Vtrivial=Rep[SU₂](0=>1);
    elseif isa(space(psi_network[1],1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Vtrivial=Rep[U₁](0=>1);
    end
    for cx=1:Lx
        for cy=1:Ly
            T=psi_network[cx,cy];
            if Rank(T)==2
                U=unitary(space(T,2)*Vtrivial,space(T,2));
                @tensor T[:]:=T[-1,1]*U[-2,-3,1];
            elseif Rank(T)==3
                U=unitary(space(T,3)*Vtrivial,space(T,3));
                @tensor T[:]:=T[-1,-2,1]*U[-3,-4,1];
            elseif Rank(T)==4
                U=unitary(space(T,4)*Vtrivial,space(T,4));
                @tensor T[:]:=T[-1,-2,-3,1]*U[-4,-5,1];
            end
            psi_network[cx,cy]=T;
        end
    end
    return psi_network
end
function remove_trivial_physical_leg(psi_network::Matrix{TensorMap})
    psi_network=deepcopy(psi_network);
    #remove trivial physical leg
    if isa(space(psi_network[1],1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        Vtrivial=Rep[SU₂](0=>1);
    elseif isa(space(psi_network[1],1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        Vtrivial=Rep[U₁](0=>1);
    end
    for cx=1:Lx
        for cy=1:Ly
            T=psi_network[cx,cy];
            if Rank(T)==3
                U=unitary(space(T,2),space(T,2)*Vtrivial);
                @tensor T[:]:=T[-1,1,2]*U[-2,1,2];
            elseif Rank(T)==4
                U=unitary(space(T,3),space(T,3)*Vtrivial);
                @tensor T[:]:=T[-1,-2,1,2]*U[-3,1,2];
            elseif Rank(T)==5
                U=unitary(space(T,4),space(T,4)*Vtrivial);
                @tensor T[:]:=T[-1,-2,-3,1,2]*U[-4,1,2];
            end
            psi_network[cx,cy]=T;
        end
    end
    return psi_network
end