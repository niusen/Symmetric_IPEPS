function construct_double_layer_swap(psi1::Matrix{Square_iPEPS},psi2::Matrix{Square_iPEPS},Lx,Ly)
    psi1=deepcopy(psi1);
    if psi2==nothing
        psi2=deepcopy(psi1)
    else
        psi2=deepcopy(psi2);
    end
    
    @assert (Lx,Ly)==size(psi1);
    
    psi_double=ones(Lx+2,Ly+2);
    psi_double=convert(Matrix{Any},psi_double);

    UL_set=ones(Lx+2,Ly+2);
    UL_set=convert(Matrix{Any},UL_set);

    UD_set=ones(Lx+2,Ly+2);
    UD_set=convert(Matrix{Any},UD_set);

    UR_set=ones(Lx+2,Ly+2);
    UR_set=convert(Matrix{Any},UR_set);

    UU_set=ones(Lx+2,Ly+2);
    UU_set=convert(Matrix{Any},UU_set);



    #psi_double=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            AA, U_L,U_D,U_R,U_U=build_double_layer_swap(psi1[cx,cy].T',psi2[cx,cy].T);
            psi_double=matrix_update(psi_double,cx+1,cy+1,AA);
            UL_set=matrix_update(UL_set,cx+1,cy+1,U_L);
            UD_set=matrix_update(UD_set,cx+1,cy+1,U_D);
            UR_set=matrix_update(UR_set,cx+1,cy+1,U_R);
            UU_set=matrix_update(UU_set,cx+1,cy+1,U_U);

        end
    end


    #add trivial boundary tensors
    V_trivial=Rep[SU₂](0=>1);

    #left boundary
    cx=1;
    for cy=1+1:Ly+1
        T=TensorMap(randn,V_trivial*space(psi_double[cx+1,cy],1)',V_trivial);
        mm=T.data.values[1];
        @assert length(mm)==1
        @ignore_derivatives mm[1]=1;
        @ignore_derivatives T.data.values[1]=mm;
        T=permute(T,(1,2,3,));
        psi_double=matrix_update(psi_double,cx,cy,T);
    end

    #right boundary
    cx=Lx+2;
    for cy=1+1:Ly+1
        T=TensorMap(randn,space(psi_double[cx-1,cy],3)'*V_trivial,V_trivial);
        mm=T.data.values[1];
        @assert length(mm)==1
        @ignore_derivatives mm[1]=1;
        @ignore_derivatives T.data.values[1]=mm;
        T=permute(T,(1,2,3,));
        psi_double=matrix_update(psi_double,cx,cy,T);
    end
    

    #bot boundary
    cy=1;
    for cx=1+1:Lx+1
        T=TensorMap(randn,V_trivial*V_trivial',space(psi_double[cx,cy+1],2));
        mm=T.data.values[1];
        @assert length(mm)==1
        @ignore_derivatives mm[1]=1;
        @ignore_derivatives T.data.values[1]=mm;
        T=permute(T,(1,2,3,));
        psi_double=matrix_update(psi_double,cx,cy,T);
    end

    #top boundary
    cy=Ly+2;
    for cx=1+1:Lx+1
        T=TensorMap(randn,V_trivial*space(psi_double[cx,cy-1],4)',V_trivial);
        mm=T.data.values[1];
        @assert length(mm)==1
        @ignore_derivatives mm[1]=1;
        @ignore_derivatives T.data.values[1]=mm;
        T=permute(T,(1,2,3,));
        psi_double=matrix_update(psi_double,cx,cy,T);
    end


    #left-bot
    cx=1;
    cy=1;
    T=TensorMap(randn,V_trivial',V_trivial);
    mm=T.data.values[1];
    @assert length(mm)==1
    @ignore_derivatives mm[1]=1;
    @ignore_derivatives T.data.values[1]=mm;
    T=permute(T,(1,2,));
    psi_double=matrix_update(psi_double,cx,cy,T);

    #right-bot
    cx=Lx+2;
    cy=1;
    T=TensorMap(randn,V_trivial,V_trivial);
    mm=T.data.values[1];
    @assert length(mm)==1
    @ignore_derivatives mm[1]=1;
    @ignore_derivatives T.data.values[1]=mm;
    T=permute(T,(1,2,));
    psi_double=matrix_update(psi_double,cx,cy,T);

    #left-top
    cx=1;
    cy=Ly+2;
    T=TensorMap(randn,V_trivial,V_trivial);
    mm=T.data.values[1];
    @assert length(mm)==1
    @ignore_derivatives mm[1]=1;
    @ignore_derivatives T.data.values[1]=mm;
    T=permute(T,(1,2,));
    psi_double=matrix_update(psi_double,cx,cy,T);

    #right-top
    cx=Lx+2;
    cy=Ly+2;
    T=TensorMap(randn,V_trivial,V_trivial');
    mm=T.data.values[1];
    @assert length(mm)==1
    @ignore_derivatives mm[1]=1;
    @ignore_derivatives T.data.values[1]=mm;
    T=permute(T,(1,2,));
    psi_double=matrix_update(psi_double,cx,cy,T);



    return psi_double,UL_set,UD_set,UR_set,UU_set
end



