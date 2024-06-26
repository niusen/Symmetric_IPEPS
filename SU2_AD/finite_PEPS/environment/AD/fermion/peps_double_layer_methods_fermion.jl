function construct_double_layer_swap_new(psi1::Matrix,Lx,Ly)
    psi1=deepcopy(psi1);
    @assert (Lx,Ly)==size(psi1);
    
    psi_double=Matrix{TensorMap}(undef,Lx,Ly);
    UL_set=Matrix{TensorMap}(undef,Lx,Ly);
    UD_set=Matrix{TensorMap}(undef,Lx,Ly);
    UR_set=Matrix{TensorMap}(undef,Lx,Ly);
    UU_set=Matrix{TensorMap}(undef,Lx,Ly);



    #psi_double=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            AA, U_L,U_D,U_R,U_U=build_double_layer_swap(psi1[cx,cy]',psi1[cx,cy]);
            psi_double[cx,cy]=AA;
            UL_set[cx,cy]=U_L;
            UD_set[cx,cy]=U_D;
            UR_set[cx,cy]=U_R;
            UU_set[cx,cy]=U_U;

        end
    end

    for cx=1:Lx
        for cy=1:Ly
            if (cx in 2:Lx-1)&&(cy in 2:Ly-1)
                continue
            else
                T=psi_double[cx,cy];
                Tnew=remove_trivial_boundary(T,cx,cy,Lx,Ly);
                psi_double=matrix_update(psi_double,cx,cy,Tnew);
            end
        end
    end

    return psi_double,UL_set,UD_set,UR_set,UU_set
end

function remove_trivial_boundary(T,cx,cy,Lx,Ly)
    #remove trivial boundary legs
    
    if (cx==1)&& (cy in 2:Ly-1) #left boundary
        Uni=@ignore_derivatives unitary(space(T,2),space(T,1)*space(T,2));
        @tensor Tnew[:]:=T[1,2,-2,-3]*Uni[-1,1,2];
    elseif (cx==Lx)&&(cy in 2:Ly-1)#right boundary
        Uni=@ignore_derivatives unitary(space(T,4),space(T,3)*space(T,4));
        @tensor Tnew[:]:=T[-1,-2,1,2]*Uni[-3,1,2];
    elseif (cy==1) && (cx in 2:Lx-1)#bot boundary
        Uni=@ignore_derivatives unitary(space(T,1),space(T,1)*space(T,2));
        @tensor Tnew[:]:=T[1,2,-2,-3]*Uni[-1,1,2];
    elseif (cy==Ly)&& (cx in 2:Lx-1)#top boundary
        Uni=@ignore_derivatives unitary(space(T,3),space(T,3)*space(T,4));
        @tensor Tnew[:]:=T[-1,-2,1,2]*Uni[-3,1,2];
    elseif (cx==1) && (cy==1)#left-bot
        Uni=@ignore_derivatives unitary(space(T,3),space(T,1)*space(T,2)*space(T,3));
        @tensor Tnew[:]:=T[1,2,3,-2]*Uni[-1,1,2,3];
    elseif (cx==Lx)&& (cy==1)#right-bot
        Uni=@ignore_derivatives unitary(space(T,1),space(T,1)*space(T,2)*space(T,3));
        @tensor Tnew[:]:=T[1,2,3,-2]*Uni[-1,1,2,3];
    elseif (cx==1)&& (cy==Ly)#left-top
        Uni=@ignore_derivatives unitary(space(T,2),space(T,1)*space(T,2)*space(T,4));
        @tensor Tnew[:]:=T[1,2,-2,3]*Uni[-1,1,2,3];
    elseif (cx==Lx) && (cy==Ly)#right-top
        Uni=@ignore_derivatives unitary(space(T,2),space(T,2)*space(T,3)*space(T,4));
        @tensor Tnew[:]:=T[-1,1,2,3,]*Uni[-2,1,2,3];
    elseif (cx in 2:Lx-1) && (cy in 2:Ly-1)#bulk
        return T
    else 
        error("unknown case")
    end
    return Tnew
end

function construct_double_layer_swap(psi1::Matrix,Lx,Ly)
    psi1=deepcopy(psi1);

    
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
            AA, U_L,U_D,U_R,U_U=build_double_layer_swap(psi1[cx,cy]',psi1[cx,cy]);
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
        @ignore_derivatives T=TensorMap(randn,V_trivial*space(psi_double[cx+1,cy],1)',V_trivial);
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
        @ignore_derivatives T=TensorMap(randn,space(psi_double[cx-1,cy],3)'*V_trivial,V_trivial);
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
        @ignore_derivatives T=TensorMap(randn,V_trivial*V_trivial',space(psi_double[cx,cy+1],2));
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
        @ignore_derivatives T=TensorMap(randn,V_trivial*space(psi_double[cx,cy-1],4)',V_trivial);
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
    @ignore_derivatives T=TensorMap(randn,V_trivial',V_trivial);
    mm=T.data.values[1];
    @assert length(mm)==1
    @ignore_derivatives mm[1]=1;
    @ignore_derivatives T.data.values[1]=mm;
    T=permute(T,(1,2,));
    psi_double=matrix_update(psi_double,cx,cy,T);

    #right-bot
    cx=Lx+2;
    cy=1;
    @ignore_derivatives T=TensorMap(randn,V_trivial,V_trivial);
    mm=T.data.values[1];
    @assert length(mm)==1
    @ignore_derivatives mm[1]=1;
    @ignore_derivatives T.data.values[1]=mm;
    T=permute(T,(1,2,));
    psi_double=matrix_update(psi_double,cx,cy,T);

    #left-top
    cx=1;
    cy=Ly+2;
    @ignore_derivatives T=TensorMap(randn,V_trivial,V_trivial);
    mm=T.data.values[1];
    @assert length(mm)==1
    @ignore_derivatives mm[1]=1;
    @ignore_derivatives T.data.values[1]=mm;
    T=permute(T,(1,2,));
    psi_double=matrix_update(psi_double,cx,cy,T);

    #right-top
    cx=Lx+2;
    cy=Ly+2;
    @ignore_derivatives T=TensorMap(randn,V_trivial,V_trivial');
    mm=T.data.values[1];
    @assert length(mm)==1
    @ignore_derivatives mm[1]=1;
    @ignore_derivatives T.data.values[1]=mm;
    T=permute(T,(1,2,));
    psi_double=matrix_update(psi_double,cx,cy,T);



    return psi_double,UL_set,UD_set,UR_set,UU_set
end



function construct_double_layer_swap_sites(psi1::Matrix,psi_double, Lx,Ly)
    #construct double layer tensors for sites, 
    #the boundary trivial envs already exist
    psi_double=deepcopy(psi_double);
    psi1=deepcopy(psi1);


    @assert (Lx,Ly)==size(psi1);
    

    #psi_double=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            AA, U_L,U_D,U_R,U_U=build_double_layer_swap(psi1[cx,cy]',psi1[cx,cy]);
            #AA,_=build_double_layer_bulk(psi1[cx,cy],psi1[cx,cy],[]);
            psi_double=matrix_update(psi_double,cx+1,cy+1,AA);


        end
    end

    return psi_double
end


function construct_double_layer_swap_sites_new(psi1::Matrix,psi_double, Lx,Ly)
    #construct double layer tensors for sites, 
    #the boundary trivial envs already exist
    psi_double=deepcopy(psi_double);
    psi1=deepcopy(psi1);


    @assert (Lx,Ly)==size(psi1);
    

    #psi_double=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            AA, U_L,U_D,U_R,U_U=build_double_layer_swap(psi1[cx,cy]',psi1[cx,cy]);
            #AA,_=build_double_layer_bulk(psi1[cx,cy],psi1[cx,cy],[]);

            if (cx in 2:Lx-1)&&(cy in 2:Ly-1)
                psi_double=matrix_update(psi_double,cx,cy,AA);
            else
                AA=remove_trivial_boundary(AA,cx,cy,Lx,Ly);
                psi_double=matrix_update(psi_double,cx,cy,AA);
            end

        end
    end




    return psi_double
end

function construct_double_layer_swap_position(psi1::Matrix,psi_double, ppx,ppy, Lx,Ly)
    #construct double layer tensors for sites, 
    #the boundary trivial envs already exist
    psi_double=deepcopy(psi_double);
    psi1=deepcopy(psi1);

    @assert (Lx,Ly)==size(psi1);
    
    cx=ppx;
    cy=ppy;

    AA, U_L,U_D,U_R,U_U=build_double_layer_swap(psi1[cx,cy]',psi1[cx,cy]);
    #AA,_=build_double_layer_bulk(psi1[cx,cy],psi1[cx,cy],[]);

    if (cx in 2:Lx-1)&&(cy in 2:Ly-1)
        psi_double=matrix_update(psi_double,cx,cy,AA);
    else
        AA=remove_trivial_boundary(AA,cx,cy,Lx,Ly);
        psi_double=matrix_update(psi_double,cx,cy,AA);
    end

    return psi_double
end
