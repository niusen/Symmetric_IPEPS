function sort_triangles(triangles::Vector{Tuple})
    ts=Matrix{Int64}(undef,length(triangles),2);
    for cc=1:length(triangles)
        ts[cc,1:2]=[triangles[cc][1] triangles[cc][2]];
    end

    #arrange from bot to top, then from left to right
    ys=ts[:,2];
    y_set=unique(ys);
    row_set=Vector{Matrix}(undef,length(y_set));
    for cc=1:length(y_set)
        pos=findall(x->x.==y_set[cc],ts[:,2]);
        row_keep=ts[pos,:];
        order=sortperm(row_keep[:,1]);
        row_set[cc]=row_keep[order,:];
    end


    return row_set
end


function contract_triangle_env(CTM, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD)
    #leading memory cost:
    #chi^2*D^4*d^4
    #D^6*d^6

    VLU=CTM["VLU"];
    VLD=CTM["VLD"];
    VRU=CTM["VRU"];
    VRD=CTM["VRD"];
    
    T1L=CTM["T1L"];T1L=permute(T1L,(1,3,2,));
    T1R=CTM["T1R"];T1R=permute(T1R,(1,3,2,));
    T3L=CTM["T3L"];T3L=permute(T3L,(2,3,1,));
    T3R=CTM["T3R"];T3R=permute(T3R,(2,3,1,));

    @tensor MM_LU[:]:=VLU[2,4,-1]*T1L[2,3,-3]*T_double_LU[4,5,3]*B_double_LU[-2,-4,5]; 
    @tensor MM_RU[:]:=T1R[-1,3,1]* VRU[1,4,-3]* T_double_RU[-2,5,3]*B_double_RU[-4,4,5];

    @tensor MM_LD[:]:=VLD[-1,3,2]*T_double_LD[3,5,-2]*B_double_LD[4,-4,5]*T3L[-3,4,2]; 
    @tensor MM_RD[:]:=VRD[-4,-3,1]*T3R[1,-2,-1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*B_double_RD[1,2,-2]; 


    @tensor LD_LU_RU[:]:=MM_LD[1,2,-1,-2]*MM_LU[1,2,3,4]*MM_RU[3,4,-3,-4];
    @tensor BigTriangle[:]:= LD_LU_RU[1,-1,2,-3]*MM_RD[1,-2,2]; # L M U = L D U
    return BigTriangle
end







function triangle_FullUpdate_coord(gate,B_set, T_set,CTM,x_range,y_range, D_max, trun_order, trun_tol, n_sweep)
    # """
    #          M1     R1
    #            \   /
    #             \ /....d1
    #              |                   T1 =  |M1, d1><D1, R1|=|M1, d1><|R1, D1   
    #              |D1

    #              |                B=|R2, D1><M3|
    #             / \

    #   M2\   /R2    M3\   /R3
    #      \ /....d2    \ /....d3
    #       |            |   
    #       |D2          |D3

    #       T2           T3

    # T2=|M2, d2><D2, R2|=|M2, d2><|R2, D2 
    # T3=|M3, d3><D3, R3|=|M3, d3><|R3, D3 
    # """

    # println(space(T_set[x_range[2],y_range[2]]));
    # println(space(T_set[x_range[1],y_range[1]]))
    # println(space(T_set[x_range[2],y_range[1]]))
    # println(space(B_set[x_range[2],y_range[1]]))
    # println(space(gate))

    B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op = split_3Tesnsors(T_set[x_range[2],y_range[2]], T_set[x_range[1],y_range[1]], T_set[x_range[2],y_range[1]], B_set[x_range[2],y_range[1]], gate);


    T_LU=B_set[x_range[1],y_range[2]];
    T_double_LU, _ = build_double_layer_swap_Tm(T_LU',T_LU, false);#L M U
    B_LU=T_set[x_range[1],y_range[2]];
    B_double_LU, _ = build_double_layer_swap_Bm(B_LU',B_LU, true);#D R M

    T_RU=B_set[x_range[2],y_range[2]];
    T_double_RU, _ = build_double_layer_swap_Tm(T_RU',T_RU, false);#L M U

    T_LD=B_set[x_range[1],y_range[1]];
    T_double_LD, _ = build_double_layer_swap_Tm(T_LD',T_LD, false);#L M U

    B_double_RU, _ = build_double_layer_swap_Bm(B1_res',B1_res,false);#D R M
    B_double_LD, _ = build_double_layer_swap_Bm(B2_res',B2_res,false);#D R M
    B_double_RD, _ = build_double_layer_swap_Bm(B3_res',B3_res,false);#D R M
    BigTriangle_double_Noswap, U_L,U_D,U_U = build_double_layer_Noswap_Tm(B1_B2_T_B3',B1_B2_T_B3_op, true);#L M U = L D U
    BigTriangle_double_env=contract_triangle_env(CTM, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD);#L M U = L D U

    @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env[1,2,3]*U_L[1,-1,-4]*U_D[2,-2,-5]*U_U[-3,-6,3]; # storage order: L', D', U',   L, D, U,  fermionic order: L',L,U',U,D,D'
    

        BigTriangle_double_env_expand=permute(BigTriangle_double_env_expand,(1,2,3,),(4,5,6,));# storage order: L', D', U',       L, D, U
        #the fowllowing swap gates are taken from  function "build_double_layer_swap_Tm"
        gate=swap_gate(BigTriangle_double_env_expand,1,3);#L'U'
        @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env_expand[1,-2,2,-4,-5,-6]*gate[-1,-3,1,2];
        gate=swap_gate(BigTriangle_double_env_expand,1,6);#L'U
        @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env_expand[1,-2,-3,-4,-5,2]*gate[-1,-6,1,2];
        gate=swap_gate(BigTriangle_double_env_expand,3,6);#U'U
        @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env_expand[-1,-2,1,-4,-5,2]*gate[-3,-6,1,2];
        gate=swap_gate(BigTriangle_double_env_expand,2,5);#D'D'
        @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env_expand[-1,1,-3,-4,2,-6]*gate[-2,-5,1,2];
        gate=parity_gate(BigTriangle_double_env_expand,2);#D'
        @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env_expand[-1,1,-3,-4,-5,-6]*gate[-2,1];
        gate=parity_gate(BigTriangle_double_env_expand,3);#U'
        @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env_expand[-1,-2,1,-4,-5,-6]*gate[-3,1];

        BigTriangle_double_env_expand=permute(BigTriangle_double_env_expand,(1,2,3,),(4,5,6,));

        #eu,ev=eigen(BigTriangle_double_env_expand);
        eu,ev=eigh(BigTriangle_double_env_expand);
        eu=check_positive(eu);
        #M=ev*eu*ev';
        # env_bot=ev';#new_ind,1,2,3
        # env_top=ev;# 1,2,3, new_ind
        env_bot=sqrt(eu)*ev';#new_ind,2,3,1
        env_top=ev*sqrt(eu);# 2,3,1, new_ind

    
    # @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env_expand[1,2,3,4,5,6]*U_L'[1,4,-1]*U_D'[2,5,-2]*U_U'[-3,3,6];
    # ov1=@tensor BigTriangle_double_env_expand[1,2,3]*BigTriangle_double_Noswap[1,2,3];
    # ov2=ob_2x2(CTM_cell,AA_cell[c1][c2],AA_cell[mod1(c1+1,Lx)][c2],AA_cell[c1][mod1(c2+1,Ly)],AA_cell[mod1(c1+1,Lx)][mod1(c2+1,Ly)],mod1(c1-1,Lx),mod1(c2-1,Ly));
    # E=E+ov1/ov2;

    ############################################
    #direct truncation
    B_new,T1_new,T2_new,T3_new, big_T_compressed=truncation_direct(B1_B2_T_B3_op,D_max, trun_order, trun_tol)
    println("direct truncation:"*string(space(B_new)))

    # #test overlap without environment
    ov12=get_overlap_env(env_top,env_bot,big_T_compressed',B1_B2_T_B3_op);
    ov11=get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=get_overlap_env(env_top,env_bot,big_T_compressed',big_T_compressed);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap without optimization:"*string(norm(ov)))

    println("overlap with environmen after optimization:");
    B_new_a,T1_new_a,T2_new_a,T3_new_a,big_T_compressed_opt_a, ov_a=sweep_optimizations(n_sweep,B1_B2_T_B3_op,env_top,env_bot, B_new,T1_new,T2_new,T3_new)


    ####################################
    #truncation with env gauge
    B_new,T1_new,T2_new,T3_new, big_T_compressed=truncation_Env_gauge(env_top,env_bot, B1_B2_T_B3_op,D_max, trun_order, trun_tol)
    println("truncation with env gauge:"*string(space(B_new)))
    # #test overlap without environment
    ov12=get_overlap_env(env_top,env_bot,big_T_compressed',B1_B2_T_B3_op);
    ov11=get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=get_overlap_env(env_top,env_bot,big_T_compressed',big_T_compressed);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap without optimization:"*string(norm(ov)))
    
    # println([ov12,ov11,ov22])
    println("overlap with environmen after optimization:");
    B_new_b,T1_new_b,T2_new_b,T3_new_b,big_T_compressed_opt_b, ov_b=sweep_optimizations(n_sweep,B1_B2_T_B3_op,env_top,env_bot, B_new,T1_new,T2_new,T3_new)

    #########################################
    if ov_a>ov_b 
        println("direct truncation better")
        B_new_=B_new_a;
        T1_new_=T1_new_a;
        T2_new_=T2_new_a;
        T3_new_=T3_new_a;
        big_T_compressed_opt_=big_T_compressed_opt_a;
    else
        println("truncation with gauge better")
        B_new_=B_new_b;
        T1_new_=T1_new_b;
        T2_new_=T2_new_b;
        T3_new_=T3_new_b;
        big_T_compressed_opt_=big_T_compressed_opt_b;
    end
    # println(space(B_new_))
    # println(space(T1_new_))
    # println(space(T2_new_))
    # println(space(T3_new_))
    # println(space(big_T_compressed_opt_))


    #T1_new: (new1, d1),  (D1) 
    #T2_new: (new2, d2),  (R2) 
    #T3_new: (M3, d3), (new3)
    #B_new: (R2, D1), (M3)



    #T1=|M1, d1><D1, R1|=|M1, d1><|R1, D1 
    #T1_res:(M1), (R1, new1)
    @tensor T1_new_opt[:]:=B1_res[-1,-2,1]*T1_new_[1,-3,-4];#(M1)(R1, new1), (new1, d1)(D1) ->  (M1,R1, d1,D1)
    T1_new_opt=permute_neighbour_ind(T1_new_opt,2,3,4);#(M1,d1, R1,D1)
    T1_new_opt=permute(T1_new_opt,(1,),(2,3,4,));#(M1),(d1,R1,D1)

    #T2=|M2, d2><D2, R2|=|M2, d2><|R2, D2 
    #T2_res: (M2), (new2, D2)
    B2_res=permute_neighbour_ind(B2_res,2,3,3);#(M2)(D2,new2)
    @tensor T2_new_opt[:]:=B2_res[-1,-2,1]*T2_new_[1,-3,-4];#(M2)(D2,new2), (new2, d2)(R2)  ->  (M2,D2, d2,R2)
    T2_new_opt=permute_neighbour_ind(T2_new_opt,2,3,4);#(M2,d2, D2,R2)
    T2_new_opt=permute_neighbour_ind(T2_new_opt,3,4,4);#(M2,d2, R2,D2)
    T2_new_opt=permute(T2_new_opt,(1,),(2,3,4,));#(M2),(d2,R2,D2)

    #T3=|M3, d3><D3, R3|=|M3, d3><|R3, D3 
    #T3_res: (new3), (R3, D3)
    @tensor T3_new_opt[:]:=T3_new_[-1,-2,1]*B3_res[1,-3,-4];#(M3, d3)(new3),  (new3)(R3, D3)  ->  (M3,d3, R3,D3)
    T3_new_opt=permute(T3_new_opt,(1,),(2,3,4,));#(M3),(d3,R3,D3)



    #B=|R2, D1><M3|
    B_new_opt=B_new_;

    @assert (length(codomain(T1_new_opt))==1)&(length(domain(T1_new_opt))==3)
    @assert (length(codomain(T2_new_opt))==1)&(length(domain(T2_new_opt))==3)
    @assert (length(codomain(T3_new_opt))==1)&(length(domain(T3_new_opt))==3)
    @assert (length(codomain(B_new_opt))==2)&(length(domain(B_new_opt))==1)

    T1_new_opt=T1_new_opt/norm(T1_new_opt);
    T2_new_opt=T2_new_opt/norm(T2_new_opt);
    T3_new_opt=T3_new_opt/norm(T3_new_opt);
    B_new_opt=B_new_opt/norm(B_new_opt);
    

    T_set[x_range[2],y_range[2]]=T1_new_opt;
    T_set[x_range[1],y_range[1]]=T2_new_opt;
    T_set[x_range[2],y_range[1]]=T3_new_opt;
    B_set[x_range[2],y_range[1]]=B_new_opt;

    return B_set, T_set
end



function prepare_env_top_to_bot(psi_double,trivial_layer_yD,trivial_layer_yU)
    Lx,Ly=size(psi_double);
    global chi, multiplet_tol
    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end


    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
        end
        return mps_top
    end

    trun_history=[];
    mps_top_set=initial_tuple(Ly);
    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),Ly);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:3+trivial_layer_yD
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs,_=mpo_mps_fun(mpo, mps_top);
        mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
        trun_history=vcat(trun_history,trun_errs);
    end

    return mps_top_set
end

function prepare_env_trivial_bot(psi_double,trivial_layer_yD,trivial_layer_yU)
    Lx,Ly=size(psi_double);
    global chi, multiplet_tol
    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end

    trun_history=[];
    mps_bot_set=initial_tuple(Ly);

    mps_bot=(psi_double[:,1]...,);
    mps_bot_set=vector_update(mps_bot_set,mps_bot,1);
    # mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    # trun_history=vcat(trun_history,trun_errs);

    for cy=2:trivial_layer_yD
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs,_=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    return mps_bot_set
end


function env_bot_to_top_update(psi_double,mps_bot_set,y0,y1,  trivial_layer_yD,trivial_layer_yU)
    Lx,Ly=size(psi_double);
    global chi, multiplet_tol
    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end

    trun_history=[];
    mps_bot=mps_bot_set[y0+trivial_layer_yD-1];
    for cy=y0+trivial_layer_yD:y1+trivial_layer_yD
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs,_=mpo_mps_fun(mpo, mps_bot);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    return mps_bot_set
end


function prepare_env_from_right_to_left(psi_double,mps_top,mps_bot, y_range, trivial_layer_xL,trivial_layer_xR,trivial_layer_yD,trivial_layer_yU)
    Lx,Ly=size(psi_double);

    #construct left anf right environment

    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    # mps_top=mps_top_set[cy+2];
    mpo_top=(psi_double[:,y_range[2]+trivial_layer_yD]...,);
    mpo_bot=(psi_double[:,y_range[1]+trivial_layer_yD]...,);
    # mps_bot=mps_bot_set[cy-1];

    #left_right_env_method=="trun"
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
    vl_up,vl_dn=split_vl_or_vr(vl);
    VL_set=vector_update(VL_set,(vl_up,vl_dn,),1);
    for cx=2:trivial_layer_xL
        @tensor vl[:]:=vl_up[4,6,7]*vl_dn[7,2,1]*mps_top[cx][4,-1,5]*mpo_top[cx][6,8,-2,5]*mpo_bot[cx][2,3,-3,8]*mps_bot[cx][1,-4,3];
        vl_up,vl_dn=split_vl_or_vr(vl);
        VL_set=vector_update(VL_set,(vl_up,vl_dn,),cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
    vr_up,vr_dn=split_vl_or_vr(vr);
    VR_set=vector_update(VR_set,(vr_up,vr_dn,),Lx);
    for cx=Lx-1:-1:2+trivial_layer_xL
        @tensor vr[:]:=vr_up[1,3,8]*vr_dn[8,5,4]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,7,3,2]*mpo_bot[cx][-3,6,5,7]*mps_bot[cx][-4,4,6];
        vr_up,vr_dn=split_vl_or_vr(vr);
        VR_set=vector_update(VR_set,(vr_up,vr_dn,),cx);
    end
    
    return VL_set,VR_set

end


function env_left_to_right_update(psi_double,VL_set,mps_top,mps_bot,y_range, x0,x1,  trivial_layer_xL,trivial_layer_xR,trivial_layer_yD,trivial_layer_yU)
    Lx,Ly=size(psi_double);
    
    #construct left anf right environment

    
    # VL_set=initial_tuple(Lx);
    # VR_set=initial_tuple(Lx);
    # mps_top=mps_top_set[cy+2];
    mpo_top=(psi_double[:,y_range[2]+trivial_layer_yD]...,);
    mpo_bot=(psi_double[:,y_range[1]+trivial_layer_yD]...,);
    # mps_bot=mps_bot_set[cy-1];

    #left_right_env_method=="trun"

    for cx=x0+trivial_layer_xL:x1+trivial_layer_xL
        vl_up,vl_dn=VL_set[cx-1];
        @tensor vl[:]:=vl_up[4,6,7]*vl_dn[7,2,1]*mps_top[cx][4,-1,5]*mpo_top[cx][6,8,-2,5]*mpo_bot[cx][2,3,-3,8]*mps_bot[cx][1,-4,3];
        vl_up,vl_dn=split_vl_or_vr(vl);
        VL_set=vector_update(VL_set,(vl_up,vl_dn,),cx);
    end
    return VL_set
end

function triangle_full_update_PESS(all_triangles,Lx,Ly,D_max,B_set, T_set,psi,psi_double, gates_bulk,gates_left,gates_top,gates_left_top, trun_tol,n_sweep,trivial_layer_xL,trivial_layer_xR,trivial_layer_yD,trivial_layer_yU)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    #Here Bset,Tset are enlarged lattice 
    trun_order="simultaneous";

    
    for tran_set_id=1:length(all_triangles)
        trangle_set=all_triangles[tran_set_id];

        trangle_set_=sort_triangles(trangle_set);
        println("triangles:");
        println(trangle_set_);flush(stdout);

        mps_top_set=prepare_env_top_to_bot(psi_double,trivial_layer_yD,trivial_layer_yU);#contract from top to 4th row (3th row on the physical lattice)
        mps_bot_set=prepare_env_trivial_bot(psi_double,trivial_layer_yD,trivial_layer_yU);#contract from bot to 1st row (0th row on the physical lattice)
        coord_y_old=1;
        for nrow=1:size(trangle_set_,1)
            row_set=trangle_set_[nrow];
            coord_y=row_set[1,2];
            println("row coordinate: "*string(coord_y));
            y_range=[coord_y,coord_y+1];

            if coord_y-1>=coord_y_old
                mps_bot_set=env_bot_to_top_update(psi_double,mps_bot_set, coord_y_old,coord_y-1,  trivial_layer_yD,trivial_layer_yU);
            end
            coord_y_old=coord_y;



            mps_bot=mps_bot_set[coord_y-1+trivial_layer_yD];
            mps_top=mps_top_set[coord_y+2+trivial_layer_yD];
            #prepare_env from right to left
            VL_set,VR_set=prepare_env_from_right_to_left(psi_double,mps_top,mps_bot, y_range, trivial_layer_xL,trivial_layer_xR,trivial_layer_yD,trivial_layer_yU);

            coord_x_old=1;
            for ncolomn=1:size(row_set,1)

                trangle_coord=row_set[ncolomn,:];
                x_range=[trangle_coord[1],trangle_coord[1]+1];
                println("triangle coordinate: "*string(trangle_coord)*"->"*string([trangle_coord[1]+1,trangle_coord[2]])*"->"*string([trangle_coord[1]+1,trangle_coord[2]+1]));flush(stdout);

                coord_x=row_set[ncolomn,1];
                if coord_x-1>=coord_x_old
                    VL_set=env_left_to_right_update(psi_double,VL_set,mps_top,mps_bot,y_range, coord_x_old, coord_x-1,  trivial_layer_xL,trivial_layer_xR,trivial_layer_yD,trivial_layer_yU);
                end
                coord_x_old=coord_x;

                # B
                #CD
                posTB=[trangle_coord[1]+1,trangle_coord[2]+1];
                posTD=[trangle_coord[1]+1,trangle_coord[2]];
                posTC=[trangle_coord[1],trangle_coord[2]];
                posBond=posTD;



                if (trangle_coord[1]>0)&&(trangle_coord[2]<Ly) #bulk triangle
                    CTM=Dict{String,TensorMap}("VLU"=>VL_set[x_range[1]-1+trivial_layer_xL][1],"VLD"=>VL_set[x_range[1]-1+trivial_layer_xL][2],  "VRU"=>VR_set[x_range[2]+1+trivial_layer_xL][1],"VRD"=>VR_set[x_range[2]+1+trivial_layer_xL][2],   "T1L"=>mps_top[x_range[1]+trivial_layer_xL],"T1R"=>mps_top[x_range[2]+trivial_layer_xL],  "T3L"=>mps_bot[x_range[1]+trivial_layer_xL],"T3R"=>mps_bot[x_range[2]+trivial_layer_xL]  )
                    B_set, T_set=triangle_FullUpdate_coord(gates_bulk[mod1(trangle_coord[1],2)],B_set, T_set,CTM,x_range.+trivial_layer_xL,y_range.+trivial_layer_yD, D_max, trun_order, trun_tol, n_sweep)

                elseif (trangle_coord[1]==0)&&(trangle_coord[2]<Ly) #left triangle
                    CTM=Dict{String,TensorMap}("VLU"=>VL_set[x_range[1]-1+trivial_layer_xL][1],"VLD"=>VL_set[x_range[1]-1+trivial_layer_xL][2],  "VRU"=>VR_set[x_range[2]+1+trivial_layer_xL][1],"VRD"=>VR_set[x_range[2]+1+trivial_layer_xL][2],   "T1L"=>mps_top[x_range[1]+trivial_layer_xL],"T1R"=>mps_top[x_range[2]+trivial_layer_xL],  "T3L"=>mps_bot[x_range[1]+trivial_layer_xL],"T3R"=>mps_bot[x_range[2]+trivial_layer_xL]  )
                    B_set, T_set=triangle_FullUpdate_coord(gates_left[mod1(trangle_coord[1],2)],B_set, T_set,CTM,x_range.+trivial_layer_xL,y_range.+trivial_layer_yD, D_max, trun_order, trun_tol, n_sweep)

                elseif (trangle_coord[1]>0)&&(trangle_coord[2]==Ly) #top triangle
                    CTM=Dict{String,TensorMap}("VLU"=>VL_set[x_range[1]-1+trivial_layer_xL][1],"VLD"=>VL_set[x_range[1]-1+trivial_layer_xL][2],  "VRU"=>VR_set[x_range[2]+1+trivial_layer_xL][1],"VRD"=>VR_set[x_range[2]+1+trivial_layer_xL][2],   "T1L"=>mps_top[x_range[1]+trivial_layer_xL],"T1R"=>mps_top[x_range[2]+trivial_layer_xL],  "T3L"=>mps_bot[x_range[1]+trivial_layer_xL],"T3R"=>mps_bot[x_range[2]+trivial_layer_xL]  )
                    B_set, T_set=triangle_FullUpdate_coord(gates_top[mod1(trangle_coord[1],2)],B_set, T_set,CTM,x_range.+trivial_layer_xL,y_range.+trivial_layer_yD, D_max, trun_order, trun_tol, n_sweep)

                elseif (trangle_coord[1]==0)&&(trangle_coord[2]==Ly) #left-top corner
                    CTM=Dict{String,TensorMap}("VLU"=>VL_set[x_range[1]-1+trivial_layer_xL][1],"VLD"=>VL_set[x_range[1]-1+trivial_layer_xL][2],  "VRU"=>VR_set[x_range[2]+1+trivial_layer_xL][1],"VRD"=>VR_set[x_range[2]+1+trivial_layer_xL][2],   "T1L"=>mps_top[x_range[1]+trivial_layer_xL],"T1R"=>mps_top[x_range[2]+trivial_layer_xL],  "T3L"=>mps_bot[x_range[1]+trivial_layer_xL],"T3R"=>mps_bot[x_range[2]+trivial_layer_xL]  )
                    B_set, T_set=triangle_FullUpdate_coord(gates_left_top[mod1(trangle_coord[1],2)],B_set, T_set,CTM,x_range.+trivial_layer_xL,y_range.+trivial_layer_yD, D_max, trun_order, trun_tol, n_sweep)

                else
                    error("unknown case")
                end

                #update tensors of 3 triangle sites
                #update psi
                #update psi_double

                coord=[x_range[1]+trivial_layer_xL,  y_range[1]+trivial_layer_yD];
                A=iPESS_to_iPEPS_tensor(T_set[coord[1],coord[2]],B_set[coord[1],coord[2]]);
                psi[coord[1],coord[2]]=A;
                AA, _=build_double_layer_swap(A',A);
                psi_double[coord[1],coord[2]]=AA;

                coord=[x_range[1]+1+trivial_layer_xL,  y_range[1]+trivial_layer_yD];
                A=iPESS_to_iPEPS_tensor(T_set[coord[1],coord[2]],B_set[coord[1],coord[2]]);
                psi[coord[1],coord[2]]=A;
                AA, _=build_double_layer_swap(A',A);
                psi_double[coord[1],coord[2]]=AA;

                coord=[x_range[1]+1+trivial_layer_xL,  y_range[1]+1+trivial_layer_yD];
                A=iPESS_to_iPEPS_tensor(T_set[coord[1],coord[2]],B_set[coord[1],coord[2]]);
                psi[coord[1],coord[2]]=A;
                AA, _=build_double_layer_swap(A',A);
                psi_double[coord[1],coord[2]]=AA;

            end
        end
    end

    return B_set, T_set, psi,psi_double
end






function Full_update_PESS(parameters, Bset, Tset,  tau, dt, Dmax, trun_tol,n_sweep)
    tol=dt*1e-3;#for determining convergence 
    println("tau, dt="*string([tau,dt]))

    trivial_layer_xL=2;#trivial layer from left side
    trivial_layer_xR=1;#trivial layer from right side
    trivial_layer_yD=1;#trivial layer from bot side
    trivial_layer_yU=2;#trivial layer from top side


    Lx,Ly=size(Tset);
    Lx_large=Lx+trivial_layer_xL+trivial_layer_xR;
    Ly_large=Ly+trivial_layer_yD+trivial_layer_yU;


    #put tensors into a larger cluster with trivial boundaries
    Bset_large=Matrix{TensorMap}(undef,Lx_large,Ly_large);
    Tset_large=Matrix{TensorMap}(undef,Lx_large,Ly_large);
    for cx=1:Lx
        for cy=1:Ly
            Bset_large[cx+trivial_layer_xL,cy+trivial_layer_yD]=Bset[cx,cy];
            Tset_large[cx+trivial_layer_xL,cy+trivial_layer_yD]=Tset[cx,cy];
        end
    end

    Vphy=space(Tset[1,1],2);
    Vtrivial=Rep[SU₂](0=>1);
    for cx=1:Lx_large
        for cy=1:Ly_large
            if ~isassigned(Tset_large, cx,cy)
                T=TensorMap(randn,Vtrivial,Vphy'*Vtrivial*Vtrivial);
                T=T/norm(T);
                Tset_large[cx,cy]=T;
            end
            if ~isassigned(Bset_large, cx,cy)
                B=TensorMap(randn,Vtrivial*Vtrivial,Vtrivial);
                B=B/norm(B);
                Bset_large[cx,cy]=B;
            end
        end
    end



    ###################################
    psi=Matrix{TensorMap}(undef,Lx_large,Ly_large);
    psi_double=Matrix{TensorMap}(undef,Lx_large,Ly_large);
    for cx=1:Lx_large
        for cy=1:Ly_large
            psi[cx,cy]=iPESS_to_iPEPS_tensor(Tset_large[cx,cy],Bset_large[cx,cy]);
            # AA, _=build_double_layer_swap(psi[cx,cy]',psi[cx,cy]);
            # psi_double[cx,cy]=AA;
        end
    end
    psi_double,_=construct_double_layer_swap_new(psi,Lx_large,Ly_large);

    #verify trivial virtual index
    for cx=1:Lx_large-1
        for cy=1:Ly_large
            @assert space(psi[cx,cy],3)==space(psi[cx+1,cy],1)';
        end
    end 
    for cx=1:Lx_large
        for cy=1:Ly_large-1
            @assert space(psi[cx,cy+1],2)==space(psi[cx,cy],4)';
        end
    end 



    ###################################

    tx=parameters["t1"];
    ty=parameters["t1"];
    t2=parameters["t2"];
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_bulk=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=parameters["t1"];
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_top=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=parameters["t1"];
    t2=0;
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_left=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_left_top=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    all_triangles=get_triangles(Lx,Ly);
    println("all triangles:");
    println(all_triangles);

    for ct=1:Int(round(tau/abs(dt)))
        println("iteration "*string(ct));flush(stdout)
        Bset_large, Tset_large, psi,psi_double= triangle_full_update_PESS(all_triangles, Lx,Ly,Dmax, Bset_large, Tset_large, psi,psi_double, gates_ru_ld_rd_bulk,gates_ru_ld_rd_left,gates_ru_ld_rd_top,gates_ru_ld_rd_left_top, trun_tol, n_sweep, trivial_layer_xL,trivial_layer_xR,trivial_layer_yD,trivial_layer_yU);

        Bset=deepcopy(Bset_large[1+trivial_layer_xL:Lx+trivial_layer_xL,1+trivial_layer_yD:Ly+trivial_layer_yD]);
        Tset=deepcopy(Tset_large[1+trivial_layer_xL:Lx+trivial_layer_xL,1+trivial_layer_yD:Ly+trivial_layer_yD]);

        psi_=deepcopy(psi[1+trivial_layer_xL:Lx+trivial_layer_xL,1+trivial_layer_yD:Ly+trivial_layer_yD]);
        psi_double_=deepcopy(psi_double[1+trivial_layer_xL:Lx+trivial_layer_xL,1+trivial_layer_yD:Ly+trivial_layer_yD]);
        E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_global(psi_,psi_double_);
        println("energy terms:");
        println("E="*string(E_total));
        println(Ex_set);
        println(Ey_set);
        println(E_ld_ru_set);
        println(occu_set);
        println(EU_set);flush(stdout);

        E_total=real(E_total);

        global E_history
        if E_total<minimum(E_history)
            E_history=vcat(E_history,E_total);
            global save_filenm
            jldsave(save_filenm; B_set=Bset, T_set=Tset);
            global starting_time
            Now=now();
            Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
            println("Time consumed: "*string(Time));flush(stdout);
        end
    end



    return Bset, Tset
end



