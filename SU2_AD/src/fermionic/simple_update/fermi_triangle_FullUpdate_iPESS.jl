using LinearAlgebra:diag,I,diagm 
###########################
"""
(1,1), (2,1), (3,1)
(1,2), (2,2), (2,3)
(1,3), (2,3), (3,3)
"""
###########################

function check_positive(T)
    T_dense=convert(Array,T);
    T_new=deepcopy(T);
    @assert (norm(diag(T_dense))-norm(T_dense))/norm(T_dense)<1e-14;#verify diagonal


    #change negative eigenvalue to zero
    if sectortype(space(T,1)) == Trivial
        mm=T.data;
        for cc=1:size(mm,1)
            if real(mm[cc,cc])<0
                mm[cc,cc]=0;
            end
        end
        T_new=TensorMap(mm,codomain(eu),domain(eu));
    else
        for cc=1:length(T.data.values)
            mm=T.data.values[cc];
            for dd=1:size(mm,1)
                if real(mm[dd,dd])<0
                    mm[dd,dd]=0;
                end
            end
            T_new.data.values[cc]=mm;
        end
    end

    @assert norm(T-T_new)/norm(T_new)<0.01;
    return T_new
end


function test_decomposition1(B_set, T_set,AA_cell,Lx,Ly)
    global Lx,Ly
    for c1=1:Lx
        for c2=1:Ly
            Tm=B_set[c1,c2];
            Tm_double, U_L,U_D,U_U = build_double_layer_swap_Tm(Tm',Tm, false);#L M U

            Bm=T_set[c1,c2];
            Bm_double, U_D,U_R,U_U = build_double_layer_swap_Bm(Bm',Bm,true);#D R M
            @tensor AA_new[:]:=Tm_double[-1,1,-4]*Bm_double[-2,-3,1];

            @assert norm(AA_new-AA_cell[c1][c2])/norm(AA_cell[c1][c2])<1e-12
        end
    end
end


function test_decomposition2(B_set, T_set,AA_cell,CTM_cell,Lx,Ly)
    global Lx,Ly,parameters
    #decomposit 4x4 cluster from iPEPS representation to iPESS representation
    for c1=1:Lx
        for c2=1:Ly
            dt=0;
            gates_ru_ld_rd=gate_RU_LD_RD(parameters,dt, typeof(space(B_set[1],1)),Lx);

            B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op = split_3Tesnsors(T_set[mod1(c1+1,Lx),c2], T_set[c1,mod1(c2+1,Ly)], T_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], B_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], gates_ru_ld_rd[mod1(c1,2)]);


            T_LU=B_set[c1,c2];
            T_double_LU, U_L,U_D,U_U = build_double_layer_swap_Tm(T_LU',T_LU, false);#L M U
            B_LU=T_set[c1,c2];
            B_double_LU, U_D,U_R,U_U = build_double_layer_swap_Bm(B_LU',B_LU, true);#D R M

            T_RU=B_set[mod1(c1+1,Lx),c2];
            T_double_RU, U_L,U_D,U_U = build_double_layer_swap_Tm(T_RU',T_RU, false);#L M U

            T_LD=B_set[c1,mod1(c2+1,Ly)];
            T_double_LD, U_L,U_D,U_U = build_double_layer_swap_Tm(T_LD',T_LD, false);#L M U

            B_double_RU, U_D,U_R,U_U = build_double_layer_swap_Bm(B1_res',B1_res,false);#D R M
            B_double_LD, U_D,U_R,U_U = build_double_layer_swap_Bm(B2_res',B2_res,false);#D R M
            B_double_RD, U_D,U_R,U_U = build_double_layer_swap_Bm(B3_res',B3_res,false);#D R M
            BigTriangle_double, U_L,U_D,U_U = build_double_layer_swap_Tm(B1_B2_T_B3',B1_B2_T_B3_op, true);#L M U
            BigTriangle_double_env=contract_triangle_env(CTM_cell, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD, mod1(c1,Lx),mod1(c2,Ly));

            ov1=@tensor BigTriangle_double[1,2,3]*BigTriangle_double_env[1,2,3];
            ov2=ob_2x2(CTM_cell,AA_cell[c1][c2],AA_cell[mod1(c1+1,Lx)][c2],AA_cell[c1][mod1(c2+1,Ly)],AA_cell[mod1(c1+1,Lx)][mod1(c2+1,Ly)],mod1(c1-1,Lx),mod1(c2-1,Ly));
            # println(ov1/ov2)
            @assert abs(ov1/ov2-1)<1e-10;
        end
    end
end


function test_decomposition3(B_set, T_set,AA_cell,CTM_cell,Lx,Ly,E_correct)
    global Lx,Ly,parameters
    #decomposit 4x4 cluster from iPEPS representation to iPESS representation
    E=0;
    for c1=1:Lx
        for c2=1:Ly

            gates_ru_ld_rd=H_RU_LD_RD(parameters, typeof(space(B_set[1],1)),Lx);

            B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op = split_3Tesnsors(T_set[mod1(c1+1,Lx),c2], T_set[c1,mod1(c2+1,Ly)], T_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], B_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], gates_ru_ld_rd[mod1(c1,2)]);


            T_LU=B_set[c1,c2];
            T_double_LU, U_L,U_D,U_U = build_double_layer_swap_Tm(T_LU',T_LU, false);#L M U
            B_LU=T_set[c1,c2];
            B_double_LU, U_D,U_R,U_U = build_double_layer_swap_Bm(B_LU',B_LU, true);#D R M

            T_RU=B_set[mod1(c1+1,Lx),c2];
            T_double_RU, U_L,U_D,U_U = build_double_layer_swap_Tm(T_RU',T_RU, false);#L M U

            T_LD=B_set[c1,mod1(c2+1,Ly)];
            T_double_LD, U_L,U_D,U_U = build_double_layer_swap_Tm(T_LD',T_LD, false);#L M U

            B_double_RU, U_D,U_R,U_U = build_double_layer_swap_Bm(B1_res',B1_res,false);#D R M
            B_double_LD, U_D,U_R,U_U = build_double_layer_swap_Bm(B2_res',B2_res,false);#D R M
            B_double_RD, U_D,U_R,U_U = build_double_layer_swap_Bm(B3_res',B3_res,false);#D R M
            BigTriangle_double, U_L,U_D,U_U = build_double_layer_swap_Tm(B1_B2_T_B3',B1_B2_T_B3_op, true);#L M U
            BigTriangle_double_env=contract_triangle_env(CTM_cell, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD, mod1(c1,Lx),mod1(c2,Ly));

            ov1=@tensor BigTriangle_double[1,2,3]*BigTriangle_double_env[1,2,3];
            ov2=ob_2x2(CTM_cell,AA_cell[c1][c2],AA_cell[mod1(c1+1,Lx)][c2],AA_cell[c1][mod1(c2+1,Ly)],AA_cell[mod1(c1+1,Lx)][mod1(c2+1,Ly)],mod1(c1-1,Lx),mod1(c2-1,Ly));
            #println(ov1/ov2)
            E=E+ov1/ov2;

        end
    end
    E_new=E/Lx/Ly;
    println([E_new,E_correct])
    #@assert abs(E_new-E_correct)/abs(E_correct)<1e-10 #this error is not quite small since in two methods expectation values are computed with different environment clusters
end


function test_positive_triangle_env(B_set, T_set,AA_cell,CTM_cell,Lx,Ly,E_correct)
    #global c1,c2
    E=0;
    for c1=1:Lx
        for c2=1:Ly

            gates_ru_ld_rd=H_RU_LD_RD(parameters, typeof(space(B_set[1],1)),Lx);

            B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op = split_3Tesnsors(T_set[mod1(c1+1,Lx),c2], T_set[c1,mod1(c2+1,Ly)], T_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], B_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], gates_ru_ld_rd[mod1(c1,Lx)]);


            T_LU=B_set[c1,c2];
            T_double_LU, _ = build_double_layer_swap_Tm(T_LU',T_LU, false);#L M U
            B_LU=T_set[c1,c2];
            B_double_LU, _ = build_double_layer_swap_Bm(B_LU',B_LU, true);#D R M

            T_RU=B_set[mod1(c1+1,Lx),c2];
            T_double_RU, _ = build_double_layer_swap_Tm(T_RU',T_RU, false);#L M U

            T_LD=B_set[c1,mod1(c2+1,Ly)];
            T_double_LD, _ = build_double_layer_swap_Tm(T_LD',T_LD, false);#L M U

            B_double_RU, _ = build_double_layer_swap_Bm(B1_res',B1_res,false);#D R M
            B_double_LD, _ = build_double_layer_swap_Bm(B2_res',B2_res,false);#D R M
            B_double_RD, _ = build_double_layer_swap_Bm(B3_res',B3_res,false);#D R M
            BigTriangle_double_Noswap, U_L,U_D,U_U = build_double_layer_Noswap_Tm(B1_B2_T_B3',B1_B2_T_B3_op, true);#L M U = L D U
            BigTriangle_double_env=contract_triangle_env(CTM_cell, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD, mod1(c1,Lx),mod1(c2,Ly));#L M U = L D U

            @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env[1,2,3]*U_L[1,-1,-4]*U_D[2,-2,-5]*U_U[-3,-6,3]; # storage order: L', D', U',   L, D, U,  fermionic order: L',L,U',U,D,D'
            method=1
            if method==1 #this method is correct: when chi is large, all signs will be positive.
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
                @assert norm(BigTriangle_double_env_expand-BigTriangle_double_env_expand')/norm(BigTriangle_double_env_expand)<1e-8;
                BigTriangle_double_env_expand=BigTriangle_double_env_expand/2+BigTriangle_double_env_expand'/2;
                eu,ev=eigen(BigTriangle_double_env_expand);

            # elseif method==2 #this method is incorrect: although the eigenvalues are real, many signs are still minus even when chi is charge. Possible reason is that swap gates for double layer tensors are incomplete. 
            #     BigTriangle_double_env_expand=permute(BigTriangle_double_env_expand,(1,4,3,6,5,2,));#L',L,U',U,D,D'
            #     BigTriangle_double_env_expand=permute_neighbour_ind(BigTriangle_double_env_expand,5,6,6);#L',L,U',U,D',D
            #     BigTriangle_double_env_expand=permute_neighbour_ind(BigTriangle_double_env_expand,4,5,6);#L',L,U',D',U,D
            #     BigTriangle_double_env_expand=permute_neighbour_ind(BigTriangle_double_env_expand,3,4,6);#L',L,D',U',U,D
            #     BigTriangle_double_env_expand=permute_neighbour_ind(BigTriangle_double_env_expand,2,3,6);#L',D',L,U',U,D
            #     BigTriangle_double_env_expand=permute_neighbour_ind(BigTriangle_double_env_expand,1,2,6);#D',L',L,U',U,D
            #     BigTriangle_double_env_expand=permute_neighbour_ind(BigTriangle_double_env_expand,3,4,6);#D',L',U',L,U,D
            #     BigTriangle_double_env_expand=permute_neighbour_ind(BigTriangle_double_env_expand,2,3,6);#D',U',L',L,U,D
            #     BigTriangle_double_env_expand=permute(BigTriangle_double_env_expand,(1,2,3,),(6,5,4,));#storage order: D',U',L',D,U,L;  fermionic order: D',U',L',L,U,D

            #     eu,ev=eigh(BigTriangle_double_env_expand);


            end
            @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env_expand[1,2,3,4,5,6]*U_L'[1,4,-1]*U_D'[2,5,-2]*U_U'[-3,3,6];
            ov1=@tensor BigTriangle_double_env_expand[1,2,3]*BigTriangle_double_Noswap[1,2,3];
            ov2=ob_2x2(CTM_cell,AA_cell[c1][c2],AA_cell[mod1(c1+1,Lx)][c2],AA_cell[c1][mod1(c2+1,Ly)],AA_cell[mod1(c1+1,Lx)][mod1(c2+1,Ly)],mod1(c1-1,Lx),mod1(c2-1,Ly));
            #println([ov1,ov2])
            E=E+ov1/ov2;
        end
    end
    E_new=E/Lx/Ly;
    println([E_new,E_correct])
end


function test_positive_triangle_env2(B_set, T_set,AA_cell,CTM_cell,Lx,Ly,E_correct)
    #global c1,c2
    E=0;
    for c1=1:Lx
        for c2=1:Ly

            gates_ru_ld_rd=H_RU_LD_RD(parameters, typeof(space(B_set[1],1)),Lx);

    

            B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op = split_3Tesnsors(T_set[mod1(c1+1,Lx),c2], T_set[c1,mod1(c2+1,Ly)], T_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], B_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], gates_ru_ld_rd[mod1(c1,Lx)]);


            T_LU=B_set[c1,c2];
            T_double_LU, _ = build_double_layer_swap_Tm(T_LU',T_LU, false);#L M U
            B_LU=T_set[c1,c2];
            B_double_LU, _ = build_double_layer_swap_Bm(B_LU',B_LU, true);#D R M

            T_RU=B_set[mod1(c1+1,Lx),c2];
            T_double_RU, _ = build_double_layer_swap_Tm(T_RU',T_RU, false);#L M U

            T_LD=B_set[c1,mod1(c2+1,Ly)];
            T_double_LD, _ = build_double_layer_swap_Tm(T_LD',T_LD, false);#L M U

            B_double_RU, _ = build_double_layer_swap_Bm(B1_res',B1_res,false);#D R M
            B_double_LD, _ = build_double_layer_swap_Bm(B2_res',B2_res,false);#D R M
            B_double_RD, _ = build_double_layer_swap_Bm(B3_res',B3_res,false);#D R M
            BigTriangle_double_Noswap, U_L,U_D,U_U = build_double_layer_Noswap_Tm(B1_B2_T_B3',B1_B2_T_B3_op, true);#L M U = L D U
            BigTriangle_double_env=contract_triangle_env(CTM_cell, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD, mod1(c1,Lx),mod1(c2,Ly));#L M U = L D U

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
            #M=ev*eu*ev';
            env_bot=sqrt(eu)*ev';#new_ind,2,3,1
            env_top=ev*sqrt(eu);# 2,3,1, new_ind

            ov1=get_overlap_env(env_top,env_bot,B1_B2_T_B3',B1_B2_T_B3_op);
            ov2=get_overlap_env(env_top,env_bot,B1_B2_T_B3',B1_B2_T_B3);
            #println([ov1,ov2])
            E=E+ov1/ov2;

        end
    end
    E_new=E/Lx/Ly;
    println([E_new,E_correct])


end

function test_triangle_FullUpdate(dt,B_set, T_set,AA_cell,CTM_cell,Lx,Ly,coord, D_max, trun_order, trun_tol)
    (c1,c2)=coord;

    gates_ru_ld_rd=gate_RU_LD_RD(parameters,dt, typeof(space(B_set[1],1)),Lx);
    #gates_ru_ld_rd=H_RU_LD_RD(parameters, typeof(space(B_set[1],1)),Lx);
    gates_ru_ld_rd=gates_ru_ld_rd[mod1(c1,Lx)];
    

    B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op = split_3Tesnsors(T_set[mod1(c1+1,Lx),c2], T_set[c1,mod1(c2+1,Ly)], T_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], B_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], gates_ru_ld_rd);


    T_LU=B_set[c1,c2];
    T_double_LU, _ = build_double_layer_swap_Tm(T_LU',T_LU, false);#L M U
    B_LU=T_set[c1,c2];
    B_double_LU, _ = build_double_layer_swap_Bm(B_LU',B_LU, true);#D R M

    T_RU=B_set[mod1(c1+1,Lx),c2];
    T_double_RU, _ = build_double_layer_swap_Tm(T_RU',T_RU, false);#L M U

    T_LD=B_set[c1,mod1(c2+1,Ly)];
    T_double_LD, _ = build_double_layer_swap_Tm(T_LD',T_LD, false);#L M U

    B_double_RU, _ = build_double_layer_swap_Bm(B1_res',B1_res,false);#D R M
    B_double_LD, _ = build_double_layer_swap_Bm(B2_res',B2_res,false);#D R M
    B_double_RD, _ = build_double_layer_swap_Bm(B3_res',B3_res,false);#D R M
    BigTriangle_double_Noswap, U_L,U_D,U_U = build_double_layer_Noswap_Tm(B1_B2_T_B3',B1_B2_T_B3_op, true);#L M U = L D U
    BigTriangle_double_env=contract_triangle_env(CTM_cell, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD, mod1(c1,Lx),mod1(c2,Ly));#L M U = L D U

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

    #direct truncation
    println("direct truncation:")
    B_new,T1_new,T2_new,T3_new, big_T_compressed=truncation_direct(B1_B2_T_B3_op,D_max, trun_order, trun_tol)
    
    #test overlap without environment
    # println(space(big_T_compressed))
    # println(space(B1_B2_T_B3_op))
    ov=dot(big_T_compressed,B1_B2_T_B3_op)/sqrt(dot(B1_B2_T_B3_op,B1_B2_T_B3_op)*dot(big_T_compressed,big_T_compressed));
    println("overlap without environmen:"*string(ov))

    #test overlap without environment
    ov12=get_overlap_env(env_top,env_bot,big_T_compressed',B1_B2_T_B3_op);
    ov11=get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=get_overlap_env(env_top,env_bot,big_T_compressed',big_T_compressed);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap with environmen:"*string(norm(ov)))
    println(space(B_new))

    ####################################
    #truncation with env gauge
    println("truncation with env gauge:")
    B_new,T1_new,T2_new,T3_new, big_T_compressed=truncation_Env_gauge(env_top,env_bot, B1_B2_T_B3_op,D_max, trun_order, trun_tol)

    #test overlap without environment
    ov=dot(big_T_compressed,B1_B2_T_B3_op)/sqrt(dot(B1_B2_T_B3_op,B1_B2_T_B3_op)*dot(big_T_compressed,big_T_compressed));
    println("overlap without environmen:"*string(ov))

    #test overlap without environment
    ov12=get_overlap_env(env_top,env_bot,big_T_compressed',B1_B2_T_B3_op);
    ov11=get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=get_overlap_env(env_top,env_bot,big_T_compressed',big_T_compressed);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap with environment:"*string(norm(ov)))
    println(space(B_new))

    println([ov12,ov11,ov22])
    


    ####################################
    T1_left,T1_right,T1_opt=partial_triangle_partial_B1(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    #test overlap after optimization 
    big_T_compressed_opt=build_triangle_from_4tensors(B_new,T1_opt,T2_new,T3_new)
    ov12=get_overlap_env(env_top,env_bot,big_T_compressed_opt',B1_B2_T_B3_op);
    ov11=get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=get_overlap_env(env_top,env_bot,big_T_compressed_opt',big_T_compressed_opt);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap with environmen after optimization B1:"*string(norm(ov)))
    ####################################
    T2_left,T2_right,T2_opt=partial_triangle_partial_B2(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    #test overlap after optimization 
    big_T_compressed_opt=build_triangle_from_4tensors(B_new,T1_new,T2_opt,T3_new)
    ov12=get_overlap_env(env_top,env_bot,big_T_compressed_opt',B1_B2_T_B3_op);
    ov11=get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=get_overlap_env(env_top,env_bot,big_T_compressed_opt',big_T_compressed_opt);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap with environmen after optimization B2:"*string(norm(ov)))
    ####################################
    T3_left,T3_right,T3_opt=partial_triangle_partial_B3(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    #test overlap after optimization 
    big_T_compressed_opt=build_triangle_from_4tensors(B_new,T1_new,T2_new,T3_opt)
    ov12=get_overlap_env(env_top,env_bot,big_T_compressed_opt',B1_B2_T_B3_op);
    ov11=get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=get_overlap_env(env_top,env_bot,big_T_compressed_opt',big_T_compressed_opt);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap with environmen after optimization B3:"*string(norm(ov)))
    ####################################
    B_left,B_right,B_opt=partial_triangle_partial_T(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    #test overlap after optimization 
    big_T_compressed_opt=build_triangle_from_4tensors(B_opt,T1_new,T2_new,T3_new)
    ov12=get_overlap_env(env_top,env_bot,big_T_compressed_opt',B1_B2_T_B3_op);
    ov11=get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=get_overlap_env(env_top,env_bot,big_T_compressed_opt',big_T_compressed_opt);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap with environmen after optimization B3:"*string(norm(ov)))

end


function get_overlap_env(env_top,env_bot,triangle_top,triangle_bot)
    #env_bot:   (env_ind),  (2,3,1)
    #env_top:   (2,3,1), (env_ind)
    #triangle_bot: (2, 1), (d123, 3)
    #triangle_top: (d123, 3), (2, 1)
    @tensor Bot[:]:=env_bot[-1,2,3,1]*triangle_bot[2,1,-2,3];
    @tensor Top[:]:=env_top[2,3,1,-1]*triangle_top[-2,3,2,1];
    ov=@tensor Bot[1,2]*Top[1,2];
    return ov
end

function build_double_layer_swap_Tm(Ap,A, with_physical)
    if ~with_physical #no physical leg
        @assert (length(codomain(A))==2)&(length(domain(A))==1)
        @assert (length(codomain(Ap))==1)&(length(domain(Ap))==2)
        #Treat (LU,M) as (LU,D)
        #Treat (M',L'U') as (D',L'U')
        # println(space(Ap))
        # println(space(A))

        gate=@ignore_derivatives swap_gate(Ap,2,3); #gate L'U'
        @tensor Ap[:]:=Ap[-1,1,2]*gate[-2,-3,1,2];  

        gate=@ignore_derivatives parity_gate(Ap,1); #gate D'
        @tensor Ap[:]:=Ap[1,-2,-3]*gate[-1,1];
        gate=@ignore_derivatives parity_gate(Ap,3); #gate U'
        @tensor Ap[:]:=Ap[-1,-2,1]*gate[-3,1];

        
        A=permute(A,(1,2,),(3,));
        Ap=permute(Ap,(1,),(2,3,));
        

        U_L=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 1)), space(Ap, 2) ⊗ space(A, 1));
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 1) ⊗ space(A, 3)), space(Ap, 1) ⊗ space(A, 3));
        U_U=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 2)', fuse(space(Ap, 3)' ⊗ space(A, 2)'));

        @tensor AA_fused[:]:=Ap[5,1,3]*A[2,4,6]*U_L[-1,1,2]*U_D[-2,5,6]*U_U[3,4,-3];

    else #with 3 physical legs grouped as one leg
        @assert (length(codomain(A))==2)&(length(domain(A))==2)
        @assert (length(codomain(Ap))==2)&(length(domain(Ap))==2)
        #Treat (LU,dM) as (LU,dD)
        #Treat (d'M',L'U') as (d'D',L'U')
        # println(space(Ap))
        # println(space(A))

        gate=@ignore_derivatives swap_gate(Ap,3,4); #gate L'U'
        @tensor Ap[:]:=Ap[-1,-2,1,2]*gate[-3,-4,1,2];  

        gate=@ignore_derivatives parity_gate(Ap,2); #gate D'
        @tensor Ap[:]:=Ap[-1,1,-3,-4]*gate[-2,1];
        gate=@ignore_derivatives parity_gate(Ap,4); #gate U'
        @tensor Ap[:]:=Ap[-1,-2,-3,1]*gate[-4,1];

        
        A=permute(A,(1,2,),(3,4,));
        Ap=permute(Ap,(1,2,),(3,4,));
        

        U_L=@ignore_derivatives unitary(fuse(space(Ap, 3) ⊗ space(A, 1)), space(Ap, 3) ⊗ space(A, 1));
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 4)), space(Ap, 2) ⊗ space(A, 4));
        U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 2)', fuse(space(Ap, 4)' ⊗ space(A, 2)'));

        @tensor AA_fused[:]:=Ap[3,6,1,4]*A[2,5,3,7]*U_L[-1,1,2]*U_D[-2,6,7]*U_U[4,5,-3];
    end


    P_odd_Lp,_=@ignore_derivatives projector_parity(space(U_L',1));
    P_odd_Up,_=@ignore_derivatives projector_parity(space(U_U',2));
    P_odd_U,_=@ignore_derivatives projector_parity(space(U_U',3));

    @tensor isom_Lp[:]:=U_L[-1,4,3]*P_odd_Lp'[4,1]*P_odd_Lp[1,2]*U_L'[2,3,-2];
    @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2];
    @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6];
    @tensor AA_Lp_U[:]:=AA_fused[1,-2,4]*isom_Lp[-1,1]*isom_U[-3,4];
    AA_fused=AA_fused-2*AA_Lp_U;
    @tensor AA_Up_U[:]:=AA_fused[-1,-2,4]*isom_Up_U[-3,4];
    AA_fused=AA_fused-2*AA_Up_U;



    P_odd_Dp,_=@ignore_derivatives projector_parity(space(U_D',1));
    P_odd_D,_=@ignore_derivatives projector_parity(space(U_D',2));
    @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2];
    @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2];
    @tensor AA_Dp_D[:]:=AA_fused[-1,2,-3]*isom_Dp_D[-2,2];
    AA_fused=AA_fused-2*AA_Dp_D;


    #double layer order: L M U = L D U 
    return AA_fused, U_L,U_D,U_U
end

function build_double_layer_Noswap_Tm(Ap,A, with_physical)
    if ~with_physical #no physical leg
        @assert (length(codomain(A))==2)&(length(domain(A))==1)
        @assert (length(codomain(Ap))==1)&(length(domain(Ap))==2)
        #Treat (LU,M) as (LU,D)
        #Treat (M',L'U') as (D',L'U')
        # println(space(Ap))
        # println(space(A))

        # gate=@ignore_derivatives swap_gate(Ap,2,3); #gate L'U'
        # @tensor Ap[:]:=Ap[-1,1,2]*gate[-2,-3,1,2];  

        # gate=@ignore_derivatives parity_gate(Ap,1); #gate D'
        # @tensor Ap[:]:=Ap[1,-2,-3]*gate[-1,1];
        # gate=@ignore_derivatives parity_gate(Ap,3); #gate U'
        # @tensor Ap[:]:=Ap[-1,-2,1]*gate[-3,1];

        
        A=permute(A,(1,2,),(3,));
        Ap=permute(Ap,(1,),(2,3,));
        

        U_L=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 1)), space(Ap, 2) ⊗ space(A, 1));
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 1) ⊗ space(A, 3)), space(Ap, 1) ⊗ space(A, 3));
        U_U=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 2)', fuse(space(Ap, 3)' ⊗ space(A, 2)'));

        @tensor AA_fused[:]:=Ap[5,1,3]*A[2,4,6]*U_L[-1,1,2]*U_D[-2,5,6]*U_U[3,4,-3];

    else #with 3 physical legs grouped as one leg
        @assert (length(codomain(A))==2)&(length(domain(A))==2)
        @assert (length(codomain(Ap))==2)&(length(domain(Ap))==2)
        #Treat (LU,dM) as (LU,dD)
        #Treat (d'M',L'U') as (d'D',L'U')
        # println(space(Ap))
        # println(space(A))

        # gate=@ignore_derivatives swap_gate(Ap,3,4); #gate L'U'
        # @tensor Ap[:]:=Ap[-1,-2,1,2]*gate[-3,-4,1,2];  

        # gate=@ignore_derivatives parity_gate(Ap,2); #gate D'
        # @tensor Ap[:]:=Ap[-1,1,-3,-4]*gate[-2,1];
        # gate=@ignore_derivatives parity_gate(Ap,4); #gate U'
        # @tensor Ap[:]:=Ap[-1,-2,-3,1]*gate[-4,1];

        
        A=permute(A,(1,2,),(3,4,));
        Ap=permute(Ap,(1,2,),(3,4,));
        

        U_L=@ignore_derivatives unitary(fuse(space(Ap, 3) ⊗ space(A, 1)), space(Ap, 3) ⊗ space(A, 1));
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 4)), space(Ap, 2) ⊗ space(A, 4));
        U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 2)', fuse(space(Ap, 4)' ⊗ space(A, 2)'));

        @tensor AA_fused[:]:=Ap[3,6,1,4]*A[2,5,3,7]*U_L[-1,1,2]*U_D[-2,6,7]*U_U[4,5,-3];
    end


    # P_odd_Lp,_=@ignore_derivatives projector_parity(space(U_L',1));
    # P_odd_Up,_=@ignore_derivatives projector_parity(space(U_U',2));
    # P_odd_U,_=@ignore_derivatives projector_parity(space(U_U',3));

    # @tensor isom_Lp[:]:=U_L[-1,4,3]*P_odd_Lp'[4,1]*P_odd_Lp[1,2]*U_L'[2,3,-2];
    # @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2];
    # @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6];
    # @tensor AA_Lp_U[:]:=AA_fused[1,-2,4]*isom_Lp[-1,1]*isom_U[-3,4];
    # AA_fused=AA_fused-2*AA_Lp_U;
    # @tensor AA_Up_U[:]:=AA_fused[-1,-2,4]*isom_Up_U[-3,4];
    # AA_fused=AA_fused-2*AA_Up_U;



    # P_odd_Dp,_=@ignore_derivatives projector_parity(space(U_D',1));
    # P_odd_D,_=@ignore_derivatives projector_parity(space(U_D',2));
    # @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2];
    # @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2];
    # @tensor AA_Dp_D[:]:=AA_fused[-1,2,-3]*isom_Dp_D[-2,2];
    # AA_fused=AA_fused-2*AA_Dp_D;


    #double layer order: L M U = L D U 
    return AA_fused, U_L,U_D,U_U
end

function build_double_layer_swap_Bm(Ap,A, with_physical)
    if with_physical #with one physical leg
        @assert (length(codomain(A))==1)&(length(domain(A))==3)
        @assert (length(codomain(Ap))==3)&(length(domain(Ap))==1)
        #treat (M,dRD) as (U,dRD)
        #treat (d'R'D',M') as (d'R'D',U')
        # println(space(Ap))
        # println(space(A))


        gate=@ignore_derivatives swap_gate(Ap,2,3); #gate D'R'
        @tensor Ap[:]:=Ap[-1,1,2,-4]*gate[-2,-3,1,2];  
        gate=@ignore_derivatives parity_gate(Ap,4); #gate U'
        @tensor Ap[:]:=Ap[-1,-2,-3,1]*gate[-4,1];
        gate=@ignore_derivatives parity_gate(Ap,3);  #gate D'
        @tensor Ap[:]:=Ap[-1,-2,1,-4]*gate[-3,1];
        
        A=permute(A,(1,),(2,3,4,));
        Ap=permute(Ap,(1,2,3,),(4,));
        
    
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 3) ⊗ space(A, 4)), space(Ap, 3) ⊗ space(A, 4));
        U_R=@ignore_derivatives unitary(space(Ap, 2)' ⊗ space(A, 3)', fuse(space(Ap, 2)' ⊗ space(A, 3)'));
        U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 1)', fuse(space(Ap, 4)' ⊗ space(A, 1)'));


        @tensor AA_fused[:]:=Ap[3,1,6,4]*A[5,3,2,7]*U_U[4,5,-3]*U_R[1,2,-2]*U_D[-1,6,7];
    else
        @assert (length(codomain(A))==1)&(length(domain(A))==2)
        @assert (length(codomain(Ap))==2)&(length(domain(Ap))==1)
        #treat (M,RD) as (U,RD)
        #treat (R'D',M') as (R'D',U')

        gate=@ignore_derivatives swap_gate(Ap,1,2); #gate D'R'
        @tensor Ap[:]:=Ap[1,2,-3]*gate[-1,-2,1,2];  
        gate=@ignore_derivatives parity_gate(Ap,3); #gate U'
        @tensor Ap[:]:=Ap[-1,-2,1]*gate[-3,1];
        gate=@ignore_derivatives parity_gate(Ap,2);  #gate D'
        @tensor Ap[:]:=Ap[-1,1,-3]*gate[-2,1];
        
        A=permute(A,(1,),(2,3,));
        Ap=permute(Ap,(1,2,),(3,));
        
    
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 3)), space(Ap, 2) ⊗ space(A, 3));
        U_R=@ignore_derivatives unitary(space(Ap, 1)' ⊗ space(A, 2)', fuse(space(Ap, 1)' ⊗ space(A, 2)'));
        U_U=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 1)', fuse(space(Ap, 3)' ⊗ space(A, 1)'));


        @tensor AA_fused[:]:=Ap[1,6,4]*A[5,2,7]*U_U[4,5,-3]*U_R[1,2,-2]*U_D[-1,6,7];
    end


    ##########################



    P_odd_Up,_=@ignore_derivatives projector_parity(space(U_U',2));
    P_odd_U,_=@ignore_derivatives projector_parity(space(U_U',3));


    @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2];
    @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6];
    @tensor AA_Up_U[:]:=AA_fused[-1,-2,4]*isom_Up_U[-3,4];
    AA_fused=AA_fused-2*AA_Up_U;



    P_odd_Dp,_=@ignore_derivatives projector_parity(space(U_D',1));
    P_odd_D,_=@ignore_derivatives projector_parity(space(U_D',2));
    P_odd_R,_=@ignore_derivatives projector_parity(space(U_R',3));
    @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2];
    @tensor isom_R[:]:=U_R[3,4,-1]*P_odd_R'[4,1]*P_odd_R[1,2]*U_R'[-2,3,2];
    @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2];
    @tensor AA_Dp_D[:]:=AA_fused[2,-2,-3]*isom_Dp_D[-1,2];
    AA_fused=AA_fused-2*AA_Dp_D;
    @tensor AA_Dp_R[:]:=AA_fused[2,3,-3]*isom_Dp[-1,2]*isom_R[-2,3];
    AA_fused=AA_fused-2*AA_Dp_R;


    #double layer order: D R M = D R U 
    return AA_fused, U_D,U_R,U_U
end



function contract_triangle_env(CTM, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD, cx,cy)
    #leading memory cost:
    #chi^2*D^4*d^4
    #D^6*d^6
    global Lx,Ly
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx-1,Lx)][mod1(cy-1,Ly)].C1[1,2]*Tset[mod1(cx,Lx)][mod1(cy-1,Ly)].T1[2,3,-3]*Tset[mod1(cx-1,Lx)][mod1(cy,Ly)].T4[-1,4,1]*T_double_LU[4,5,3]*B_double_LU[-2,-4,5]; 
    @tensor MM_RU[:]:=Tset[mod1(cx+1,Lx)][mod1(cy-1,Ly)].T1[-1,3,1]* Cset[mod1(cx+2,Lx)][mod1(cy-1,Ly)].C2[1,2]* T_double_RU[-2,5,3]*B_double_RU[-4,4,5]*Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx-1,Lx)][mod1(cy+1,Ly)].T4[1,3,-1]*T_double_LD[3,5,-2]*B_double_LD[4,-4,5]*Cset[mod1(cx-1,Lx)][mod1(cy+2,Ly)].C4[2,1]*Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T3[-3,4,2]; 
    @tensor MM_RD[:]:=Tset[mod1(cx+2,Lx)][mod1(cy+1,Ly)].T2[-4,-3,2]*Tset[mod1(cx+1,Lx)][mod1(cy+2,Ly)].T3[1,-2,-1]*Cset[mod1(cx+2,Lx)][mod1(cy+2,Ly)].C3[2,1]; 
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*B_double_RD[1,2,-2]; 


    @tensor LD_LU_RU[:]:=MM_LD[1,2,-1,-2]*MM_LU[1,2,3,4]*MM_RU[3,4,-3,-4];
    @tensor BigTriangle[:]:= LD_LU_RU[1,-1,2,-3]*MM_RD[1,-2,2]; # L M U = L D U
    return BigTriangle
end

function build_triangle_from_4tensors(T,B1_keep,B2_keep,B3_keep)
    @tensor B2_T[:]:=B2_keep[-1,-2,1]*T[1,-3,-4];     #(new2, d2, R2),  (R2, D1, M3) => (new2, d2, D1, M3)
    B2_T=permute_neighbour_ind(B2_T,2,3,4);#(new2, D1, d2, M3)
    B2_T=permute_neighbour_ind(B2_T,1,2,4);#(D1, new2, d2, M3)
    @tensor B1_B2_T[:]:=B1_keep[-1,-2,1]*B2_T[1,-3,-4,-5];#(new1, d1,  D1), (D1, new2, d2, M3) => (new1, d1, new2, d2, M3)

    @tensor B1_B2_T_B3[:]:=B1_B2_T[-1,-2,-3,-4,1]*B3_keep[1,-5,-6];#(new1, d1, new2, d2, M3), (M3, d3, new3) => (new1, d1, new2, d2, d3, new3)
    B1_B2_T_B3=permute_neighbour_ind(B1_B2_T_B3,2,3,6);# new1, new2, d1, d2, d3, new3

    Up=unitary(fuse(space(B1_B2_T_B3,3)*space(B1_B2_T_B3,4)*space(B1_B2_T_B3,5)), space(B1_B2_T_B3,3)*space(B1_B2_T_B3,4)*space(B1_B2_T_B3,5));
    global Up
    @tensor B1_B2_T_B3[:]:=B1_B2_T_B3[-1,-2,1,2,3,-4]*Up[-3,1,2,3];# new1, new2, d123, new3

    B1_B2_T_B3=permute_neighbour_ind(B1_B2_T_B3,1,2,4);# new2, new1, d123, new3
    B1_B2_T_B3=permute(B1_B2_T_B3,(1,2,),(3,4,));# (new2, new1), (d123, new3)

    #########################################
    
    # big_T_compressed=permute_neighbour_ind(B_new,1,2,3);#(D1_new, R2_new, M3_new)
    # @tensor big_T_compressed[:]:=T1_new[-1,-2,1]*big_T_compressed[1,-3,-4];#(M1_R1, d1, R2_new, M3_new) 
    # big_T_compressed=permute_neighbour_ind(big_T_compressed,2,3,4);#(M1_R1,R2_new, d1,  M3_new) 
    # big_T_compressed=permute_neighbour_ind(big_T_compressed,1,2,4);#(R2_new, M1_R1, d1,  M3_new) 
    # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,-3,1]*T3_new[1,-4,-5];#(R2_new, M1_R1, d1,  d3, R3_D3)
    # @tensor big_T_compressed[:]:=T2_new[-1,-2,1]*big_T_compressed[1,-3,-4,-5,-6];#(M2_D2, d2,  M1_R1, d1,  d3, R3_D3)

    # big_T_compressed=permute_neighbour_ind(big_T_compressed,2,3,6);#(M2_D2,  M1_R1, d2, d1,  d3, R3_D3)
    # big_T_compressed=permute_neighbour_ind(big_T_compressed,3,4,6);#(M2_D2,  M1_R1, d1, d2,  d3, R3_D3)
    # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,1,2,3,-4]*Up[-3,1,2,3];#(new2, new1,  d123, new3)
    # big_T_compressed=permute(big_T_compressed,(1,2,),(3,4,))#(new2, new1), (d123, new3)
    

    return B1_B2_T_B3
end

function truncation_direct(big_T,D_max, trun_order, trun_tol)
    #big_T: (new2, new1), (d123, new3)
    global Up
    @tensor big_T[:]:=big_T[-1,-2,1,-6]*Up'[-3,-4,-5,1];# new2, new1, d1,d2,d3, new3
    #big_T=big_T/norm(big_T);
    if trun_order=="simultaneous"

    
        big_T=permute_neighbour_ind(big_T,1,2,6);# new1, new2, d1,d2,d3, new3
        big_T=permute_neighbour_ind(big_T,2,3,6);# new1, d1, new2,d2,d3, new3

        if isa(space(big_T,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
            U1,S1,V1=tsvd(big_T,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
            U3,S3,V3=tsvd(big_T,(1,2,3,4,),(5,6,);trunc=truncdim(D_max));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
        elseif isa(space(big_T,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
            U1,S1,V1=tsvd(big_T,(1,2,),(3,4,5,6,));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
            U1,S1,V1=Truncations(U1,S1,V1,D_max,trun_tol);#println(norm(U1*S1*V1-M_old)/norm(M_old))
            U3,S3,V3=tsvd(big_T,(1,2,3,4,),(5,6,));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
            U3,S3,V3=Truncations(U3,S3,V3,D_max,trun_tol);#println(norm(U3*S3*V3-M_old)/norm(M_old))
        elseif isa(space(big_T,1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
            U1,S1,V1=tsvd(big_T,(1,2,),(3,4,5,6,));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
            U1,S1,V1=Truncations(U1,S1,V1,D_max,trun_tol);#println(norm(U1*S1*V1-M_old)/norm(M_old))
            U3,S3,V3=tsvd(big_T,(1,2,3,4,),(5,6,));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
            U3,S3,V3=Truncations(U3,S3,V3,D_max,trun_tol);#println(norm(U3*S3*V3-M_old)/norm(M_old))
        end
    
    
        big_T=permute_neighbour_ind(big_T,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3
        big_T=permute_neighbour_ind(big_T,1,2,6);# M2_D2, M1_R1, d1, d2, d3, R3_D3
        big_T=permute_neighbour_ind(big_T,3,4,6);# M2_D2, M1_R1, d2, d1, d3, R3_D3
        big_T=permute_neighbour_ind(big_T,2,3,6);# M2_D2, d2, M1_R1, d1, d3, R3_D3
        # T1_T2_B_T3=T1_T2_B_T3/norm(T1_T2_B_T3);
        if isa(space(big_T,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
            U2,S2,V2=tsvd(big_T,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
        elseif isa(space(big_T,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
            U2,S2,V2=tsvd(big_T,(1,2,),(3,4,5,6,));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
            U2,S2,V2=Truncations(U2,S2,V2,D_max,trun_tol);#println(norm(U2*S2*V2-M_old)/norm(M_old))
        elseif isa(space(big_T,1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
            U2,S2,V2=tsvd(big_T,(1,2,),(3,4,5,6,));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
            U2,S2,V2=Truncations(U2,S2,V2,D_max,trun_tol);#println(norm(U2*S2*V2-M_old)/norm(M_old))
        end
    
        # λ_2_new=permute(S2,(2,),(1,));
    
        # @tensor T2_new[:]:=T2_res[-1,1]*U2[1,-2,-3];#(M2_D2, d2, R2_new)
        # T1_B_T3=S2*V2;#(R2_new, M1_R1, d1, d3, R3_D3)
        # @tensor T1_B[:]:=T1_B_T3[-1,-2,-3,1,2]*V3'[1,2,-4];#(R2_new, M1_R1, d1, M3_new) 
        # @tensor T3_new[:]:=V3[-1,-2,1]*T3_res[1,-3];#(M3_new, d3, R3_D3)
        # λ_3_new=S3;
    
        # T1_B=permute_neighbour_ind(T1_B,1,2,4);#(M1_R1, R2_new, d1, M3_new) 
        # T1_B=permute_neighbour_ind(T1_B,2,3,4);#(M1_R1, d1, R2_new, M3_new) 
        # @tensor B_new[:]:=U1'[-1,1,2]*T1_B[1,2,-2,-3];#(D1_new, R2_new, M3_new) 
        # @tensor T1_new[:]:=T1_res[-1,1]*U1[1,-2,-3];#(M1_R1, d1, D1_new) 
        # λ_1_new=permute(S1,(2,),(1,));
    
        # #B_new: (D1_new, R2_new, M3_new) => (R2, D1, M3)
        # B_new=permute_neighbour_ind(B_new,1,2,3);#(R2_new, D1_new, M3_new)
        # B_new=permute(B_new,(1,2,),(3,));
    
        # #T1_new: (M1_R1, d1, D1_new) => (M1, d1, R1, D1)
        # @tensor T1_new[:]:=T1_new[1,-3,-4]*Ut1'[-1,-2,1];#(M1, R1, d1, D1_new)
        # T1_new=permute_neighbour_ind(T1_new,2,3,4);#(M1, d1, R1, D1_new)
    
        # #T2_new: (M2_D2, d2, R2_new) => (M2, d2, R2, D2) 
        # @tensor T2_new[:]:=T2_new[1,-3,-4]*Ut2'[-1,-2,1];#(M2, D2, d2, R2_new)
        # T2_new=permute_neighbour_ind(T2_new,2,3,4);#(M2, d2, D2, R2_new)
        # T2_new=permute_neighbour_ind(T2_new,3,4,4);#(M2, d2, R2_new, D2)
    
        # #T3_new: (M3_new, d3, R3_D3) => (M3, d3, R3, D3)
        # @tensor T3_new[:]:=T3_new[-1,-2,1]*Ut3'[-3,-4,1];#(M3_new, d3, R3, D3)


    
        T2_new=U2*sqrt(S2);#(M2_D2, d2, R2_new)     #here how to absorb S doesn't matter, as we will redetermine S by svd after sweep optimization 
        T2_new=permute(T2_new,(1,2,),(3,));
        B_new=sqrt(S2)*V2;#(R2_new, M1_R1, d1, d3, R3_D3)
        λ_2_new=sqrt(S2);

        λ_3_new=sqrt(S3);
        λ_3_new_inv=my_pinv(λ_3_new);
        @tensor B_new[:]:=B_new[-1,-2,-3,1,2]*V3'[1,2,3]*λ_3_new_inv'[3,-4];#(R2_new, M1_R1, d1, M3_new) 
        T3_new=sqrt(S3)*V3;#(M3_new, d3, R3_D3)
        T3_new=permute(T3_new,(1,2,),(3,));
    
        B_new=permute_neighbour_ind(B_new,1,2,4);#(M1_R1, R2_new, d1, M3_new) 
        B_new=permute_neighbour_ind(B_new,2,3,4);#(M1_R1, d1, R2_new, M3_new) 
        λ_1_new=sqrt(S1);
        λ_1_new_inv=my_pinv(λ_1_new);
        @tensor B_new[:]:=λ_1_new_inv'[-1,3]*U1'[3,1,2]*B_new[1,2,-2,-3];#(D1_new, R2_new, M3_new) 
        T1_new=U1*sqrt(S1);#(M1_R1, d1, D1_new) 
        T1_new=permute(T1_new,(1,2,),(3,));
    
        #B_new: (D1_new, R2_new, M3_new) => (R2, D1, M3)
        B_new=permute_neighbour_ind(B_new,1,2,3);#(R2_new, D1_new, M3_new)
        B_new=permute(B_new,(1,2,),(3,));
    
        method=2;#two methods are the same
        if method==1
            # big_T_compressed=permute_neighbour_ind(B_new,1,2,3);#(D1_new, R2_new, M3_new)
            # @tensor big_T_compressed[:]:=T1_new[-1,-2,1]*big_T_compressed[1,-3,-4];#(M1_R1, d1, R2_new, M3_new) 
            # big_T_compressed=permute_neighbour_ind(big_T_compressed,2,3,4);#(M1_R1,R2_new, d1,  M3_new) 
            # big_T_compressed=permute_neighbour_ind(big_T_compressed,1,2,4);#(R2_new, M1_R1, d1,  M3_new) 
            # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,-3,1]*T3_new[1,-4,-5];#(R2_new, M1_R1, d1,  d3, R3_D3)
            # @tensor big_T_compressed[:]:=T2_new[-1,-2,1]*big_T_compressed[1,-3,-4,-5,-6];#(M2_D2, d2,  M1_R1, d1,  d3, R3_D3)

            # big_T_compressed=permute_neighbour_ind(big_T_compressed,2,3,6);#(M2_D2,  M1_R1, d2, d1,  d3, R3_D3)
            # big_T_compressed=permute_neighbour_ind(big_T_compressed,3,4,6);#(M2_D2,  M1_R1, d1, d2,  d3, R3_D3)
            # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,1,2,3,-4]*Up[-3,1,2,3];#(new2, new1,  d123, new3)
            # big_T_compressed=permute(big_T_compressed,(1,2,),(3,4,))#(new2, new1), (d123, new3)


        else
            big_T_compressed=build_triangle_from_4tensors(B_new,T1_new,T2_new,T3_new)
        end
        

        return B_new,T1_new,T2_new,T3_new, big_T_compressed
    elseif trun_order=="successive"
    end
end

function truncation_Env_gauge(env_top,env_bot, big_T,D_max, trun_order, trun_tol)
    #big_T: (2, 1), (d123, 3)
    #env_bot: new_ind,2,3,1
    #env_top: 2,3,1, new_ind
    u,s,v=tsvd(env_bot,(1,2,3,),(4,));
    gauge1=s*v;#11,1
    gauge1_inv=v'*my_pinv(s);
    u,s,v=tsvd(env_bot,(1,3,4,),(2,));
    gauge2=s*v;#2,2
    gauge2_inv=v'*my_pinv(s);
    u,s,v=tsvd(env_bot,(1,2,4,),(3,));
    gauge3=s*v;#33,3
    gauge3_inv=v'*my_pinv(s);

    @tensor big_T_new[:]:=big_T[2,1,-3,3]*gauge1[-2,1]*gauge2[-1,2]*gauge3[-4,3];
    big_T_new=permute(big_T_new,(1,2,),(3,4,));

    B_new,T1_new,T2_new,T3_new, big_T_compressed=truncation_direct(big_T_new,D_max, trun_order, trun_tol);
    @tensor big_T_compressed[:]:=big_T_compressed[2,1,-3,3]*gauge1_inv[-2,1]*gauge2_inv[-1,2]*gauge3_inv[-4,3];
    big_T_compressed=permute(big_T_compressed,(1,2,),(3,4,));

    #T1_new: (M1_R1, d1, D1_new) 
    @tensor T1_new[:]:=T1_new[1,-2,-3]*gauge1_inv[-1,1];
    T1_new=permute(T1_new,(1,2,),(3,));
    #T2_new: (M2_D2, d2, R2_new)
    @tensor T2_new[:]:=T2_new[1,-2,-3]*gauge2_inv[-1,1];
    T2_new=permute(T2_new,(1,2,),(3,));
    #T3_new: (M3_new, d3, R3_D3)
    @tensor T3_new[:]:=T3_new[-1,-2,1]*gauge3_inv[-3,1];
    T3_new=permute(T3_new,(1,2,),(3,));

    return B_new,T1_new,T2_new,T3_new, big_T_compressed
end



function triangle_FullUpdate(dt,B_set, T_set,CTM_cell,Lx,Ly,coord, D_max, trun_order, trun_tol, n_sweep)
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

    (c1,c2)=coord;

    gates_ru_ld_rd=gate_RU_LD_RD(parameters,dt, typeof(space(B_set[1],1)),Lx);
    #gates_ru_ld_rd=H_RU_LD_RD(parameters, typeof(space(B_set[1],1)),Lx);
    gates_ru_ld_rd=gates_ru_ld_rd[mod1(c1,Lx)];
    

    B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op = split_3Tesnsors(T_set[mod1(c1+1,Lx),c2], T_set[c1,mod1(c2+1,Ly)], T_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], B_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], gates_ru_ld_rd);


    T_LU=B_set[c1,c2];
    T_double_LU, _ = build_double_layer_swap_Tm(T_LU',T_LU, false);#L M U
    B_LU=T_set[c1,c2];
    B_double_LU, _ = build_double_layer_swap_Bm(B_LU',B_LU, true);#D R M

    T_RU=B_set[mod1(c1+1,Lx),c2];
    T_double_RU, _ = build_double_layer_swap_Tm(T_RU',T_RU, false);#L M U

    T_LD=B_set[c1,mod1(c2+1,Ly)];
    T_double_LD, _ = build_double_layer_swap_Tm(T_LD',T_LD, false);#L M U

    B_double_RU, _ = build_double_layer_swap_Bm(B1_res',B1_res,false);#D R M
    B_double_LD, _ = build_double_layer_swap_Bm(B2_res',B2_res,false);#D R M
    B_double_RD, _ = build_double_layer_swap_Bm(B3_res',B3_res,false);#D R M
    BigTriangle_double_Noswap, U_L,U_D,U_U = build_double_layer_Noswap_Tm(B1_B2_T_B3',B1_B2_T_B3_op, true);#L M U = L D U
    BigTriangle_double_env=contract_triangle_env(CTM_cell, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD, mod1(c1,Lx),mod1(c2,Ly));#L M U = L D U

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

    (c1,c2)=coord;
    T_set[mod1(c1+1,Lx),c2]=T1_new_opt;
    T_set[c1,mod1(c2+1,Ly)]=T2_new_opt;
    T_set[mod1(c1+1,Lx),mod1(c2+1,Ly)]=T3_new_opt;
    B_set[mod1(c1+1,Lx),mod1(c2+1,Ly)]=B_new_opt;

    return B_set, T_set
end





function FullUpdate_iPESS(tau,dt,B_set, T_set,Lx,Ly, D_max, trun_order, trun_tol, n_sweep, ENV_ctm_setting)
    println("tau, dt="*string([tau,dt]))

    #get initial CTM tensors
    A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,ENV_ctm_setting);
    ENV_ctm_setting.CTM_ite_info=false;


    for ct=1:Int(round(tau/abs(dt)))
        println("iteration "*string(ct));flush(stdout)
        for ca=1:Lx
            for cb=1:Ly
                coord=[ca,cb];

                B_set, T_set=triangle_FullUpdate(dt,B_set, T_set,CTM_cell,Lx,Ly,coord, D_max, trun_order, trun_tol, n_sweep)
                
                #get CTM tensors for energy and next optimization
                A_cell_iPEPS=convert_iPESS_to_iPEPS(B_set,T_set);
                init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
                @time CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell_iPEPS,chi,init, init_CTM,ENV_ctm_setting);
                println("ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
            end
        end

        E_total,  ex_set, ey_set, e_diagonala_set, e0_set, eU_set=evaluate_ob_cell(parameters, A_cell_iPEPS, AA_cell, CTM_cell, ENV_ctm_setting, energy_setting);
        println("E= "*string(E_total)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonala_set= "*string(e_diagonala_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"eU_set= "*string(eU_set[:]));flush(stdout);

        global E_history
        if E_total<minimum(E_history)
            E_history=vcat(E_history,E_total);
            global save_filenm
            jldsave(save_filenm; B_set, T_set);
            global starting_time
            Now=now();
            Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
            println("Time consumed: "*string(Time));flush(stdout);
        end

    end
    return Bset, Tset
end




