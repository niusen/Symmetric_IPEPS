#step 1: 
# replace gate by Hamiltonian, verify energies of each triangle, but this is hard to do due to renormalization of tensors during tebd


#step 2:
#set dt=0, check the state and energies don't change

#step 3:
#do evolution on each single terms to verify evolution code 


function verif2(parameters_,D_max, B_set0, T_set0, trun_tol)
    lambdaset1,lambdaset2,lambdaset3=get_trivial_lambda(B_set0);

    psi=B_T_sets_to_PESS(B_set0,T_set0);
    psi_PEPS=PESS_to_PEPS_matrix(psi);
    psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi_PEPS,psi_PEPS,Lx,Ly);
    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_old(psi_PEPS,psi_double);
    println(E_total)

    B_set, T_set, lambdaset1, lambdaset2, lambdaset3=tebd_PESS_no_Hamiltonian(parameters, B_set0, T_set0, lambdaset1, lambdaset2, lambdaset3,  10, Dmax, trun_tol)

    psi=B_T_sets_to_PESS(B_set,T_set);
    psi_PEPS=PESS_to_PEPS_matrix(psi);
    psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi_PEPS,psi_PEPS,Lx,Ly);
    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_old(psi_PEPS,psi_double);
    println(E_total)

end


function verif1(parameters_,D_max, psi0, Bset0, Tset0, trun_tol)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    Lx,Ly=size(Bset0);
    Norm=overlap(psi0,psi0,2,2);
    # println(Norm)



    tx=parameters_["t1"];
    ty=parameters_["t1"];
    t2=parameters_["t2"];
    ϕ=parameters_["ϕ"];
    U=parameters_["U"];
    gates_bulk=H_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=parameters_["t1"];
    ty=0;
    t2=0;
    ϕ=parameters_["ϕ"];
    U=parameters_["U"];
    gates_top=H_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=parameters_["t1"];
    t2=0;
    ϕ=parameters_["ϕ"];
    U=parameters_["U"];
    gates_left=H_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=0;
    t2=0;
    ϕ=parameters_["ϕ"];
    U=parameters_["U"];
    gates_left_top=H_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);






    
    all_triangles=get_triangles(Lx,Ly);
    for tran_set_id=1:length(all_triangles)
        trangle_set=all_triangles[tran_set_id];

        for cs=1:length(trangle_set)
            trangle_coord=trangle_set[cs];println(trangle_coord)
            # B
            #CD
            posTB=[trangle_coord[1]+1,trangle_coord[2]+1];
            posTD=[trangle_coord[1]+1,trangle_coord[2]];
            posTC=[trangle_coord[1],trangle_coord[2]];
            posBond=posTD;

            #absorb λ tensors into physical tensors
            # λ tensors not used for update, only used for checking convergence
            if (trangle_coord[1]>0)&&(trangle_coord[2]<Ly) #bulk triangle
                Bset=deepcopy(Bset0);
                Tset=deepcopy(Tset0);

                TB=Tset[posTB[1],posTB[2]];
                TC=Tset[posTC[1],posTC[2]];
                TD=Tset[posTD[1],posTD[2]];

                B=Bset[posBond[1],posBond[2]];

                TB, TC, TD, B, lambda_1, lambda_2, lambda_3=triangle_gate_iPESS_simplified_no_normalization(D_max,gates_bulk[mod1(trangle_coord[1],2)], TB, TC, TD, B, trun_tol);

                TB=permute(TB,(1,),(2,3,4,));
                TC=permute(TC,(1,),(2,3,4,));
                TD=permute(TD,(1,),(2,3,4,));

                TB=TB/norm(TB);
                TC=TC/norm(TC);
                TD=TD/norm(TD);
                B=B/norm(B);
                lambda_1=lambda_1/norm(lambda_1);
                lambda_2=lambda_2/norm(lambda_2);
                lambda_3=lambda_3/norm(lambda_3);
                
                lambdaset1[posTD[1],posTD[2]]=lambda_1;
                lambdaset2[posTD[1],posTD[2]]=lambda_2;
                lambdaset3[posTD[1],posTD[2]]=lambda_3;
                Tset[posTB[1],posTB[2]]=TB;
                Tset[posTC[1],posTC[2]]=TC;
                Tset[posTD[1],posTD[2]]=TD;
                Bset[posBond[1],posBond[2]]=B;


                psi=B_T_sets_to_PESS(Bset,Tset);
                ob=overlap(psi0,psi,trangle_coord[1],trangle_coord[2]);
                println(ob/Norm)

            elseif (trangle_coord[1]==0)&&(trangle_coord[2]<Ly) #left triangle
                Bset=deepcopy(Bset0);
                Tset=deepcopy(Tset0);

                #TC is dummy
                TB=Tset[posTB[1],posTB[2]];
                #TC=Tset[posTC[1],posTC[2]];
                TD=Tset[posTD[1],posTD[2]];

                B=Bset[posBond[1],posBond[2]];

                #define trivial TC:
                Vp=space(TD)[2];
                Vv=space(B)[1];
                V_trivial=Rep[SU₂](0=>1);
                TC=TensorMap(randn,V_trivial,Vp'*Vv*V_trivial);
                TC=TC/norm(TC);
                
                TB, TC, TD, B, lambda_1, lambda_2, lambda_3=triangle_gate_iPESS_simplified_no_normalization(D_max,gates_left[mod1(trangle_coord[1],2)], TB, TC, TD, B, trun_tol);

                TB=permute(TB,(1,),(2,3,4,));
                #TC=permute(TC,(1,),(2,3,4,));
                TD=permute(TD,(1,),(2,3,4,));

                TB=TB/norm(TB);
                #TC=TC/norm(TC);
                TD=TD/norm(TD);
                B=B/norm(B);

                Tset[posTB[1],posTB[2]]=TB;
                #Tset[posTC[1],posTC[2]]=TC;
                Tset[posTD[1],posTD[2]]=TD;
                Bset[posBond[1],posBond[2]]=B;


                psi=B_T_sets_to_PESS(Bset,Tset);
                ob=overlap(psi0,psi,trangle_coord[1],trangle_coord[2]);
                println(ob/Norm)

            elseif (trangle_coord[1]>0)&&(trangle_coord[2]==Ly) #top triangle
                Bset=deepcopy(Bset0);
                Tset=deepcopy(Tset0);

                #TB is dummy
                #TB=Tset[posTB[1],posTB[2]];
                TC=Tset[posTC[1],posTC[2]];
                TD=Tset[posTD[1],posTD[2]];

                B=Bset[posBond[1],posBond[2]];

                #define trivial TB:
                Vp=space(TD)[2];
                Vv=space(B)[2];
                V_trivial=Rep[SU₂](0=>1);
                TB=TensorMap(randn,V_trivial,Vp'*V_trivial*Vv);
                TB=TB/norm(TB);

                
                TB, TC, TD, B, lambda_1, lambda_2, lambda_3=triangle_gate_iPESS_simplified_no_normalization(D_max,gates_top[mod1(trangle_coord[1],2)], TB, TC, TD, B, trun_tol);

                #TB=permute(TB,(1,),(2,3,4,));
                TC=permute(TC,(1,),(2,3,4,));
                TD=permute(TD,(1,),(2,3,4,));

                #TB=TB/norm(TB);
                TC=TC/norm(TC);
                TD=TD/norm(TD);
                B=B/norm(B);

                #Tset[posTB[1],posTB[2]]=TB;
                Tset[posTC[1],posTC[2]]=TC;
                Tset[posTD[1],posTD[2]]=TD;
                Bset[posBond[1],posBond[2]]=B;

                psi=B_T_sets_to_PESS(Bset,Tset);
                ob=overlap(psi0,psi,trangle_coord[1],trangle_coord[2]);
                println(ob/Norm)

            elseif (trangle_coord[1]==0)&&(trangle_coord[2]==Ly) #left-top corner
                Bset=deepcopy(Bset0);
                Tset=deepcopy(Tset0);

                #TB, TC are dummy
                #TB=Tset[posTB[1],posTB[2]];
                #TC=Tset[posTC[1],posTC[2]];
                TD=Tset[posTD[1],posTD[2]];

                B=Bset[posBond[1],posBond[2]];

                #define trivial TC:
                Vp=space(TD)[2];
                Vv=space(B)[2];
                V_trivial=Rep[SU₂](0=>1);
                TC=TensorMap(randn,V_trivial,Vp'*Vv*V_trivial);

                TC=TC/norm(TC);

                #define trivial TB:
                Vp=space(TD)[2];
                Vv=space(B)[1];
                V_trivial=Rep[SU₂](0=>1);
                TB=TensorMap(randn,V_trivial,Vp'*V_trivial*Vv);
                
                TB=TB/norm(TB);

                println(B)

                
                TB, TC, TD, B, lambda_1, lambda_2, lambda_3=triangle_gate_iPESS_simplified_no_normalization(D_max,gates_left_top[mod1(trangle_coord[1],2)], TB, TC, TD, B, trun_tol);

                #TB=permute(TB,(1,),(2,3,4,));
                #TC=permute(TC,(1,),(2,3,4,));
                TD=permute(TD,(1,),(2,3,4,));
                println(B)

                #TB=TB/norm(TB);
                #TC=TC/norm(TC);
                # TD=TD/norm(TD);
                # B=B/norm(B);

                #Tset[posTB[1],posTB[2]]=TB;
                #Tset[posTC[1],posTC[2]]=TC;
                Tset[posTD[1],posTD[2]]=TD;
                Bset[posBond[1],posBond[2]]=B;

                ########################
                # Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
                # U_op=n_double_set[1]-(1/2)*N_occu_set[1]+(1/4)*Ident_set[1];
                # @tensor TD[:]:=TD[-1,1,-3,-4]*U_op[-2,1];
                # TD=permute(TD,(1,),(2,3,4,));
                # Tset[posTD[1],posTD[2]]=TD;
                ##########################

                psi=B_T_sets_to_PESS(Bset,Tset);
                ob=overlap(psi0,psi,trangle_coord[1],trangle_coord[2]);
                println(ob/Norm)
            else
                error("unknown case")
            end



        end
    end


end


function triangle_gate_iPESS_simplified_no_normalization(D_max, op_LD_RD_RU, T1, T2, T3, B, trun_tol)
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

    T1=permute_neighbour_ind(T1,2,3,4);#M1, R1, d1,  D1
    Ut1=unitary(fuse(space(T1,1)*space(T1,2)), space(T1,1)*space(T1,2));
    @tensor T1[:]:=Ut1[-1,1,2]*T1[1,2,-2,-3];#M1_R1, d1,  D1
    uu,ss,vv=tsvd(permute(T1,(1,),(2,3,)));
    T1_res=uu;
    T1_keep=ss*vv;

    T2=permute_neighbour_ind(T2,3,4,4);#M2, d2, D2, R2
    T2=permute_neighbour_ind(T2,2,3,4);#M2, D2, d2, R2
    Ut2=unitary(fuse(space(T2,1)*space(T2,2)), space(T2,1)*space(T2,2));
    @tensor T2[:]:=Ut2[-1,1,2]*T2[1,2,-2,-3];#M2_D2, d2, R2
    uu,ss,vv=tsvd(permute(T2,(1,),(2,3,)));
    T2_res=uu;
    T2_keep=ss*vv;

    T3=T3;#M3, d3, R3, D3 
    Ut3=unitary(fuse(space(T3,3)*space(T3,4)), space(T3,3)*space(T3,4));
    @tensor T3[:]:=T3[-1,-2,1,2]*Ut3[-3,1,2];#M3, d3, R3_D3 
    uu,ss,vv=tsvd(permute(T3,(1,2,),(3,)));
    T3_keep=uu*ss;
    T3_res=vv;

    @tensor T2_B[:]:=T2_keep[-1,-2,1]*B[1,-3,-4];     #(M2_D2, d2, R2),  (R2, D1, M3) => (M2_D2, d2, D1, M3)
    T2_B=permute_neighbour_ind(T2_B,2,3,4);#(M2_D2, D1, d2, M3)
    T2_B=permute_neighbour_ind(T2_B,1,2,4);#(D1, M2_D2, d2, M3)
    @tensor T1_T2_B[:]:=T1_keep[-1,-2,1]*T2_B[1,-3,-4,-5];#(M1_R1, d1,  D1), (D1, M2_D2, d2, M3) => (M1_R1, d1, M2_D2, d2, M3)

    @tensor T1_T2_B_T3[:]:=T1_T2_B[-1,-2,-3,-4,1]*T3_keep[1,-5,-6];#(M1_R1, d1, M2_D2, d2, M3), (M3, d3, R3_D3) => (M1_R1, d1, M2_D2, d2, d3, R3_D3)
    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3

    #d2',d3',d1', d2,d3,d1
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,5,6,6);#d2',d3',d1', d2,d1,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,4,5,6);#d2',d3',d1', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,2,3,6);#d2',d1',d3', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,1,2,6);#d1',d2',d3', d1,d2,d3
    @tensor T1_T2_B_T3[:]:=T1_T2_B_T3[-1,-2,1,2,3,-6]*op_LD_RD_RU[-3,-4,-5,1,2,3];


    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, d1, M2_D2, d2, d3, R3_D3
    #T1_T2_B_T3=T1_T2_B_T3/norm(T1_T2_B_T3);
    if isa(space(T1,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        U1,S1,V1=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
        U3,S3,V3=tsvd(T1_T2_B_T3,(1,2,3,4,),(5,6,);trunc=truncdim(D_max));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
    elseif isa(space(T1,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        U1,S1,V1=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
        U1,S1,V1=Truncations(U1,S1,V1,D_max,trun_tol);#println(norm(U1*S1*V1-M_old)/norm(M_old))
        U3,S3,V3=tsvd(T1_T2_B_T3,(1,2,3,4,),(5,6,));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
        U3,S3,V3=Truncations(U3,S3,V3,D_max,trun_tol);#println(norm(U3*S3*V3-M_old)/norm(M_old))
    end


    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3
    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,1,2,6);# M2_D2, M1_R1, d1, d2, d3, R3_D3
    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,3,4,6);# M2_D2, M1_R1, d2, d1, d3, R3_D3
    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M2_D2, d2, M1_R1, d1, d3, R3_D3
    #T1_T2_B_T3=T1_T2_B_T3/norm(T1_T2_B_T3);
    if isa(space(T1,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        U2,S2,V2=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
    elseif isa(space(T1,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        U2,S2,V2=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
        U2,S2,V2=Truncations(U2,S2,V2,D_max,trun_tol);#println(norm(U2*S2*V2-M_old)/norm(M_old))
    end

    λ_2_new=permute(S2,(2,),(1,));

    @tensor T2_new[:]:=T2_res[-1,1]*U2[1,-2,2]*S2[2,-3];#(M2_D2, d2, R2_new)
    T1_B_T3=V2;#(R2_new, M1_R1, d1, d3, R3_D3)

    S3_inv=my_pinv(S3);
    @tensor T1_B[:]:=T1_B_T3[-1,-2,-3,1,2]*V3'[1,2,3]*S3_inv'[3,-4];#(R2_new, M1_R1, d1, M3_new) 
    @tensor T3_new[:]:=S3[-1,2]*V3[2,-2,1]*T3_res[1,-3];#(M3_new, d3, R3_D3)
    λ_3_new=S3;

    T1_B=permute_neighbour_ind(T1_B,1,2,4);#(M1_R1, R2_new, d1, M3_new) 
    T1_B=permute_neighbour_ind(T1_B,2,3,4);#(M1_R1, d1, R2_new, M3_new) 
    S1_inv=my_pinv(S1);
    @tensor B_new[:]:=S1_inv'[-1,3]*U1'[3,1,2]*T1_B[1,2,-2,-3];#(D1_new, R2_new, M3_new) 
    @tensor T1_new[:]:=T1_res[-1,1]*U1[1,-2,2]*S1[2,-3];#(M1_R1, d1, D1_new) 
    λ_1_new=permute(S1,(2,),(1,));

    #B_new: (D1_new, R2_new, M3_new) => (R2, D1, M3)
    B_new=permute_neighbour_ind(B_new,1,2,3);#(R2_new, D1_new, M3_new)
    B_new=permute(B_new,(1,2,),(3,));

    #T1_new: (M1_R1, d1, D1_new) => (M1, d1, R1, D1)
    @tensor T1_new[:]:=T1_new[1,-3,-4]*Ut1'[-1,-2,1];#(M1, R1, d1, D1_new)
    T1_new=permute_neighbour_ind(T1_new,2,3,4);#(M1, d1, R1, D1_new)

    #T2_new: (M2_D2, d2, R2_new) => (M2, d2, R2, D2) 
    @tensor T2_new[:]:=T2_new[1,-3,-4]*Ut2'[-1,-2,1];#(M2, D2, d2, R2_new)
    T2_new=permute_neighbour_ind(T2_new,2,3,4);#(M2, d2, D2, R2_new)
    T2_new=permute_neighbour_ind(T2_new,3,4,4);#(M2, d2, R2_new, D2)

    #T3_new: (M3_new, d3, R3_D3) => (M3, d3, R3, D3)
    @tensor T3_new[:]:=T3_new[-1,-2,1]*Ut3'[-3,-4,1];#(M3_new, d3, R3, D3)

    return T1_new, T2_new, T3_new, B_new, λ_1_new, λ_2_new, λ_3_new
end