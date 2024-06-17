function iPEPS_to_iPESS(A::Square_iPEPS)
    T=A.T;
    T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
    u,s,v=tsvd(permute(T,(1,2,),(3,4,5,)));
    Tm=u*s;#|LU><M|
    Bm=v;#|Md><|RD
    A_new=Triangle_iPESS(Bm,Tm)
    return A_new
end

function iPEPS_to_iPESS(T::TensorMap)
    T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
    u,s,v=tsvd(permute(T,(1,2,),(3,4,5,)));
    Tm=u*s;#|LU><M|
    Bm=v;#|Md><|RD
    A_new=Triangle_iPESS(Bm,Tm)
    return A_new
end

function iPESS_to_iPEPS(A::Triangle_iPESS)
    Tm=A.Tm;#|LU><M|
    Bm=A.Bm;#|Md><|RD
    T=permute(Tm*Bm,(1,5,4,2,3,));#L,D,R,U,d,
    A_new=Square_iPEPS(T)
    return A_new
end


function iPEPS_to_iPESS_tuple(A_set::Tuple)    
    global Lx,Ly
    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=iPEPS_to_iPESS(A_set[cx][cy]);
            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end
    return A_cell
end


function iPESS_to_iPEPS_tuple(A_set::Tuple)    
    global Lx,Ly
    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=iPESS_to_iPEPS(A_set[cx][cy]::Triangle_iPESS);
            A_cell=fill_tuple(A_cell, A.T, cx,cy);
        end
    end
    return A_cell
end


function iPESS_to_iPEPS_matrix(A_set::Matrix{Triangle_iPESS})    
    Lx,Ly=size(A_set)
    A_cell=Matrix{Square_iPEPS}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=iPESS_to_iPEPS(A_set[cx,cy]::Triangle_iPESS);
            A_cell[cx,cy]=A;
        end
    end
    return A_cell
end

# function Ham_triangle_RU_LD_RD(parameters, space_type,Lx)

#     if space_type==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
#         Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
#     elseif space_type==GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
#         Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
#     elseif space_type==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
#         Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_Z2();
#     end
    
#     t1=parameters["t1"];
#     t2=parameters["t2"];
#     ϕ=parameters["ϕ"];
#     U=parameters["U"];

#     tx_coe_set=[exp(im*ϕ),exp(im*ϕ)]*t1;
#     # ty_coe_set=[-1,1]*t1;
#     # t2_coe_set=[-1,1]*t2;
#     ty_coe_set=[1,-1]*t1;
#     t2_coe_set=[1,-1]*t2;
#     U_coe=U;

#     H_set=Matrix{TensorMap}(undef,2,1);
#     for cx=1:2;
#         ####################
#         # O1=Cdag_set[mod1(cx,Lx)];
#         # O2=C_set[mod1(cx+1,Lx)];
#         O1=Cdag_set[mod1(cx,2)];
#         O2=C_set[mod1(cx+1,2)];
#         @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
#         op=op*tx_coe_set[cx];
#         op=permute(op,(1,2,),(3,4,));
#         hh=op+op';
#         Id=unitary(space(hh,2),space(hh,2));
#         @tensor hh_tx[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
#         ######################
#         # O1=Cdag_set[mod1(cx+1,Lx)];
#         # O2=C_set[mod1(cx+1,Lx)];
#         O1=Cdag_set[mod1(cx+1,2)];
#         O2=C_set[mod1(cx+1,2)];
#         @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
#         op=op*ty_coe_set[cx];
#         op=permute(op,(1,2,),(3,4,));
#         hh=op+op';
#         # Id=unitary(space(Cdag_set[mod1(cx,Lx)],2),space(Cdag_set[mod1(cx,Lx)],2));
#         Id=unitary(space(Cdag_set[mod1(cx,2)],2),space(Cdag_set[mod1(cx,2)],2));
#         @tensor hh_ty[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
#         #####################
#         # O1=Cdag_set[mod1(cx,Lx)];
#         # O2=C_set[mod1(cx+1,Lx)];
#         O1=Cdag_set[mod1(cx,2)];
#         O2=C_set[mod1(cx+1,2)];
#         @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
#         op=-op;#!!!!!!! somehow this minus sign is required
#         op=op*t2_coe_set[cx];
#         op=permute(op,(1,2,),(3,4,));
#         hh=op+op';
#         Id=unitary(space(hh,2),space(hh,2));
#         @tensor hh[:]:=hh[-1,-3,-4,-6]*Id[-2,-5];
#         sgate=swap_gate(hh,2,3);
#         @tensor hh_t2[:]:=sgate[-2,-3,1,2]*hh[-1,1,2,-4,3,4]*sgate'[3,4,-5,-6];
#         #################
#         # OU_LD=n_double_set[mod1(cx,Lx)]-(1/2)*N_occu_set[mod1(cx,Lx)]+(1/4)*Ident_set[mod1(cx,Lx)];
#         # OU_RU=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
#         # OU_RD=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
#         OU_LD=n_double_set[mod1(cx,2)]-(1/2)*N_occu_set[mod1(cx,2)]+(1/4)*Ident_set[mod1(cx,2)];
#         OU_RU=n_double_set[mod1(cx+1,2)]-(1/2)*N_occu_set[mod1(cx+1,2)]+(1/4)*Ident_set[mod1(cx+1,2)];
#         OU_RD=n_double_set[mod1(cx+1,2)]-(1/2)*N_occu_set[mod1(cx+1,2)]+(1/4)*Ident_set[mod1(cx+1,2)];
#         Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
#         Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
#         Id_RD=unitary(space(OU_RD,1),space(OU_RD,1));
#         @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
#         # @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
#         # @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
#         # hh_U=(hh_LD+hh_RU+hh_RD)*U_coe;
#         hh_U=(hh_LD)*U_coe;
        
#         #################
#         hh=permute(hh_tx+hh_ty+hh_t2+hh_U,(1,2,3,),(4,5,6,));#hh_tx+hh_ty+hh_t2+hh_U
#         H_set[cx]=hh;
#     end
#     return H_set
# end


# function triangle_update_iPESS(Bset, Tset, Hs)
#     """
#     ABABABAB
#     CDCDCDCD
#     ABABABAB
#     CDCDCDCD
#     """
#     Lx,Ly=size(Bset);
#     for ca=1:Lx
#         for cb=1:Ly
#             # B
#             #CD
#             posTB=[mod1(ca+1,Lx),mod1(cb+1,Ly)];
#             posTD=[mod1(ca+1,Lx),mod1(cb,Ly)];
#             posTC=[mod1(ca,Lx),mod1(cb,Ly)];
#             posBond=posTD;

#             TB=Tset[posTB[1],posTB[2]];
#             TC=Tset[posTC[1],posTC[2]];
#             TD=Tset[posTD[1],posTD[2]];
#             lambda_A_1=lambdaset1[mod1(ca,Lx),mod1(cb-1,Ly)];
#             lambda_A_2=lambdaset2[mod1(ca+1+1,Lx),mod1(cb+1,Ly)];
#             lambda_B_1=lambdaset1[mod1(ca+1,Lx),mod1(cb-1,Ly)];
#             lambda_B_3=lambdaset3[mod1(ca+1,Lx),mod1(cb+1,Ly)];
#             lambda_C_2=lambdaset2[mod1(ca+1+1,Lx),mod1(cb,Ly)];
#             lambda_C_3=lambdaset3[mod1(ca,Lx),mod1(cb,Ly)];
#             B=Bset[posBond[1],posBond[2]];

#             @tensor TB[:]:=TB[1,-2,3,-4]*lambda_B_3[-1,1]*lambda_A_2[-3,3];
#             @tensor TC[:]:=TC[1,-2,-3,4]*lambda_C_3[-1,1]*lambda_A_1[-4,4];
#             @tensor TD[:]:=TD[-1,-2,3,4]*lambda_C_2[-3,3]*lambda_B_1[-4,4];
            
#             TB, TC, TD, B, lambda_1, lambda_2, lambda_3=triangle_gate_iPESS_simplified(D_max,gates[mod1(ca,2)], TB, TC, TD, B, trun_tol);

#             lambda_A_1_inv=my_pinv(lambda_A_1);
#             lambda_A_2_inv=my_pinv(lambda_A_2);
#             lambda_B_1_inv=my_pinv(lambda_B_1);
#             lambda_B_3_inv=my_pinv(lambda_B_3);
#             lambda_C_2_inv=my_pinv(lambda_C_2);
#             lambda_C_3_inv=my_pinv(lambda_C_3);
#             @tensor TB[:]:=TB[1,-2,3,-4]*lambda_B_3_inv[-1,1]*lambda_A_2_inv[-3,3];
#             @tensor TC[:]:=TC[1,-2,-3,4]*lambda_C_3_inv[-1,1]*lambda_A_1_inv[-4,4];
#             @tensor TD[:]:=TD[-1,-2,3,4]*lambda_C_2_inv[-3,3]*lambda_B_1_inv[-4,4];
#             TB=permute(TB,(1,),(2,3,4,));
#             TC=permute(TC,(1,),(2,3,4,));
#             TD=permute(TD,(1,),(2,3,4,));


#             TB=TB/norm(TB);
#             TC=TC/norm(TC);
#             TD=TD/norm(TD);
#             B=B/norm(B);
#             lambda_1=lambda_1/norm(lambda_1);
#             lambda_2=lambda_2/norm(lambda_2);
#             lambda_3=lambda_3/norm(lambda_3);
            
#             lambdaset1[posTD[1],posTD[2]]=lambda_1;
#             lambdaset2[posTD[1],posTD[2]]=lambda_2;
#             lambdaset3[posTD[1],posTD[2]]=lambda_3;
#             Tset[posTB[1],posTB[2]]=TB;
#             Tset[posTC[1],posTC[2]]=TC;
#             Tset[posTD[1],posTD[2]]=TD;
#             Bset[posBond[1],posBond[2]]=B;


#             if mod(ct,20)==0
#                 println(space(lambda_1))
#                 println(space(lambda_2))
#                 println(space(lambda_3))
#             end
#         end
#     end




#     return Bset, Tset, lambdaset1, lambdaset2, lambdaset3
# end


# function triangle_gate_iPESS_simplified(D_max, op_LD_RD_RU, T1, T2, T3, B, trun_tol)
#     # """
#     #          M1     R1
#     #            \   /
#     #             \ /....d1
#     #              |                   T1 =  |M1, d1><D1, R1|=|M1, d1><|R1, D1   
#     #              |D1

#     #              |                B=|R2, D1><M3|
#     #             / \

#     #   M2\   /R2    M3\   /R3
#     #      \ /....d2    \ /....d3
#     #       |            |   
#     #       |D2          |D3

#     #       T2           T3

#     # T2=|M2, d2><D2, R2|=|M2, d2><|R2, D2 
#     # T3=|M3, d3><D3, R3|=|M3, d3><|R3, D3 
#     # """

#     T1=permute_neighbour_ind(T1,2,3,4);#M1, R1, d1,  D1
#     Ut1=unitary(fuse(space(T1,1)*space(T1,2)), space(T1,1)*space(T1,2));
#     @tensor T1[:]:=Ut1[-1,1,2]*T1[1,2,-2,-3];#M1_R1, d1,  D1
#     uu,ss,vv=tsvd(permute(T1,(1,),(2,3,)));
#     T1_res=uu;
#     T1_keep=ss*vv;

#     T2=permute_neighbour_ind(T2,3,4,4);#M2, d2, D2, R2
#     T2=permute_neighbour_ind(T2,2,3,4);#M2, D2, d2, R2
#     Ut2=unitary(fuse(space(T2,1)*space(T2,2)), space(T2,1)*space(T2,2));
#     @tensor T2[:]:=Ut2[-1,1,2]*T2[1,2,-2,-3];#M2_D2, d2, R2
#     uu,ss,vv=tsvd(permute(T2,(1,),(2,3,)));
#     T2_res=uu;
#     T2_keep=ss*vv;

#     T3=T3;#M3, d3, R3, D3 
#     Ut3=unitary(fuse(space(T3,3)*space(T3,4)), space(T3,3)*space(T3,4));
#     @tensor T3[:]:=T3[-1,-2,1,2]*Ut3[-3,1,2];#M3, d3, R3_D3 
#     uu,ss,vv=tsvd(permute(T3,(1,2,),(3,)));
#     T3_keep=uu*ss;
#     T3_res=vv;

#     @tensor T2_B[:]:=T2_keep[-1,-2,1]*B[1,-3,-4];     #(M2_D2, d2, R2),  (R2, D1, M3) => (M2_D2, d2, D1, M3)
#     T2_B=permute_neighbour_ind(T2_B,2,3,4);#(M2_D2, D1, d2, M3)
#     T2_B=permute_neighbour_ind(T2_B,1,2,4);#(D1, M2_D2, d2, M3)
#     @tensor T1_T2_B[:]:=T1_keep[-1,-2,1]*T2_B[1,-3,-4,-5];#(M1_R1, d1,  D1), (D1, M2_D2, d2, M3) => (M1_R1, d1, M2_D2, d2, M3)

#     @tensor T1_T2_B_T3[:]:=T1_T2_B[-1,-2,-3,-4,1]*T3_keep[1,-5,-6];#(M1_R1, d1, M2_D2, d2, M3), (M3, d3, R3_D3) => (M1_R1, d1, M2_D2, d2, d3, R3_D3)
#     T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3

#     #d2',d3',d1', d2,d3,d1
#     op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,5,6,6);#d2',d3',d1', d2,d1,d3
#     op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,4,5,6);#d2',d3',d1', d1,d2,d3
#     op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,2,3,6);#d2',d1',d3', d1,d2,d3
#     op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,1,2,6);#d1',d2',d3', d1,d2,d3
#     @tensor T1_T2_B_T3[:]:=T1_T2_B_T3[-1,-2,1,2,3,-6]*op_LD_RD_RU[-3,-4,-5,1,2,3];


#     T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, d1, M2_D2, d2, d3, R3_D3
#     if isa(space(T1,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
#         U1,S1,V1=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
#         U3,S3,V3=tsvd(T1_T2_B_T3,(1,2,3,4,),(5,6,);trunc=truncdim(D_max));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
#     elseif isa(space(T1,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
#         U1,S1,V1=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
#         U1,S1,V1=Truncations(U1,S1,V1,D_max,trun_tol);#println(norm(U1*S1*V1-M_old)/norm(M_old))
#         U3,S3,V3=tsvd(T1_T2_B_T3,(1,2,3,4,),(5,6,));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
#         U3,S3,V3=Truncations(U3,S3,V3,D_max,trun_tol);#println(norm(U3*S3*V3-M_old)/norm(M_old))
#     end


#     T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3
#     T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,1,2,6);# M2_D2, M1_R1, d1, d2, d3, R3_D3
#     T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,3,4,6);# M2_D2, M1_R1, d2, d1, d3, R3_D3
#     T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M2_D2, d2, M1_R1, d1, d3, R3_D3
#     if isa(space(T1,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
#         U2,S2,V2=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
#     elseif isa(space(T1,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
#         U2,S2,V2=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
#         U2,S2,V2=Truncations(U2,S2,V2,D_max,trun_tol);#println(norm(U2*S2*V2-M_old)/norm(M_old))
#     end

#     λ_2_new=permute(S2,(2,),(1,));

#     @tensor T2_new[:]:=T2_res[-1,1]*U2[1,-2,-3];#(M2_D2, d2, R2_new)
#     T1_B_T3=S2*V2;#(R2_new, M1_R1, d1, d3, R3_D3)
#     @tensor T1_B[:]:=T1_B_T3[-1,-2,-3,1,2]*V3'[1,2,-4];#(R2_new, M1_R1, d1, M3_new) 
#     @tensor T3_new[:]:=V3[-1,-2,1]*T3_res[1,-3];#(M3_new, d3, R3_D3)
#     λ_3_new=S3;

#     T1_B=permute_neighbour_ind(T1_B,1,2,4);#(M1_R1, R2_new, d1, M3_new) 
#     T1_B=permute_neighbour_ind(T1_B,2,3,4);#(M1_R1, d1, R2_new, M3_new) 
#     @tensor B_new[:]:=U1'[-1,1,2]*T1_B[1,2,-2,-3];#(D1_new, R2_new, M3_new) 
#     @tensor T1_new[:]:=T1_res[-1,1]*U1[1,-2,-3];#(M1_R1, d1, D1_new) 
#     λ_1_new=permute(S1,(2,),(1,));

#     #B_new: (D1_new, R2_new, M3_new) => (R2, D1, M3)
#     B_new=permute_neighbour_ind(B_new,1,2,3);#(R2_new, D1_new, M3_new)
#     B_new=permute(B_new,(1,2,),(3,));

#     #T1_new: (M1_R1, d1, D1_new) => (M1, d1, R1, D1)
#     @tensor T1_new[:]:=T1_new[1,-3,-4]*Ut1'[-1,-2,1];#(M1, R1, d1, D1_new)
#     T1_new=permute_neighbour_ind(T1_new,2,3,4);#(M1, d1, R1, D1_new)

#     #T2_new: (M2_D2, d2, R2_new) => (M2, d2, R2, D2) 
#     @tensor T2_new[:]:=T2_new[1,-3,-4]*Ut2'[-1,-2,1];#(M2, D2, d2, R2_new)
#     T2_new=permute_neighbour_ind(T2_new,2,3,4);#(M2, d2, D2, R2_new)
#     T2_new=permute_neighbour_ind(T2_new,3,4,4);#(M2, d2, R2_new, D2)

#     #T3_new: (M3_new, d3, R3_D3) => (M3, d3, R3, D3)
#     @tensor T3_new[:]:=T3_new[-1,-2,1]*Ut3'[-3,-4,1];#(M3_new, d3, R3, D3)

#     return T1_new, T2_new, T3_new, B_new, λ_1_new, λ_2_new, λ_3_new
# end
