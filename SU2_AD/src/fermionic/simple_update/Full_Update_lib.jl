function split_3Tesnsors(B1, B2, B3, T, op_LD_RD_RU)
    # """
    #          M1     R1
    #            \   /
    #             \ /....d1
    #              |                   B1 =  |M1, d1><D1, R1|=|M1, d1><|R1, D1   
    #              |D1

    #              |                T=|R2, D1><M3|
    #             / \

    #   M2\   /R2    M3\   /R3
    #      \ /....d2    \ /....d3
    #       |            |   
    #       |D2          |D3

    #       B2           B3

    # B2=|M2, d2><D2, R2|=|M2, d2><|R2, D2 
    # B3=|M3, d3><D3, R3|=|M3, d3><|R3, D3 
    # """

    @assert (length(codomain(B1))==1)&(length(domain(B1))==3)
    @assert (length(codomain(B2))==1)&(length(domain(B2))==3)
    @assert (length(codomain(B3))==1)&(length(domain(B3))==3)
    @assert (length(codomain(T))==2)&(length(domain(T))==1)

    B1=permute_neighbour_ind(B1,2,3,4);#M1, R1, d1,  D1
    uu,ss,vv=tsvd(permute(B1,(1,2,),(3,4,)));
    B1_res=uu; #M1, R1, new1
    B1_keep=ss*vv; #new1, d1,  D1
    B1_res=permute(B1_res,(1,),(2,3,));#(M1), (R1, new1)


    B2=permute_neighbour_ind(B2,3,4,4);#M2, d2, D2, R2
    B2=permute_neighbour_ind(B2,2,3,4);#M2, D2, d2, R2
    uu,ss,vv=tsvd(permute(B2,(1,2,),(3,4,)));
    B2_res=uu;#M2, D2, new2
    B2_keep=ss*vv; #new2, d2, R2
    B2_res=permute_neighbour_ind(B2_res,2,3,3);#M2, new2, D2
    B2_res=permute(B2_res,(1,),(2,3,));#(M2), (new2, D2)

    B3=B3;#M3, d3, R3, D3 
    uu,ss,vv=tsvd(permute(B3,(1,2,),(3,4,)));
    B3_keep=uu*ss; #M3, d3, new3,
    B3_res=vv;#new3, R3, D3
    B3_res=permute(B3_res,(1,),(2,3,)); #(new3), (R3, D3)

    ##################


    B1_B2_T_B3=build_triangle_from_4tensors(T,B1_keep,B2_keep,B3_keep);

    #d2',d3',d1', d2,d3,d1
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,5,6,6);#d2',d3',d1', d2,d1,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,4,5,6);#d2',d3',d1', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,2,3,6);#d2',d1',d3', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,1,2,6);#d1',d2',d3', d1,d2,d3
    @tensor op_LD_RD_RU[:]:=Up[-1,1,2,3]*op_LD_RD_RU[1,2,3,4,5,6]*Up'[4,5,6,-2];

    @tensor B1_B2_T_B3_op[:]:=B1_B2_T_B3[-1,-2,1,-4]*op_LD_RD_RU[-3,1];# new2, new1, d123, new3
    B1_B2_T_B3_op=permute(B1_B2_T_B3_op,(1,2,),(3,4,));# (new2, new1), (d123, new3)


    return B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op
end


function partial_triangle_partial_B1(Big_triangle,env_bot, T,B1_keep,B2_keep,B3_keep)
    #B1_keep: (new1, d1),  (D1)
    #Big_triangle: (new2, new1), (d123, new3)
    # gate1=swap_gate(space(B2_keep,2),space(T,2));
    # gate1=permute(gate1,(2,1,),(3,4,));
    # gate2=swap_gate(space(B2_keep,1),space(T,2));
    # gate2=permute(gate2,(2,1,),(3,4,));
    gate3=swap_gate(space(B1_keep,2),space(B2_keep,1));
    gate3=permute(gate3,(2,1,),(3,4,));
    # gate4=swap_gate(space(B1_keep,1),space(B2_keep,1));
    # gate4=permute(gate4,(2,1,),(3,4,));


    #env_bot: new_ind,new2,new3,new1
    env_bot_new=permute(env_bot,(1,3,2,4,));#new_ind,new3,new2,new1
    #apply gate4
    env_bot_new=permute_neighbour_ind(env_bot_new,3,4,4);#new_ind,new3,new1,new2

    @tensor B2_T[:]:=B2_keep[-1,-2,1]*T[1,-3,-4];     #(new2, d2, R2),  (R2, D1, M3) => (new2, d2, D1, M3)
    B2_T=permute_neighbour_ind(B2_T,2,3,4);#(new2, D1, d2, M3)
    B2_T=permute_neighbour_ind(B2_T,1,2,4);#(D1, new2, d2, M3)
    @tensor B2_T_B3[:]:=B2_T[-1,-2,-3,1]*B3_keep[1,-4,-5];#(D1, new2, d2, M3) (M3, d3, new3) -> (D1, new2, d2, d3, new3)
    UU=unitary(fuse(space(B2_T_B3,3)*space(B2_T_B3,4)), space(B2_T_B3,3)*space(B2_T_B3,4));
    @tensor B2_T_B3[:]:=B2_T_B3[-1,-2,1,2,-4]*UU[-3,1,2];#(D1, new2, d2d3, new3)

    @tensor gate3_B2_T_B3[:]:=gate3[-1,-2,-3,1]*B2_T_B3[-4,1,-5,-7];#(new2,d1),(d1,new2)  (D1, new2, d2d3, new3) -> (new2,d1,d1,  D1, d2d3, new3)


    @tensor env_bot_new_gate3_B2_T_B3[:]:=env_bot_new[-1,1,-2,2]*gate3_B2_T_B3[2,-3,-4,-5,-6,1];#new_ind,new3,new1,new2     (new2,d1,d1,  D1, d2d3, new3) -> new_ind, new1,  d1,d1,  D1, d2d3, 

    #left side
    @tensor rho[:]:=env_bot_new_gate3_B2_T_B3'[1,-1,2,-2,-3,3]*env_bot_new_gate3_B2_T_B3[1,-4,2,-5,-6,3];#(new_ind,new1,  d1,d1,  D1, d2d3),     (new_ind,new1,  d1,d1,  D1, d2d3) -> (new1,  d1,  D1),     (new1,  d1,  D1)
    
    rho=permute(rho,(1,2,3,),(4,5,6,));
    @assert norm(rho-rho')/norm(rho)<1e-10;
    rho=rho/2+rho'/2;
    eu,ev=eigh(rho);
    eu=check_positive(eu);
    rho_inv=ev*my_pinv(eu)*ev';
    

    #right side
    global Up
    @tensor Big_triangle[:]:=Big_triangle[-1,-2,1,-5]*Up'[-3,2,3,1]*UU[-4,2,3];#(new2, new1, d1, d2d3, new3)
    @tensor gate3_B2_T_B3_Big_triangle[:]:=gate3_B2_T_B3'[-1,1,-2,-3,2,-4]*Big_triangle[-5,-6,1,2,-7];#(new2,d1,d1,  D1, d2d3, new3)     (new2, new1, d1, d2d3, new3) -> (new2, d1,  D1, new3)     (new2, new1, new3)
    #env_bot_new: new_ind,new3,new1,new2
    #env_bot: new_ind,new2,new3,new1
    @tensor env_bot_gate3_B2_T_B3_Big_triangle[:]:=env_bot[-1,2,3,1]*gate3_B2_T_B3_Big_triangle[-2,-3,-4,-5,2,1,3];#(new_ind,new2,new3,new1),  (new2, d1,  D1, new3 | new2, new1, new3) -> (new_ind,  new2, d1,  D1, new3 )
    @tensor rightside[:]:=env_bot_new'[4,3,-1,2]*env_bot_gate3_B2_T_B3_Big_triangle[4,2,-2,-3,3];#(new_ind,new3,new1,new2), (new_ind,  new2, d1,  D1, new3 ) -> new1, d1, D1

    # norm1=@tensor rho[1,2,3,4,5,6]*B1_keep'[3,1,2]*B1_keep[4,5,6];#(new1,  d1,  D1 | new1,  d1,  D1) (D1 | new1, d1)    (new1, d1 | D1) 
    # norm2=@tensor rightside[1,2,3]*B1_keep'[3,1,2]; #(new1, d1, D1)  (D1 | new1, d1) -
    # println([norm1,norm2])
    
    
    @tensor B1_updated[:]:=rho_inv[-1,-2,-3,1,2,3]*rightside[1,2,3];#(new1,  d1,  D1  |  new1,  d1,  D1)    (new1, d1, D1)  -> new1,  d1,  D1 

    B1_updated=permute(B1_updated,(1,2,),(3,));#(new1,  d1),  (D1) 
    return rho, rightside, B1_updated
end




function partial_triangle_partial_B2(Big_triangle,env_bot, T,B1_keep,B2_keep,B3_keep)  
    #B1_keep: (new1, d1), (D1)
    #B2_keep: (new2, d2), (R2)
    #Big_triangle: (new2, new1), (d123, new3)
    gate1=swap_gate(space(B2_keep,2),space(T,2));
    gate1=permute(gate1,(2,1,),(3,4,));
    gate2=swap_gate(space(B2_keep,1),space(T,2));
    gate2=permute(gate2,(2,1,),(3,4,));
    gate3=swap_gate(space(B1_keep,2),space(B2_keep,1));
    gate3=permute(gate3,(2,1,),(3,4,));
    gate4=swap_gate(space(B1_keep,1),space(B2_keep,1));
    gate4=permute(gate4,(2,1,),(3,4,));




    @tensor gate4_gate3[:]:=gate4[-1,-2,-4,1]*gate3[1,-3,-5,-6];#new2,new1,d1,  new1,d1,new2
    @tensor gate4_gate3_B1[:]:=gate4_gate3[-1,-2,-3,1,2,-5]*B1_keep[1,2,-4];#(new2,new1,d1,  new1,d1,new2), (new1, d1),  (D1)  -> (new2,new1,d1,  D1, new2) 
    @tensor gate4_gate3_B1_gate2[:]:=gate4_gate3_B1[-1,-2,-3,1,2]*gate2[1,2,-4,-5];#(new2,new1,d1,  D1, new2)  -> (new2,new1,d1,  new2, D1)

    #env_bot: new_ind,new2,new3,new1
    @tensor env_bot_gate4_gate3_B1_gate2[:]:=env_bot[-1,2,-2,1]*gate4_gate3_B1_gate2[2,1,-3,-4,-5];#(new_ind,new2,new3,new1), (new2,new1,d1,  new2, D1) ->(new_ind, new3, d1, new2, D1)

    @tensor gate1_T_B3[:]:=gate1[-1,-2,-5,1]*T[-6,1,2]*B3_keep[2,-3,-4];#(D1,d2,d2,D1) (R2, D1, M3), (M3, d3, new3) -> (D1,d2,d3, new3,   d2, R2)

    @tensor leftside[:]:=env_bot_gate4_gate3_B1_gate2'[1,-1,2,-2,-3]*env_bot_gate4_gate3_B1_gate2[1,-4,2,-5,-6];#(new_ind, new3, d1, new2, D1) (new_ind, new3, d1, new2, D1) -> (, new3, , new2, D1) (, new3, , new2, D1)
    
    Uu=unitary(fuse(space(gate1_T_B3,5)*space(gate1_T_B3,6)), space(gate1_T_B3,5)*space(gate1_T_B3,6));
    @tensor gate1_T_B3[:]:=gate1_T_B3[-1,-2,-3,-4,1,2]*Uu[-5,1,2];#(D1,d2,d3, new3,   d2, R2), ->(D1,d2,d3, new3,   d2R2)

    @tensor double_gate1_T_B3[:]:=gate1_T_B3'[-1,1,2,-2,-3]*gate1_T_B3[-4,1,2,-5,-6];#(D1,d2,d3, new3,   d2R2),  (D1,d2,d3, new3,   d2R2) -> (D1, new3, d2R2,       D1, new3, d2R2)
    
    @tensor rho[:]:=leftside[2,-1,1, 4,-3,3]*double_gate1_T_B3[1,2,-2, 3,4,-4];#(new3, new2, D1,| new3, new2, D1),  (D1, new3, d2R2, |    D1, new3, d2R2)-> (new2, d2R2,  new2, d2R2)
    @tensor rho[:]:=rho[-1,1,-4,2]*Uu[1,-2,-3]*Uu'[-5,-6,2];#(new2, d2,R2,  new2, d2,R2)

    rho=permute(rho,(1,2,3,),(4,5,6,));
    @assert norm(rho-rho')/norm(rho)<1e-10
    rho=rho/2+rho'/2;
    eu,ev=eigh(rho);
    eu=check_positive(eu);
    rho_inv=ev*my_pinv(eu)*ev';

    global Up
    U21=unitary(fuse(space(Big_triangle,1)*space(Big_triangle,2)), space(Big_triangle,1)*space(Big_triangle,2));
    @tensor Big_triangle[:]:=Big_triangle[1,2,3,-6]*U21[-1,1,2]*Up'[-3,-4,-5,3];#(new2new1,  d1,d2,d3, new3) 
    @tensor rightside[:]:=gate1_T_B3'[-1,1,2,-2,-3]*Big_triangle[-4,-5,1,2,-6];#(D1,d2,d3, new3,   d2R2),  (new2new1,  d1,d2,d3, new3) -> (D1, new3,   d2R2 | new2new1,  d1, new3)

    @tensor env_bot__[:]:=env_bot[-1,1,-3,2]*U21'[1,2,-2];#(new_ind,new2,new3,new1)->(new_ind,new2new1,new3)
    @tensor rightside2[:]:=env_bot_gate4_gate3_B1_gate2'[1,-1,-2,-3,-4]*env_bot__[1,-5,-6]; #(new_ind, new3, d1, new2, D1), (new_ind,new2new1,new3) -> (, new3, d1, new2, D1,   new2new1,new3) 

    @tensor rightside[:]:=rightside[1,2,-1, 3,5,4]*rightside2[2,5,-2,1, 3,4];#(D1, new3,   d2R2 | new2new1,  d1, new3), (new3, d1, new2, D1|   new2new1,new3)  -> d2R2, new2
    @tensor rightside[:]:=rightside[1,-3]*Uu[1,-1,-2];#d2,R2, new2
    rightside=permute(rightside,(3,1,2,));#new2,d2,R2

    @tensor B2_updated[:]:=rho_inv[-1,-2,-3,1,2,3]*rightside[1,2,3];#(new2, d2,R2,  new2, d2,R2),    (new2,d2,R2)  -> new2, d2,R2 
    B2_updated=permute(B2_updated,(1,2,),(3,));#(new2, d2),  (R2) 


    # norm1=@tensor rho[1,2,3,4,5,6]*B2_keep'[3,1,2]*B2_keep[4,5,6];#(new2, d2,R2,  new2, d2,R2)  (R2|new2, d2),     (new2, d2|R2) 
    # norm2=@tensor rightside[1,2,3]*B2_keep'[3,1,2]; #(new2,d2,R2)  (R2|new2, d2) -
    # println([norm1,norm2])

    return rho,rightside,B2_updated
end




function partial_triangle_partial_B3(Big_triangle,env_bot, T,B1_keep,B2_keep,B3_keep)
    @tensor B2_T[:]:=B2_keep[-1,-2,1]*T[1,-3,-4];     #(new2, d2, R2),  (R2, D1, M3) => (new2, d2, D1, M3)
    B2_T=permute_neighbour_ind(B2_T,2,3,4);#(new2, D1, d2, M3)
    B2_T=permute_neighbour_ind(B2_T,1,2,4);#(D1, new2, d2, M3)
    @tensor B1_B2_T[:]:=B1_keep[-1,-2,1]*B2_T[1,-3,-4,-5];#(new1, d1,  D1), (D1, new2, d2, M3) => (new1, d1, new2, d2, M3)

    B1_B2_T=permute_neighbour_ind(B1_B2_T,2,3,5);# new1, new2, d1, d2, M3
    B1_B2_T=permute_neighbour_ind(B1_B2_T,1,2,5);# new2, new1, d1, d2, M3

    #env_bot: new_ind,new2,new3,new1
    @tensor env_bot_B1_B2_T[:]:=env_bot[-1,2,-2,1]*B1_B2_T[2,1,-3,-4,-5];#(new_ind,new2,new3,new1), (new2, new1, d1, d2, M3) -> (new_ind, new3, d1, d2, M3)

    @tensor Big_triangle[:]:=Big_triangle[-1,-2,1,-6]*Up'[-3,-4,-5,1];#(new2,new1,  d1,d2,d3, new3) 
    
    Id=unitary(space(B3_keep,2),space(B3_keep,2));
    @tensor leftside[:]:=env_bot_B1_B2_T'[1,-1,2,3,-2]*env_bot_B1_B2_T[1,-3,2,3,-4];#(new_ind, new3, d1, d2, M3), (new_ind, new3, d1, d2, M3) ->(new3, M3,  new3,  M3) 
    @tensor rho[:]:=leftside[-1,-2,-4,-5]*Id[-3,-6];#(new3, M3,d3,    new3,  M3,d3) 


    rho=permute(rho,(1,2,3,),(4,5,6,));
    @assert (norm(rho-rho')/norm(rho))<1e-10
    rho=rho/2+rho'/2;
    eu,ev=eigh(rho);
    eu=check_positive(eu);
    rho_inv=ev*my_pinv(eu)*ev';


    @tensor env_bot_Big_triangle[:]:=env_bot[-1,2,3,1]*Big_triangle[2,1,-2,-3,-4,3];#(new_ind,new2,new3,new1), (new2,new1,  d1,d2,d3, new3) -> (new_ind,  d1,d2,d3)
    @tensor rightside[:]:=env_bot_B1_B2_T'[1,-1,2,3,-2]*env_bot_Big_triangle[1,2,3,-3];#(new_ind, new3, d1, d2, M3), (new_ind,  d1,d2,d3) -> (new3, M3,d3) 
    
    @tensor B3_updated[:]:=rho_inv[-1,-2,-3,1,2,3]*rightside[1,2,3];#(new3, M3,d3,    new3,  M3,d3) ,    (new3, M3,d3)  -> (new3, M3,d3)
    B3_updated=permute(B3_updated,(2,3,),(1,));#(M3, d3), (new3)

    # norm1=@tensor rho[1,2,3,4,5,6]*B3_keep'[1,2,3]*B3_keep[5,6,4];#(new3, M3,d3,    new3,  M3,d3)   (new3)(M3, d3),    (M3, d3)(new3)
    # norm2=@tensor rightside[1,2,3]*B3_keep'[1,2,3]; #(new3, M3,d3)   (new3)(M3, d3)
    # println([norm1,norm2])

    return rho,rightside,B3_updated
end

function partial_triangle_partial_T(Big_triangle,env_bot, T,B1_keep,B2_keep,B3_keep)
    #T:(R2, D1), (M3)
    #B1_keep: (new1, d1), (D1)
    #B2_keep: (new2, d2), (R2)
    #B3_keep: (M3, d3), (new3)
    #Big_triangle: (new2, new1), (d123, new3)
    gate1=swap_gate(space(B2_keep,2),space(T,2));
    gate1=permute(gate1,(2,1,),(3,4,));
    gate2=swap_gate(space(B2_keep,1),space(T,2));
    gate2=permute(gate2,(2,1,),(3,4,));
    gate3=swap_gate(space(B1_keep,2),space(B2_keep,1));
    gate3=permute(gate3,(2,1,),(3,4,));
    gate4=swap_gate(space(B1_keep,1),space(B2_keep,1));
    gate4=permute(gate4,(2,1,),(3,4,));




    @tensor gate4_gate3[:]:=gate4[-1,-2,-4,1]*gate3[1,-3,-5,-6];#new2,new1,d1,  new1,d1,new2
    @tensor gate4_gate3_B1[:]:=gate4_gate3[-1,-2,-3,1,2,-5]*B1_keep[1,2,-4];#(new2,new1,d1,  new1,d1,new2), (new1, d1),  (D1)  -> (new2,new1,d1,  D1, new2) 
    @tensor gate4_gate3_B1_gate2[:]:=gate4_gate3_B1[-1,-2,-3,1,2]*gate2[1,2,-4,-5];#(new2,new1,d1,  D1, new2)  -> (new2,new1,d1,  new2, D1)
    @tensor gate1_B2[:]:=gate1[-2,-3,1,-5]*B2_keep[-1,1,-4];#   new2,D1,d2,    R2, D1
    @tensor gate4_gate3_B1_gate2_gate1_B2[:]:=gate4_gate3_B1_gate2[-1,-2,-3,1,2]*gate1_B2[1,2,-4,-5,-6];#(new2,new1,d1,  new2, D1), (new2,D1,d2,  R2, D1) ->(new2,new1,d1,      d2, R2, D1)  

    @tensor env_bot_new[:]:=env_bot[-1,1,-2,2]*gate4_gate3_B1_gate2_gate1_B2[1,2,-3,-4,-5,-6];#(new_ind,new2,new3,new1), (new2,new1,d1,    d2, R2, D1) -> (new_ind,new3,      d1, d2, R2, D1)
    @tensor leftside[:]:=env_bot_new'[1,-1,2,3,-2,-3]*env_bot_new[1,-4,2,3,-5,-6];#(new_ind,new3,   d1, d2, R2, D1),(new_ind,new3,   d1, d2, R2, D1) -> (new3, R2, D1,    new3, R2, D1)
    @tensor rho[:]:=leftside[1,-1,-2,2,-4,-5]*B3_keep'[1,-3,3]*B3_keep[-6,3,2];#(new3, R2, D1,    new3, R2, D1), (new3)(M3, d3)', (M3, d3)(new3) ->(R2,D1,M3,  R2,D1,M3)

    rho=permute(rho,(1,2,3,),(4,5,6,));
    @assert (norm(rho-rho')/norm(rho))<1e-10
    rho=rho/2+rho'/2;
    eu,ev=eigh(rho);
    eu=check_positive(eu);
    rho_inv=ev*my_pinv(eu)*ev';

    global Up
    @tensor Big_triangle[:]:=Big_triangle[-1,-2,1,-6]*Up'[-3,-4,-5,1];#(new2,new1,  d1,d2,d3, new3) 
    @tensor env_bot_Big_triangle[:]:=env_bot[-1,2,3,1]*Big_triangle[2,1,-2,-3,-4,3];#(new_ind,new2,new3,new1), (new2,new1,  d1,d2,d3, new3) -> (new_ind,  d1,d2,d3)
    @tensor rightside[:]:=env_bot_new'[1,-1,2,3,-2,-3]*env_bot_Big_triangle[1,2,3,-4];#(new_ind,new3,  d1, d2, R2, D1), (new_ind,  d1,d2,d3) -> (new3, R2, D1, d3)
    @tensor rightside[:]:=rightside[1,-1,-2,2]*B3_keep'[1,-3,2];# (new3, R2, D1, d3),  (new3)(M3, d3) -> R2, D1, M3






    @tensor T_updated[:]:=rho_inv[-1,-2,-3,1,2,3]*rightside[1,2,3];#(R2,D1,M3,  R2,D1,M3),    R2, D1, M3  -> R2, D1, M3
    T_updated=permute(T_updated,(1,2,),(3,));#(R2, D1), (M3)


    # norm1=@tensor rho[1,2,3,4,5,6]*T'[3,1,2]*T[4,5,6];#(R2,D1,M3,  R2,D1,M3)   (M3)(R2, D1),     (R2, D1)(M3)
    # norm2=@tensor rightside[1,2,3]*T'[3,1,2]; #(R2, D1, M3)  (M3)(R2, D1)
    # println([norm1,norm2])

    return rho,rightside,T_updated


end


function sweep_iteration(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new)
    ####################################
    T1_left,T1_right,T1_new=partial_triangle_partial_B1(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    T2_left,T2_right,T2_new=partial_triangle_partial_B2(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    T3_left,T3_right,T3_new=partial_triangle_partial_B3(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
    B_left,B_right,B_new=partial_triangle_partial_T(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);

    #T1_new: (new1, d1),  (D1) 
    #T2_new: (new2, d2),  (R2) 
    #T3_new: (M3, d3), (new3)
    #B_new: (R2, D1), (M3)

    #set the gauge:
    @tensor T2_B[:]:=T2_new[-1,-2,1]*B_new[1,-3,-4];#(new2, d2,    D1,M3)
    u,s,v=tsvd(permute(T2_B,(1,2,),(3,4,));trunc=truncdim(dim(space(T2_new,3))));
    T2_new=u*sqrt(s);
    B_new=sqrt(s)*v;
    B_new=permute(B_new,(1,2,),(3,));

    @tensor B_T3[:]:=B_new[-1,-2,1]*T3_new[1,-3,-4];#(R2,D1,  d3,new3)
    u,s,v=tsvd(permute(B_T3,(1,2,),(3,4,));trunc=truncdim(dim(space(T3_new,1))));
    B_new=u*sqrt(s);
    T3_new=sqrt(s)*v;
    T3_new=permute(T3_new,(1,2,),(3,));

    B_new=permute_neighbour_ind(B_new,1,2,3);#(D1, R2, M3)
    @tensor T1_B[:]:=T1_new[-1,-2,1]*B_new[1,-3,-4]; #(new1, d1,  R2, M3)
    u,s,v=tsvd(permute(T1_B,(1,2,),(3,4,));trunc=truncdim(dim(space(T1_new,3))));
    T1_new=u*sqrt(s);
    B_new=sqrt(s)*v;
    B_new=permute_neighbour_ind(B_new,1,2,3);
    B_new=permute(B_new,(1,2,),(3,));


    return B_new,T1_new,T2_new,T3_new
end



function sweep_optimizations(n_sweep,B1_B2_T_B3_op,env_top,env_bot, B_new,T1_new,T2_new,T3_new)
    
    ov_history=zeros(n_sweep);
    for ci=1:n_sweep
        B_new,T1_new,T2_new,T3_new=sweep_iteration(B1_B2_T_B3_op,env_bot, B_new,T1_new,T2_new,T3_new);
        big_T_compressed_opt=build_triangle_from_4tensors(B_new,T1_new,T2_new,T3_new);

        ov12=get_overlap_env(env_top,env_bot,big_T_compressed_opt',B1_B2_T_B3_op);
        ov11=get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
        ov22=get_overlap_env(env_top,env_bot,big_T_compressed_opt',big_T_compressed_opt);
        ov=ov12/sqrt(ov11*ov22);
        print(string(norm(ov))*" , ")
        ov_history[ci]=norm(ov);
        if ((ci>4)&& (abs(ov_history[ci]/ov_history[ci-1]-1)<1e-7))|(ci==n_sweep);
            print("\n")
            return B_new,T1_new,T2_new,T3_new,big_T_compressed_opt, norm(ov)
        end
    end

    
end