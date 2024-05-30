
# function build_triangle_from_4tensors(T,B1_keep,B2_keep,B3_keep)
#     @tensor B2_T[:]:=B2_keep[-1,-2,1]*T[1,-3,-4];     #(new2, d2, R2),  (R2, D1, M3) => (new2, d2, D1, M3)
#     B2_T=permute_neighbour_ind(B2_T,2,3,4);#(new2, D1, d2, M3)
#     B2_T=permute_neighbour_ind(B2_T,1,2,4);#(D1, new2, d2, M3)
#     @tensor B1_B2_T[:]:=B1_keep[-1,-2,1]*B2_T[1,-3,-4,-5];#(new1, d1,  D1), (D1, new2, d2, M3) => (new1, d1, new2, d2, M3)

#     @tensor B1_B2_T_B3[:]:=B1_B2_T[-1,-2,-3,-4,1]*B3_keep[1,-5,-6];#(new1, d1, new2, d2, M3), (M3, d3, new3) => (new1, d1, new2, d2, d3, new3)
#     B1_B2_T_B3=permute_neighbour_ind(B1_B2_T_B3,2,3,6);# new1, new2, d1, d2, d3, new3

#     Up=unitary(fuse(space(B1_B2_T_B3,3)*space(B1_B2_T_B3,4)*space(B1_B2_T_B3,5)), space(B1_B2_T_B3,3)*space(B1_B2_T_B3,4)*space(B1_B2_T_B3,5));
#     global Up
#     @tensor B1_B2_T_B3[:]:=B1_B2_T_B3[-1,-2,1,2,3,-4]*Up[-3,1,2,3];# new1, new2, d123, new3

#     B1_B2_T_B3=permute_neighbour_ind(B1_B2_T_B3,1,2,4);# new2, new1, d123, new3
#     B1_B2_T_B3=permute(B1_B2_T_B3,(1,2,),(3,4,));# (new2, new1), (d123, new3)

#     #########################################
    
#     # big_T_compressed=permute_neighbour_ind(B_new,1,2,3);#(D1_new, R2_new, M3_new)
#     # @tensor big_T_compressed[:]:=T1_new[-1,-2,1]*big_T_compressed[1,-3,-4];#(M1_R1, d1, R2_new, M3_new) 
#     # big_T_compressed=permute_neighbour_ind(big_T_compressed,2,3,4);#(M1_R1,R2_new, d1,  M3_new) 
#     # big_T_compressed=permute_neighbour_ind(big_T_compressed,1,2,4);#(R2_new, M1_R1, d1,  M3_new) 
#     # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,-3,1]*T3_new[1,-4,-5];#(R2_new, M1_R1, d1,  d3, R3_D3)
#     # @tensor big_T_compressed[:]:=T2_new[-1,-2,1]*big_T_compressed[1,-3,-4,-5,-6];#(M2_D2, d2,  M1_R1, d1,  d3, R3_D3)

#     # big_T_compressed=permute_neighbour_ind(big_T_compressed,2,3,6);#(M2_D2,  M1_R1, d2, d1,  d3, R3_D3)
#     # big_T_compressed=permute_neighbour_ind(big_T_compressed,3,4,6);#(M2_D2,  M1_R1, d1, d2,  d3, R3_D3)
#     # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,1,2,3,-4]*Up[-3,1,2,3];#(new2, new1,  d123, new3)
#     # big_T_compressed=permute(big_T_compressed,(1,2,),(3,4,))#(new2, new1), (d123, new3)
    

#     return B1_B2_T_B3
# end

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
    eu,ev=eigen(rho,(1,2,3,),(4,5,6,));
    

    #right side
    global Up
    @tensor Big_triangle[:]:=Big_triangle[-1,-2,1,-5]*Up'[-3,2,3,1]*UU[-4,2,3];#(new2, new1, d1, d2d3, new3)
    @tensor gate3_B2_T_B3_Big_triangle[:]:=gate3_B2_T_B3'[-1,1,-2,-3,2,-4]*Big_triangle[-5,-6,1,2,-7];#(new2,d1,d1,  D1, d2d3, new3)     (new2, new1, d1, d2d3, new3) -> (new2, d1,  D1, new3)     (new2, new1, new3)
    #env_bot_new: new_ind,new3,new1,new2
    #env_bot: new_ind,new2,new3,new1
    @tensor env_bot_gate3_B2_T_B3_Big_triangle[:]:=env_bot[-1,2,3,1]*gate3_B2_T_B3_Big_triangle[-2,-3,-4,-5,2,1,3];#(new_ind,new2,new3,new1),  (new2, d1,  D1, new3 | new2, new1, new3) -> (new_ind,  new2, d1,  D1, new3 )
    @tensor rightside[:]:=env_bot_new'[4,3,-1,2]*env_bot_gate3_B2_T_B3_Big_triangle[4,2,-2,-3,3];#(new_ind,new3,new1,new2), (new_ind,  new2, d1,  D1, new3 ) -> new1, d1, D1

    norm1=@tensor rho[1,2,3,4,5,6]*B1_keep'[3,1,2]*B1_keep[4,5,6];#(new1,  d1,  D1 | new1,  d1,  D1) (D1 | new1, d1)    (new1, d1 | D1) 
    norm2=@tensor rightside[1,2,3]*B1_keep'[3,1,2]; #(new1, d1, D1)  (D1 | new1, d1) -

    println([norm1,norm2])
    rho_inv=ev*my_pinv(eu)*ev';
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
    eu,ev=eigen(rho,(1,2,3,),(4,5,6,));
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


    norm1=@tensor rho[1,2,3,4,5,6]*B2_keep'[3,1,2]*B2_keep[4,5,6];#(new2, d2,R2,  new2, d2,R2)  (R2|new2, d2),     (new2, d2|R2) 
    norm2=@tensor rightside[1,2,3]*B2_keep'[3,1,2]; #(new2,d2,R2)  (R2|new2, d2) -
    println([norm1,norm2])

    return rho,rightside,B2_updated
end




function partial_triangle_partial_B2(Big_triangle,env_bot, T,B1_keep,B2_keep,B3_keep)
    @tensor B2_T[:]:=B2_keep[-1,-2,1]*T[1,-3,-4];     #(new2, d2, R2),  (R2, D1, M3) => (new2, d2, D1, M3)
    B2_T=permute_neighbour_ind(B2_T,2,3,4);#(new2, D1, d2, M3)
    B2_T=permute_neighbour_ind(B2_T,1,2,4);#(D1, new2, d2, M3)
    @tensor B1_B2_T[:]:=B1_keep[-1,-2,1]*B2_T[1,-3,-4,-5];#(new1, d1,  D1), (D1, new2, d2, M3) => (new1, d1, new2, d2, M3)

    B1_B2_T=permute_neighbour_ind(B1_B2_T,2,3,5);# new1, new2, d1, d2, M3
    B1_B2_T=permute_neighbour_ind(B1_B2_T,1,2,5);# new2, new1, d1, d2, M3

    #env_bot: new_ind,new2,new3,new1
    @tensor env_bot_B1_B2_T[:]:=env_bot[]*B1_B2_T[]


    



    @tensor B1_B2_T_B3[:]:=B1_B2_T[-1,-2,-3,-4,1]*B3_keep[1,-5,-6];#(new1, d1, new2, d2, M3), (M3, d3, new3) => (new1, d1, new2, d2, d3, new3)
    B1_B2_T_B3=permute_neighbour_ind(B1_B2_T_B3,2,3,6);# new1, new2, d1, d2, d3, new3

    Up=unitary(fuse(space(B1_B2_T_B3,3)*space(B1_B2_T_B3,4)*space(B1_B2_T_B3,5)), space(B1_B2_T_B3,3)*space(B1_B2_T_B3,4)*space(B1_B2_T_B3,5));
    global Up
    @tensor B1_B2_T_B3[:]:=B1_B2_T_B3[-1,-2,1,2,3,-4]*Up[-3,1,2,3];# new1, new2, d123, new3

    B1_B2_T_B3=permute_neighbour_ind(B1_B2_T_B3,1,2,4);# new2, new1, d123, new3
    B1_B2_T_B3=permute(B1_B2_T_B3,(1,2,),(3,4,));# (new2, new1), (d123, new3)

    #########################################
    


    return 
end