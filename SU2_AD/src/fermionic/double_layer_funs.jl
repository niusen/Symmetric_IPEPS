function build_double_layer_NoSwap_extra_leg(Ap,A)
    #The last index of A tensor is an extra virtual index, such as that comes from decomposition of Heisenberg interaction

    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5,6));
    
    # U_L=unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    # U_D=unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    # U_R=inv(U_L);
    # U_U=inv(U_D);

    # U_Lp=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    # U_Dp=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    # U_Rp=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    # U_Up=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # println(norm(U_R-U_Rp)/norm(U_R))
    # println(norm(U_L-U_Lp)/norm(U_L))
    # println(norm(U_D-U_Dp)/norm(U_D))
    # println(norm(U_U-U_Up)/norm(U_U))

    U_L=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    U_D=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    U_R=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    U_U=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uMp,sMp,vMp=tsvd(Ap);
    uMp=uMp*sMp;
    uM,sM,vM=tsvd(A);
    uM=uM*sM;

    uMp=permute(uMp,(1,2,3,),())
    uM=permute(uM,(1,2,3,),())
    Vp=space(uMp,3);
    V=space(vM,1);
    U=unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

    @tensor double_LD[:]:=uMp[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vMp=permute(vMp,(1,2,3,4,),());
    vM=permute(vM,(1,2,3,4,5,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5,-6];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2,-6];

    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))


    double_RU=permute(double_RU,(1,4,5,6,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,5,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,4,3,));
    AA_fused=double_LD*double_RU;


    ##########################


    return AA_fused, U_L,U_D,U_R,U_U
end



function build_double_layer_NoSwap_op(A1,O1,has_extra_leg)
    A1=deepcopy(A1)
    A1_origin=deepcopy(A1)



    if has_extra_leg
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1,-6]#the last index is extra
        A1_new=A1
        A1_double,_,_,_,_=build_double_layer_NoSwap_extra_leg(A1_origin',A1_new)
    else
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1]
        A1_new=A1
        A1_double,_,_,_,_=build_double_layer_NoSwap(A1_origin',A1_new)
    end

    return A1_double
end






function build_double_layer_swap_op(A1,O1,has_extra_leg)
    A1=deepcopy(A1)
    A1_origin=deepcopy(A1)



    if has_extra_leg
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1,-6]#the last index is extra
        A1_new=A1
        A1_double,_,_,_,_=build_double_layer_swap_extra_leg(A1_origin',A1_new)
    else
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*O1[-5,1]
        A1_new=A1
        A1_double,_,_,_,_=build_double_layer_swap(A1_origin',A1_new)
    end

    return A1_double
end


function build_double_layer_swap_op(A1,A_mid,A2,H_term)
    A1=deepcopy(A1)
    A2=deepcopy(A2)
    A_mid=deepcopy(A_mid);
    A1_origin=deepcopy(A1)
    A2_origin=deepcopy(A2)
    A_mid_origin=deepcopy(A_mid);



    if H_term["p1"]%2==1 # has extra leg
        #the first index of O is dummy
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*H_term["O1"][-6,-5,1]
        @tensor A2[:]:= A2[-1,-2,-3,-4,1]*H_term["O2"][-6,-5,1]
        O_string=unitary(space(H_term["O1"],1),space(H_term["O1"],1));

        if H_term["direction"]=="x"
            @assert H_term["sign1"]==[1,1,1,1,0];
            @assert H_term["sign2"]==[0,0,0,1,0];
            
            gate=parity_gate(A1,1); @tensor A1[:]:=A1[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=parity_gate(A1,2); @tensor A1[:]:=A1[-1,1,-3,-4,-5,-6]*gate[-2,1];
            #gate=parity_gate(A1,3); @tensor A1[:]:=A1[-1,-2,1,-4,-5,-6]*gate[-3,1];
            gate=parity_gate(A1,4); @tensor A1[:]:=A1[-1,-2,-3,1,-5,-6]*gate[-4,1];

            gate=parity_gate(A_mid,2); @tensor A_mid[:]:=A_mid[-1,1,-3,-4,-5]*gate[-2,1];


            gate=parity_gate(A2,1); @tensor A2[:]:=A2[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=parity_gate(A2,4); @tensor A2[:]:=A2[-1,-2,-3,1,-5,-6]*gate[-4,1];
        end

        @assert H_term["ind1"]==3
        @assert H_term["ind2"]==1

        U=unitary(fuse(space(A1,3)⊗space(A1,6)), space(A1,3)⊗space(A1,6)); 
        @tensor A1_new[:]:=A1[-1,-2,1,-4,-5,2]*U[-3,1,2];
        @tensor A_mid_new[:]:=A_mid[1,-2,3,-4,-5]*O_string[4,2]*U'[1,2,-1]*U[-3,3,4];
        @tensor A2_new[:]:=A2[1,-2,-3,-4,-5,2]*U'[1,2,-1];

        A1_double,_,_,_,_=build_double_layer_swap(A1_origin',A1_new)
        A_mid_double,_,_,_,_=build_double_layer_swap(A_mid_origin',A_mid_new)
        A2_double,_,_,_,_=build_double_layer_swap(A2_origin',A2_new)

        return A1_double,A_mid_double,A2_double



    else # No extra leg
        @tensor A1[:]:= A1[-1,-2,-3,-4,1]*H_term["O1"][-5,1]
        @tensor A2[:]:= A2[-1,-2,-3,-4,1]*H_term["O2"][-5,1]
        A1_new=A1
        A2_new=A2

        A1_double,_,_,_,_=build_double_layer_swap(A1_origin',A1_new)
        A2_double,_,_,_,_=build_double_layer_swap(A2_origin',A2_new)

        return A1_double,nothing,A2_double
    end

end





function build_double_layer_swap_extra_leg(Ap,A)
    #The last index of A tensor is an extra virtual index, such as that comes from decomposition of Heisenberg interaction


    gate=swap_gate(Ap,1,4); @tensor Ap[:]:=Ap[1,-2,-3,2,-5]*gate[-1,-4,1,2];  
    gate=swap_gate(Ap,2,3); @tensor Ap[:]:=Ap[-1,1,2,-4,-5]*gate[-2,-3,1,2];  
    gate=parity_gate(Ap,4); @tensor Ap[:]:=Ap[-1,-2,-3,1,-5]*gate[-4,1];
    gate=parity_gate(Ap,2); @tensor Ap[:]:=Ap[-1,1,-3,-4,-5]*gate[-2,1];


    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5,6));
    
    # U_L=unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    # U_D=unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    # U_R=inv(U_L);
    # U_U=inv(U_D);

    # U_Lp=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    # U_Dp=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    # U_Rp=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    # U_Up=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # println(norm(U_R-U_Rp)/norm(U_R))
    # println(norm(U_L-U_Lp)/norm(U_L))
    # println(norm(U_D-U_Dp)/norm(U_D))
    # println(norm(U_U-U_Up)/norm(U_U))

    U_L=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    U_D=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    U_R=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    U_U=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uMp,sMp,vMp=tsvd(Ap);
    uMp=uMp*sMp;
    uM,sM,vM=tsvd(A);
    uM=uM*sM;

    uMp=permute(uMp,(1,2,3,),())
    uM=permute(uM,(1,2,3,),())
    Vp=space(uMp,3);
    V=space(vM,1);
    U=unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

    @tensor double_LD[:]:=uMp[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vMp=permute(vMp,(1,2,3,4,),());
    vM=permute(vM,(1,2,3,4,5,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5,-6];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2,-6];

    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))


    double_RU=permute(double_RU,(1,4,5,6,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,5,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,4,3,));
    AA_fused=double_LD*double_RU;


    ##########################
    @tensor U_LU[:]:=U_L'[-1,-2,-5]*U_U'[-6,-3,-4];
    gate1=swap_gate(U_LU,1,4);
    gate2=swap_gate(U_LU,3,4);
    @tensor U_LU[:]:=U_LU[1,-2,-3,2,-5,-6]*gate1[-1,-4,1,2];
    @tensor U_LU[:]:=U_LU[-1,-2,1,2,-5,-6]*gate2[-3,-4,1,2];
    @tensor U_LU[:]:=U_LU[1,2,3,4,-3,-4]*U_L[-1,1,2]*U_U[3,4,-2];
    @tensor AA_fused[:]:=AA_fused[1,-2,-3,2,-5]*U_LU[-1,-4,1,2];


    @tensor U_DR[:]:=U_D'[-1,-2,-5]*U_R'[-6,-3,-4];
    gate1=swap_gate(U_DR,1,2);
    gate2=swap_gate(U_DR,1,4);
    @tensor U_DR[:]:=U_DR[1,2,-3,-4,-5,-6]*gate1[-1,-2,1,2];
    @tensor U_DR[:]:=U_DR[1,-2,-3,2,-5,-6]*gate2[-1,-4,1,2];

    @tensor U_DR[:]:=U_DR[1,2,3,4,-3,-4]*U_D[-1,1,2]*U_R[3,4,-2];
    @tensor AA_fused[:]:=AA_fused[-1,1,2,-4,-5]*U_DR[-2,-3,1,2];

    return AA_fused, U_L,U_D,U_R,U_U
end




# function build_double_layer_swap_extra_leg(Ap,A)
#     #The last index of A tensor is an extra virtual index, such as C^dag C


#     gate=swap_gate(Ap,1,4); @tensor Ap[:]:=Ap[1,-2,-3,2,-5]*gate[-1,-4,1,2];  
#     gate=swap_gate(Ap,2,3); @tensor Ap[:]:=Ap[-1,1,2,-4,-5]*gate[-2,-3,1,2];  
#     gate=parity_gate(Ap,4); @tensor Ap[:]:=Ap[-1,-2,-3,1,-5]*gate[-4,1];
#     gate=parity_gate(Ap,2); @tensor Ap[:]:=Ap[-1,1,-3,-4,-5]*gate[-2,1];


#     Ap=permute(Ap,(1,2,),(3,4,5))
#     A=permute(A,(1,2,),(3,4,5,6));
    
#     # U_L=unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
#     # U_D=unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
#     # U_R=inv(U_L);
#     # U_U=inv(U_D);

#     # U_Lp=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
#     # U_Dp=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
#     # U_Rp=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
#     # U_Up=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

#     # println(norm(U_R-U_Rp)/norm(U_R))
#     # println(norm(U_L-U_Lp)/norm(U_L))
#     # println(norm(U_D-U_Dp)/norm(U_D))
#     # println(norm(U_U-U_Up)/norm(U_U))

#     U_L=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
#     U_D=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
#     U_R=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
#     U_U=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

#     # display(space(U_L))
#     # display(space(U_D))
#     # display(space(U_R))
#     # display(space(U_D))

#     uMp,sMp,vMp=tsvd(Ap);
#     uMp=uMp*sMp;
#     uM,sM,vM=tsvd(A);
#     uM=uM*sM;

#     uMp=permute(uMp,(1,2,3,),())
#     uM=permute(uM,(1,2,3,),())
#     Vp=space(uMp,3);
#     V=space(vM,1);
#     U=unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

#     @tensor double_LD[:]:=uMp[-1,-2,1]*U'[1,-3,-4];
#     @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

#     vMp=permute(vMp,(1,2,3,4,),());
#     vM=permute(vM,(1,2,3,4,5,),());

#     @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5,-6];
#     @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2,-6];

#     #display(space(double_RU))

#     double_LD=permute(double_LD,(1,2,),(3,4,5,));
#     double_LD=U_L*double_LD;
#     double_LD=permute(double_LD,(2,3,),(1,4,));
#     double_LD=U_D*double_LD;
#     double_LD=permute(double_LD,(2,1,),(3,));
#     #display(space(double_LD))


#     double_RU=permute(double_RU,(1,4,5,6,),(2,3,));
#     double_RU=double_RU*U_R;
#     double_RU=permute(double_RU,(1,5,4,),(2,3,));
#     double_RU=double_RU*U_U;
#     double_LD=permute(double_LD,(1,2,),(3,));
#     double_RU=permute(double_RU,(1,),(2,4,3,));
#     AA_fused=double_LD*double_RU;


#     ##########################
#     @tensor U_LU[:]:=U_L'[-1,-2,-5]*U_U'[-6,-3,-4];
#     gate1=swap_gate(U_LU,1,4);
#     gate2=swap_gate(U_LU,3,4);
#     @tensor U_LU[:]:=U_LU[1,-2,-3,2,-5,-6]*gate1[-1,-4,1,2];
#     @tensor U_LU[:]:=U_LU[-1,-2,1,2,-5,-6]*gate2[-3,-4,1,2];
#     @tensor U_LU[:]:=U_LU[1,2,3,4,-3,-4]*U_L[-1,1,2]*U_U[3,4,-2];
#     @tensor AA_fused[:]:=AA_fused[1,-2,-3,2,-5]*U_LU[-1,-4,1,2];


#     @tensor U_DR[:]:=U_D'[-1,-2,-5]*U_R'[-6,-3,-4];
#     gate1=swap_gate(U_DR,1,2);
#     gate2=swap_gate(U_DR,1,4);
#     @tensor U_DR[:]:=U_DR[1,2,-3,-4,-5,-6]*gate1[-1,-2,1,2];
#     @tensor U_DR[:]:=U_DR[1,-2,-3,2,-5,-6]*gate2[-1,-4,1,2];

#     @tensor U_DR[:]:=U_DR[1,2,3,4,-3,-4]*U_D[-1,1,2]*U_R[3,4,-2];
#     @tensor AA_fused[:]:=AA_fused[-1,1,2,-4,-5]*U_DR[-2,-3,1,2];

#     return AA_fused, U_L,U_D,U_R,U_U
# end