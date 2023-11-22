using KrylovKit
using LinearAlgebra
using TensorKit


function create_H_term(O1,O2,direction,is_odd)
    if direction=="x"
        if is_odd
            #c1dag c2
            sign1=[1,1,1,1,0];
            sign2=[0,0,0,1,0];
            ind1=3;#index p
            ind2=1;#index p
            p1=1;
            p2=1;
        else
            # n1 n2 
            sign1=[0,0,0,0,0];
            sign2=[0,0,0,0,0];
            ind1=3;#index p
            ind2=1;#index p
            p1=0;
            p2=0;
        end

        H_term=Dict([("direction","x"),("O1", O1), ("O2", O2), ("sign1",sign1), ("sign2",sign2), ("ind1",ind1), ("ind2",ind2), ("p1",p1), ("p2",p2)]);
    end
    return H_term

end

function Hamiltonians(M,U_phy1,U_phy2)
    if M==1

        Vdummy=ℂ[U1Irrep](-1=>1);
        V=ℂ[U1Irrep](0=>1,1=>1);

        Id=[1 0;0 1];
        sm=[0 1;0 0]; sp=[0 0;1 0]; sz=[1 0; 0 -1]; occu=[0 0; 0 1];
        
        @tensor Ident[:]:=Id[-1,-3]*Id[-2,-4];
        Ident=TensorMap(Ident,  V ⊗ V ← V ⊗ V);

        @tensor NA[:]:=occu[-1,-3]*Id[-2,-4];
        NA=TensorMap(NA,  V ⊗ V ← V ⊗ V);
        
        @tensor NB[:]:=Id[-1,-3]*occu[-2,-4];
        NB=TensorMap(NB,  V ⊗ V ← V ⊗ V);

        @tensor NANB[:]:=occu[-1,-3]*occu[-2,-4];
        NANB=TensorMap(NANB,  V ⊗ V ← V ⊗ V);

        @tensor cAdag[:]:=sp[-1,-3]*Id[-2,-4];
        CAdag=zeros(1,2,2,2,2);
        CAdag[1,:,:,:,:]=cAdag;
        CAdag=TensorMap(CAdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);

        @tensor cBdag[:]:=sz[-1,-3]*sp[-2,-4];
        CBdag=zeros(1,2,2,2,2);
        CBdag[1,:,:,:,:]=cBdag;
        CBdag=TensorMap(CBdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);

        @tensor cA[:]:=sm[-1,-3]*Id[-2,-4];
        CA=zeros(1,2,2,2,2);
        CA[1,:,:,:,:]=cA;
        CA=TensorMap(CA, Vdummy' ⊗ V ⊗ V ← V ⊗ V);

        @tensor cB[:]:=sz[-1,-3]*sm[-2,-4];
        CB=zeros(1,2,2,2,2);
        CB[1,:,:,:,:]=cB;
        CB=TensorMap(CB, Vdummy' ⊗ V ⊗ V ← V ⊗ V);



        @tensor Ident[:]:=Ident[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Ident[:]:=Ident[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor NANB[:]:=NANB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NANB[:]:=NANB[3,4]*U_phy2[-1,3,1,2]*U_phy2'[4,1,2,-2];

        @tensor CAdag[:]:=CAdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CAdag[:]:=CAdag[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CBdag[:]:=CBdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CBdag[:]:=CBdag[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CA[:]:=CA[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CA[:]:=CA[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];

        @tensor CB[:]:=CB[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CB[:]:=CB[-1,3,4]*U_phy2[-2,3,1,2]*U_phy2'[4,1,2,-3];
    elseif M==2
        Vdummy=ℂ[U1Irrep](-1=>1)';
        V=ℂ[U1Irrep](0=>1,1=>1)';
    
        Id=[1 0;0 1];
        sm=[0 1;0 0]; sp=[0 0;1 0]; sz=[1 0; 0 -1]; occu=[0 0; 0 1];
        
        @tensor Ident[:]:=Id[-1,-3]*Id[-2,-4];
        Ident=TensorMap(Ident,  V ⊗ V ← V ⊗ V);
    
        @tensor NA[:]:=occu[-1,-3]*Id[-2,-4];
        NA=TensorMap(NA,  V ⊗ V ← V ⊗ V);
        
        @tensor NB[:]:=Id[-1,-3]*occu[-2,-4];
        NB=TensorMap(NB,  V ⊗ V ← V ⊗ V);
    
        @tensor NANB[:]:=occu[-1,-3]*occu[-2,-4];
        NANB=TensorMap(NANB,  V ⊗ V ← V ⊗ V);
    
        @tensor cAdag[:]:=sp[-1,-3]*Id[-2,-4];
        CAdag=zeros(1,2,2,2,2);
        CAdag[1,:,:,:,:]=cAdag;
        CAdag=TensorMap(CAdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);
    
        @tensor cBdag[:]:=sz[-1,-3]*sp[-2,-4];
        CBdag=zeros(1,2,2,2,2);
        CBdag[1,:,:,:,:]=cBdag;
        CBdag=TensorMap(CBdag, Vdummy ⊗ V ⊗ V ← V ⊗ V);
    
        @tensor cA[:]:=sm[-1,-3]*Id[-2,-4];
        CA=zeros(1,2,2,2,2);
        CA[1,:,:,:,:]=cA;
        CA=TensorMap(CA, Vdummy' ⊗ V ⊗ V ← V ⊗ V);
    
        @tensor cB[:]:=sz[-1,-3]*sm[-2,-4];
        CB=zeros(1,2,2,2,2);
        CB[1,:,:,:,:]=cB;
        CB=TensorMap(CB, Vdummy' ⊗ V ⊗ V ← V ⊗ V);
    
    
    
        @tensor Ident[:]:=Ident[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor Ident[:]:=Ident[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    
        @tensor NA[:]:=NA[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NA[:]:=NA[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    
        @tensor NB[:]:=NB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NB[:]:=NB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    
        @tensor NANB[:]:=NANB[1,2,4,5]*U_phy1[-1,3,1,2]*U_phy1'[3,4,5,-2];
        @tensor NANB[:]:=NANB[3,4]*U_phy2[-1,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-2];
    
        @tensor CAdag[:]:=CAdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CAdag[:]:=CAdag[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    
        @tensor CBdag[:]:=CBdag[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CBdag[:]:=CBdag[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    
        @tensor CA[:]:=CA[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CA[:]:=CA[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    
        @tensor CB[:]:=CB[-1,1,2,4,5]*U_phy1[-2,3,1,2]*U_phy1'[3,4,5,-3];
        @tensor CB[:]:=CB[-1,3,4]*U_phy2[-2,3,1,2,5,6]*U_phy2'[4,1,2,5,6,-3];
    end    


    return Ident, NA, NB, NANB, CAdag, CA, CBdag, CB 
end

function evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    H_term=create_H_term(O1,O2,direction,is_odd);
    
    AA1p,_,AA2p=build_double_layer_swap_op(A_fused,A_fused,A_fused,H_term);

    if direction=="x"
        norm=ob_2sites_x(CTM,AA_fused,AA_fused,false);
        ob=ob_2sites_x(CTM,AA1p,AA2p,is_odd);
        norm=blocks(norm)[U1Irrep(0)][1];
        ob=blocks(ob)[U1Irrep(0)][1];


    end
    
    return ob/norm
    
end

function ob_2sites_x(CTM,AA1,AA2,is_odd)

    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    @tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
    @tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];


    if is_odd
        @tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-2]*AA1[-1,2,5,-3,3]*Tset[3][-4,5,4];
        @tensor envR[:]:=Tset[1][-1,3,1]*AA2[-2,5,2,3,-4]*Tset[3][4,5,-3]*envR[1,2,4];
        # println(space(envL,1))
        # println(space(envR,4))
        # println(space(envL,2))
        # println(space(envR,1))
        # println(space(envL,3))
        # println(space(envR,2))
        # println(space(envL,4))
        # println(space(envR,3))
        @tensor rho[:]:=envL[4,1,2,3]*envR[1,2,3,4];
    else
        @tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA1[2,5,-2,3]*Tset[3][-3,5,4];
        @tensor envR[:]:=Tset[1][-1,3,1]*AA2[-2,5,2,3]*Tset[3][4,5,-3]*envR[1,2,4];
        @tensor rho[:]:=envL[1,2,3]*envR[1,2,3];
    end
    return rho;
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




function build_double_layer_swap_extra_leg(Ap,A)
    #The last index of A tensor is an extra virtual index, such as C^dag C


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


function evaluate_correl_Cdag_C(direction, AA_fused, AA_op1, AA_op2, CTM, distance,is_odd)
    correl_funs=Vector(undef,distance);

    C1=CTM["Cset"][1];
    C2=CTM["Cset"][2];
    C3=CTM["Cset"][3];
    C4=CTM["Cset"][4];
    T1=CTM["Tset"][1];
    T2=CTM["Tset"][2];
    T3=CTM["Tset"][3];
    T4=CTM["Tset"][4];





    if direction=="x"
        if is_odd
            @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4]*T3[-3,6,7];
            @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
            @tensor ov[:]:=va[1,2,3]*vb[1,2,3]
            correl_funs[1]=blocks(ov)[U1Irrep(0)][1];
            
            for dis=2:distance
                @tensor va[:]:=va[1,3,5]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                @tensor ov[:]:=va[1,2,3]*vb[1,2,3]
                correl_funs[dis]=blocks(ov)[U1Irrep(0)][1];
            end
        else
            if direction=="x"
                @tensor va[:]:=C1[1,3]*T4[2,5,1]*C4[7,2]*T1[3,4,-1]*AA_op1[5,6,-2,4]*T3[-3,6,7];
                @tensor vb[:]:=T1[-1,4,3]*AA_op2[-2,6,5,4]*T3[7,6,-3]*C2[3,1]*T2[1,5,2]*C3[2,7];
                @tensor ov[:]:=va[1,2,3]*vb[1,2,3]
                correl_funs[1]=blocks(ov)[U1Irrep(0)][1];
                
                for dis=2:distance
                    @tensor va[:]:=va[1,3,5]*T1[1,2,-1]*AA_fused[3,4,-2,2]*T3[-3,4,5];
                    @tensor ov[:]:=va[1,2,3]*vb[1,2,3]
                    correl_funs[dis]=blocks(ov)[U1Irrep(0)][1];
                end
                return correl_funs
            end
        end
        return correl_funs
    end

end


function correl_TransOp(vl,Tup,Tdown,AAfused)
    if AAfused==[]
        
        @tensor vl[:]:=vl[-1,1,3]*Tup[1,2,-2]*Tdown[-3,2,3];
        
    else
        
        @tensor vl[:]:=vl[-1,1,3,5]*Tup[1,2,-2]*AAfused[3,4,-3,2]*Tdown[-4,4,5];
        
    end
    return vl
end
function solve_correl_length(n_values,AA_fused,CTM,direction)
    T1=CTM["Tset"][1];
    T2=CTM["Tset"][2];
    T3=CTM["Tset"][3];
    T4=CTM["Tset"][4];
    println(fuse(space(T1,1)'⊗space(AA_fused,1)', space(T3,3)))
    if direction=="x"
        correl_TransOp_fx(x)=correl_TransOp(x,T1,T3,AA_fused)

        Vl=Rep[U₁](0=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        eu,ev=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
        eus=eu;
        QN=eu*0;

        Vl=Rep[U₁](1=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.+1);
        end



        Vl=Rep[U₁](2=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.+2);
        end

        Vl=Rep[U₁](-1=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.-1);
        end



        Vl=Rep[U₁](-2=>1);
        vl_init = permute(TensorMap(randn, Vl⊗space(T1,1)'⊗space(AA_fused,1)', space(T3,3)), (1,2,3,4,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
        if norm(vl_init)>0
            eu,_=eigsolve(correl_TransOp_fx, vl_init, n_values,:LM,Arnoldi());
            eus=vcat(eus,eu);
            QN=vcat(QN,0*eu.-2);
        end




        eus_abs=abs.(eus);
        @assert maximum(eus_abs)==eus_abs[1]

        eus_abs_sorted=sort(eus_abs,rev=true);
        eus_abs_sorted=eus_abs_sorted/eus_abs_sorted[1];
        QN=QN[sortperm(eus_abs,rev=true)];

        
        return eus_abs_sorted, QN
    end
  
end


function cal_correl(M,A_fused, AA_fused,U_phy1,U_phy2, chi,CTM, distance)
    #M: number of virtual modes 
    
    Ident, NA, NB, NANB, CAdag, CA, CBdag, CB=Hamiltonians(M,U_phy1,U_phy2)

    O1=NA;
    O2=Ident;
    direction="x";
    is_odd=false;
    NA=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    O1=NB;
    O2=Ident;
    direction="x";
    is_odd=false;
    NB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    O1=NANB;
    O2=Ident;
    direction="x";
    is_odd=false;
    NANB=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    
    @tensor O1[:]:=CAdag[1,-1,2]*CB[1,2,-2];
    O2=Ident;
    direction="x";
    is_odd=false;
    CAdagCB_onsite=evaluate_ob(O1, O2, A_fused, AA_fused, CTM, direction, is_odd)
    
    
    
    
    println("NA=   "*string(NA))
    println("NB=   "*string(NB))
    println("NANB=   "*string(NANB))
    println("CAdagCB_onsite=   "*string(CAdagCB_onsite))
    
    
    
    
    
    O1=CAdag;
    O2=CA;
    direction="x";
    is_odd=true;
    H_term=create_H_term(O1,O2,direction,is_odd);
    AA_CAdag,AA_mid,AA_CA=build_double_layer_swap_op(A_fused,A_fused,A_fused,H_term);
    
    # gate=parity_gate(AA_CAdag,4);
    # @tensor AA_CAdag[:]:=AA_CAdag[-1,-2,-3,1,-5]*gate[-4,1];
    # gate=parity_gate(AA_CA,1);
    # @tensor AA_CA[:]:=AA_CA[1,-2,-3,-4,-5]*gate[-1,1];
    
    
    O1=CBdag;
    O2=CB;
    direction="x";
    is_odd=true;
    H_term=create_H_term(O1,O2,direction,is_odd);
    AA_CBdag,AA_mid,AA_CB=build_double_layer_swap_op(A_fused,A_fused,A_fused,H_term);
    



    
    norms=evaluate_correl_Cdag_C("x", AA_fused, AA_fused, AA_fused, CTM, 10, false);
    norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
    norms=evaluate_correl_Cdag_C("x", AA_fused/norm_coe, AA_fused, AA_fused, CTM, distance, false);
    

    CAdag_CA_ob=evaluate_correl_Cdag_C("x", AA_mid/norm_coe, AA_CAdag, AA_CA, CTM, distance, true);
    CAdag_CB_ob=evaluate_correl_Cdag_C("x", AA_mid/norm_coe, AA_CAdag, AA_CB, CTM, distance, true);
    CBdag_CA_ob=evaluate_correl_Cdag_C("x", AA_mid/norm_coe, AA_CBdag, AA_CA, CTM, distance, true);
    CBdag_CB_ob=evaluate_correl_Cdag_C("x", AA_mid/norm_coe, AA_CBdag, AA_CB, CTM, distance, true);

    CAdag_CA_ob=CAdag_CA_ob./norms;
    CAdag_CB_ob=CAdag_CB_ob./norms;
    CBdag_CA_ob=CBdag_CA_ob./norms;
    CBdag_CB_ob=CBdag_CB_ob./norms;

    println(norms)

    eus_x,  QN_x=solve_correl_length(5,AA_fused/norm_coe,CTM,"x");


    _,corner_spec=svd(convert(Array,CTM["Cset"][1]))



    CAdag_CA_ob=[NA;CAdag_CA_ob];
    CBdag_CB_ob=[NB;CBdag_CB_ob];
    CAdag_CB_ob=[CAdagCB_onsite;CAdag_CB_ob];
    CBdag_CA_ob=[CAdagCB_onsite';CBdag_CA_ob];

    mat_filenm="correl_M"*string(M)*"_chi"*string(chi)*".mat";
    matwrite(mat_filenm, Dict(
        "corner_spec" => corner_spec,
        "CAdag_CA_ob" => CAdag_CA_ob,
        "CAdag_CB_ob" => CAdag_CB_ob,
        "CBdag_CA_ob" => CBdag_CA_ob,
        "CBdag_CB_ob" => CBdag_CB_ob,
        "eus_x" => eus_x,
        "QN_x"=> QN_x,
        "CTM_space"=> string(space(CTM["Cset"][1]))
    ); compress = false)
    return CAdag_CA_ob,CAdag_CB_ob,CBdag_CA_ob,CBdag_CB_ob
end


function cal_correl_FP_edge(M, AA_fused,AA_SS,AA_SAL,AA_SBL,AA_SAR,AA_SBR, chi,CTM, distance)
    #M: number of virtual modes 
    


    #single-unitcell correlations
    norm=ob_1site_closed(CTM,AA_fused);
    
    SS_cell_ob=ob_1site_closed(CTM,AA_SS);
    SS_cell_ob=SS_cell_ob/norm;

    
    norms=evaluate_correl_spinspin_FP_edge("x", AA_fused, AA_fused, AA_fused, CTM, "dimerdimer", 10);
    norm_coe=norms[5]/norms[4] #get a rough normalization coefficient to avoid that the number becomes two small
    norms=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_fused, AA_fused, CTM, "dimerdimer", distance);
    dimer_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SS, AA_SS, CTM, "dimerdimer", distance);

    SASA_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SAL, AA_SAR, CTM, "spinspin", distance);
    SASB_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SAL, AA_SBR, CTM, "spinspin", distance);
    SBSA_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SBL, AA_SAR, CTM, "spinspin", distance);
    SBSB_ob=evaluate_correl_spinspin_FP_edge("x", AA_fused/norm_coe, AA_SBL, AA_SBR, CTM, "spinspin", distance);

    dimer_ob=dimer_ob./norms;
    SASA_ob=SASA_ob./norms;
    SASB_ob=SASB_ob./norms;
    SBSA_ob=SBSA_ob./norms;
    SBSB_ob=SBSB_ob./norms;

    println(norms)

    eus_x, Qspin_x, QN_x=solve_correl_length(5,AA_fused/norm_coe,CTM,"x");


    _,corner_spec=svd(convert(Array,CTM["Cset"][1]))

    mat_filenm="correl_FP_edge_M"*string(M)*"_chi"*string(chi)*".mat";
    matwrite(mat_filenm, Dict(
        "corner_spec" => corner_spec,
        "SS_cell_ob" => SS_cell_ob,
        "dimer_ob" => dimer_ob,
        "SASA_ob" => SASA_ob,
        "SASB_ob" => SASB_ob,
        "SBSA_ob" => SBSA_ob,
        "SBSB_ob" => SBSB_ob,
        "eus_x" => eus_x,
        "Qspin_x"=> Qspin_x,
        "QN_x"=> QN_x,
        "CTM_space"=> string(space(CTM["Cset"][1]))
    ); compress = false)
end

function ob_1site_closed(CTM,AA_fused)
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    @tensor envL[:]:=Cset[1][1,-1]*Tset[4][2,-2,1]*Cset[4][-3,2];
    @tensor envR[:]:=Cset[2][-1,1]*Tset[2][1,-2,2]*Cset[3][2,-3];
    @tensor envL[:]:=envL[1,2,4]*Tset[1][1,3,-1]*AA_fused[2,5,-2,3]*Tset[3][-3,5,4];
    @tensor Norm[:]:=envL[1,2,3]*envR[1,2,3];
    Norm=blocks(Norm)[(Irrep[U₁](0) ⊠ Irrep[SU₂](0))][1];
    return Norm;
end