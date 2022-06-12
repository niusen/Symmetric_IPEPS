function gauge_operator(spin_space)
    Z=id(spin_space);
    secs=blocksectors(Z);
    Z_dict=convert(Dict,Z);
    for cs=1:length(secs)
        Z_dict[:data][string(secs[cs])]=Z_dict[:data][string(secs[cs])]*(-1)^(dim(secs[cs])+1);
    end
    Z=convert(TensorMap,Z_dict);
end

function build_double_layer_string(A,operator,stringp_posit,string_posit)

    A=permute(A,(1,2,),(3,4,5));
    U_L=unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    U_D=unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    U_R=inv(U_L);
    U_U=inv(U_D);
    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    Ap=deepcopy(A);#prepare for Adagger
    if stringp_posit==nothing
    elseif stringp_posit=="L"
        Z=gauge_operator(space(Ap,1));
        @tensor Ap[:]=Z[-1,1]*Ap[1,-2,-3,-4,-5];
        Ap=permute(Ap,(1,2,),(3,4,5));
    elseif stringp_posit=="D"
        Z=gauge_operator(space(Ap,2));
        @tensor Ap[:]=Z[-2,1]*Ap[-1,1,-3,-4,-5];
        Ap=permute(Ap,(1,2,),(3,4,5));
    elseif stringp_posit=="R"
        Z=gauge_operator(space(Ap,3));
        @tensor Ap[:]=Z[-3,1]*Ap[-1,-2,1,-4,-5];
        Ap=permute(Ap,(1,2,),(3,4,5));
    elseif stringp_posit=="U"
        Z=gauge_operator(space(Ap,4));
        @tensor Ap[:]=Z[-4,1]*Ap[-1,-2,-3,1,-5];
        Ap=permute(Ap,(1,2,),(3,4,5));
    end

    if string_posit==nothing
    elseif string_posit=="L"
        Z=gauge_operator(space(A,1));
        @tensor A[:]=Z[-1,1]*A[1,-2,-3,-4,-5];
        A=permute(A,(1,2,),(3,4,5));
    elseif string_posit=="D"
        Z=gauge_operator(space(A,2));
        @tensor A[:]=Z[-2,1]*A[-1,1,-3,-4,-5];
        A=permute(A,(1,2,),(3,4,5));
    elseif string_posit=="R"
        Z=gauge_operator(space(A,3));
        @tensor A[:]=Z[-3,1]*A[-1,-2,1,-4,-5];
        A=permute(A,(1,2,),(3,4,5));
    elseif string_posit=="U"
        Z=gauge_operator(space(A,4));
        @tensor A[:]=Z[-4,1]*A[-1,-2,-3,1,-5];
        A=permute(A,(1,2,),(3,4,5));
    end

    uM,sM,vM=tsvd(A);
    uM=uM*sM
    uMp,sMp,vMp=tsvd(Ap);
    uMp=uMp*sMp

    uM=permute(uM,(1,2,3,),())
    uMp=permute(uMp,(1,2,3,),())
    V=space(vM,1);
    U=unitary(fuse(V' ⊗ V), V' ⊗ V);

    @tensor double_LD[:]:=uMp'[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vM=permute(vM,(1,2,3,4,),());
    vMp=permute(vMp,(1,2,3,4,),());
    if operator==[]
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
        @tensor double_RU[:]:=vMp'[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];
    else
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
        @tensor double_RU[:]:=vMp'[3,-2,-4,1]*operator[2,1]*double_RU[-1,3,-3,-5,2];
    end
    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))
    double_RU=permute(double_RU,(1,4,5,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,3,));
    AA_fused=double_LD*double_RU;

    return AA_fused
end


function evaluate_energy(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, method,stringx,stringy)

    H_triangle, H_bond, H12_tensorkit, H31_tensorkit, H23_tensorkit=Hamiltonians(U_phy,parameters["J1"],parameters["J2"],parameters["J3"],parameters["Jchi"],parameters["Jtrip"])
    
    AA1, U_ss=build_double_layer_open(A_unfused,"1",U_phy,U_L,U_D,U_R,U_U);
    AA2, U_ss=build_double_layer_open(A_unfused,"2",U_phy,U_L,U_D,U_R,U_U);
    AA3, U_ss=build_double_layer_open(A_unfused,"3",U_phy,U_L,U_D,U_R,U_U);



    AA_H, _,_,_,_=build_double_layer(A_fused,H_triangle);
    E_up=ob_RD(CTM,AA_H,AA_fused,stringx,stringy)/ob_RD(CTM,AA_fused,AA_fused,stringx,stringy);
    E_up=blocks(E_up)[Irrep[SU₂](0)][1];


    rho_LU_RU_LD=ob_LU_RU_LD(CTM,AA_fused,AA2,AA1,AA3,stringx,stringy);
    rho_LU_RU_LD=permute(rho_LU_RU_LD,(1,3,2,),());#anti-clock-wise order
    @tensor rho_LU_RU_LD[:]:=U_ss[-1,-4,1]*U_ss[-2,-5,2]*U_ss[-3,-6,3]*rho_LU_RU_LD[1,2,3];
    @tensor rho_LU_RU_LD[:]:=U_phy'[4,5,6,-1]*rho_LU_RU_LD[4,5,6,1,2,3]*U_phy[-2,1,2,3];
    #@tensor e[:]:=rho_LU_RU_LD[1,2]*H_triangle[2,1];
    rho_LU_RU_LD=convert(Array,rho_LU_RU_LD);
    @tensor norm_LU_RU_LD[:]:=rho_LU_RU_LD[1,1];
    H_triangle=convert(Array,H_triangle);
    @tensor E_down[:]:=rho_LU_RU_LD[1,2]*H_triangle[2,1];
    E_down=E_down[1]/norm_LU_RU_LD[1];
    return E_up, E_down


end










# function ob_RD_string(CTM,AA_RD,AA_fused,stringx,stringy)
#     Cset=deepcopy(CTM["Cset"]);
#     Tset=deepcopy(CTM["Tset"]);
#     gauge_operator

#     @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-3]*Tset[4][-1,4,1]*AA_fused[4,-2,-4,3]; 
#     @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_fused[-2,-4,4,3]* Tset[2][2,4,-3];

#     @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_fused[3,4,-5,-3]*Cset[4][2,1]*Tset[3][-4,4,2]; 
#     @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_RD[-2,1,2,-4]; 

#     MM_LU=permute(MM_LU,(1,2,),(3,4,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,));
#     MM_LD=permute(MM_LD,(1,2,),(3,4,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:=up[1,2,3,4,]*down[1,2,3,4];
# end





# function ob_LU_RU_LD_string(CTM,AA_fused,AA_LU,AA_RU,AA_LD,stringx,stringy)
#     Cset=deepcopy(CTM["Cset"]);
#     Tset=deepcopy(CTM["Tset"]);

#     @tensor MM_LU[:]:=Cset[1][1,2]*Tset[1][2,3,-4]*Tset[4][-2,4,1]*AA_LU[4,-3,-5,3,-1]; 
#     @tensor MM_RU[:]:=Tset[1][-1,3,1]* Cset[2][1,2]* AA_RU[-2,-4,4,3,-5]* Tset[2][2,4,-3];

#     @tensor MM_LD[:]:=Tset[4][1,3,-2]*AA_LD[3,4,-5,-3,-1]*Cset[4][2,1]*Tset[3][-4,4,2]; 
#     @tensor MM_RD[:]:=Tset[2][-4,-3,2]*Tset[3][1,-2,-1]*Cset[3][2,1]; 
#     @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*AA_fused[-2,1,2,-4]; 

#     MM_LU=permute(MM_LU,(1,2,3,),(4,5,));
#     MM_RU=permute(MM_RU,(1,2,),(3,4,5,));
#     MM_LD=permute(MM_LD,(1,2,3,),(4,5,));
#     MM_RD=permute(MM_RD,(1,2,),(3,4,));

#     up=MM_LU*MM_RU;
#     down=MM_LD*MM_RD;
#     @tensor rho[:]:=up[-1,1,2,3,4,-2]*down[-3,1,2,3,4];
# end

