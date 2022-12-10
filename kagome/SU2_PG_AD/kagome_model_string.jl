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


function fuse_CTM_legs_string(Ttensor,posit,U_L,U_D,U_R,U_U)
    #fuse CTM legs
    if posit=="T1"
        T1=permute(Ttensor,(2,3,),(1,4,));
        T1=U_D*T1;
        T1=permute(T1,(2,1,3,),());
        #display(space(T1))
        return T1
    elseif posit=="T3"
        T3=permute(Ttensor,(1,4,),(2,3,));
        T3=T3*U_U;
        T3=permute(T3,(1,3,2,),());
        return T3
    end
end






function CTMRG_string(A,chi,conv_check,tol,init,CTM_ite_nums, CTM_trun_tol,CTM_ite_info=true)

    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)
    #initial corner transfer matrix

    CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A,init["init_type"],CTM_ite_info);
    T1__=init_CTM_string_T1(A,false,false,U_L,U_D,U_R,U_U);
    T1Z_=init_CTM_string_T1(A,true,false,U_L,U_D,U_R,U_U);
    T1_Z=init_CTM_string_T1(A,false,true,U_L,U_D,U_R,U_U);
    T1ZZ=init_CTM_string_T1(A,true,true,U_L,U_D,U_R,U_U);
    T3__=init_CTM_string_T3(A,false,false,U_L,U_D,U_R,U_U);
    T3Z_=init_CTM_string_T3(A,true,false,U_L,U_D,U_R,U_U);
    T3_Z=init_CTM_string_T3(A,false,true,U_L,U_D,U_R,U_U);
    T3ZZ=init_CTM_string_T3(A,true,true,U_L,U_D,U_R,U_U);

    @assert norm(T3__-CTM["Tset"][3])/norm(CTM["Tset"][3])<1e-14;
    @assert norm(T1__-CTM["Tset"][1])/norm(CTM["Tset"][1])<1e-14;

    AAR__=build_double_layer_string(A_fused,[],nothing,nothing);
    AARZ_=build_double_layer_string(A_fused,[],"R",nothing);
    AAR_Z=build_double_layer_string(A_fused,[],nothing,"R");
    AARZZ=build_double_layer_string(A_fused,[],"R","R");

    @assert norm(AAR__-AA_fused)/norm(AA_fused)<1e-14;

    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    conv_check="singular_value"

    ss_old1=ones(chi)*2;
    ss_old2=ones(chi)*2;
    ss_old3=ones(chi)*2;
    ss_old4=ones(chi)*2;
    d=2;
    rho_old=Matrix(I,d^3,d^3);

    #Iteration

    print_corner=false;
    if print_corner
        println("corner 4:")
        C4_spec=svdvals(convert(Array,Cset[4]));
        C4_spec=C4_spec/C4_spec[1];
        println(C4_spec);
        println("corner 1:")
        C1_spec=svdvals(convert(Array,Cset[1]));
        C1_spec=C1_spec/C1_spec[1];
        println(C1_spec);
        println("corner 3:")
        C3_spec=svdvals(convert(Array,Cset[3]));
        C3_spec=C3_spec/C3_spec[1];
        println(C3_spec);
        println("corner 2:")
        C2_spec=svdvals(convert(Array,Cset[2]));
        C2_spec=C2_spec/C2_spec[1];
        println(C2_spec);
        println("CTM init finished")
    end
    


    if CTM_ite_info
        println("start CTM iterations:")
    end
    for ci=1:CTM_ite_nums
        #direction_order=[1,2,3,4];
        #direction_order=[4,1,2,3];
        direction_order=[3,4,1,2];
        for direction in direction_order
            Cset,Tset,T1__,T1Z_,T1_Z,T1ZZ,T3__,T3Z_,T3_Z,T3ZZ=CTM_ite_string(Cset, Tset, AA_fused, AAR__,AARZ_,AAR_Z,AARZZ, chi, direction,CTM_trun_tol,CTM_ite_info,T1__,T1Z_,T1_Z,T1ZZ,T3__,T3Z_,T3_Z,T3ZZ);
        end

        print_corner=false;
        if print_corner
            println("corner 4:")
            C4_spec=svdvals(convert(Array,Cset[4]));
            C4_spec=C4_spec/C4_spec[1];
            println(C4_spec);
            println("corner 1:")
            C1_spec=svdvals(convert(Array,Cset[1]));
            C1_spec=C1_spec/C1_spec[1];
            println(C1_spec);
            println("corner 3:")
            C3_spec=svdvals(convert(Array,Cset[3]));
            C3_spec=C3_spec/C3_spec[1];
            println(C3_spec);
            println("corner 2:")
            C2_spec=svdvals(convert(Array,Cset[2]));
            C2_spec=C2_spec/C2_spec[1];
            println(C2_spec);
            println("next iteration:")
        end
        


        if conv_check=="singular_value" #check convergence of singular value
            er1,ss_new1=spectrum_conv_check(ss_old1,Cset[1]);
            er2,ss_new2=spectrum_conv_check(ss_old2,Cset[2]);
            er3,ss_new3=spectrum_conv_check(ss_old3,Cset[3]);
            er4,ss_new4=spectrum_conv_check(ss_old4,Cset[4]);

            er=maximum([er1,er2,er3,er4]);
            if CTM_ite_info
                println("CTMRG iteration: "*string(ci)*", CTMRG err: "*string(er));
            end
            if er<tol
                break;
            end
            ss_old1=ss_new1;
            ss_old2=ss_new2;
            ss_old3=ss_new3;
            ss_old4=ss_new4;
        elseif conv_check=="density_matrix" #check reduced density matrix

            # ob_opts.SiteNumber=1;
            # CTM_tem.Cset=Cset;
            # CTM_tem.Tset=Tset;
            # rho_new=ob_CTMRG(CTM_tem,A,ob_opts).A;
            # er=sum(sum((abs(rho_old-rho_new))));
            # disp(['CTMRG iteration: ',num2str(ci),' CTMRG err: ',num2str(er)]);
            # if er<tol
            #     break;
            # end
            # rho_old=rho_new;
        end

        # if ci==CTM_ite_nums
        #     display(er)
        #     warn("CTMRG does not converge: " * string(er));
        # end
    end

    CTM["Cset"]=Cset;
    CTM["Tset"]=Tset;
    return CTM, AA_fused, U_L,U_D,U_R,U_U,AAR__,AARZ_,AAR_Z,AARZZ,T1__,T1Z_,T1_Z,T1ZZ,T3__,T3Z_,T3_Z,T3ZZ

end

function CTM_ite_string(Cset, Tset, AA_fused, AAR__,AARZ_,AAR_Z,AARZZ, chi, direction, trun_tol,CTM_ite_info,T1__,T1Z_,T1_Z,T1ZZ,T3__,T3Z_,T3_Z,T3ZZ)

    AA=permute(AA_fused, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());

    AAR__=permute(AAR__, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
    AARZ_=permute(AARZ_, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
    AAR_Z=permute(AAR_Z, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
    AARZZ=permute(AARZZ, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
    

    @tensor MMup[:]:=Cset[mod1(direction,4)][1,2]*Tset[mod1(direction,4)][2,3,-3]*Tset[mod1(direction-1,4)][-1,4,1]*AA[4,-2,-4,3];
    @tensor MMlow[:]:=Tset[mod1(direction-1,4)][1,3,-1]*AA[3,4,-4,-2]*Cset[mod1(direction-1,4)][2,1]*Tset[mod1(direction-2,4)][-3,4,2];

    @tensor MMup_reflect[:]:=Tset[mod1(direction,4)][-1,3,1]* Cset[mod1(direction+1,4)][1,2]* AA[-2,-4,4,3]* Tset[mod1(direction+1,4)][2,4,-3];
    #@tensor MMlow_reflect[:]:=AA[-2,4,3,-4]*Tset[mod1(direction+1,4)][-3,3,1]*Tset[mod1(direction-2,4)][2,4,-1]*Cset[mod1(direction-2,4)][1,2]; #this is slow compared to other coners, I don't know why
    @tensor MMlow_reflect[:]:=Tset[mod1(direction+1,4)][-4,-3,2]*Tset[mod1(direction-2,4)][1,-2,-1]*Cset[mod1(direction-2,4)][2,1];
    @tensor MMlow_reflect[:]:=MMlow_reflect[-1,1,2,-3]*AA[-2,1,2,-4];

    MMup=permute(MMup,(1,2,),(3,4,))

    # _,ss,_=tsvd(MMup)
    # display(convert(Array,ss))

    MMlow=permute(MMlow,(1,2,),(3,4,))
    MMup_reflect=permute(MMup_reflect,(1,2,),(3,4,))
    MMlow_reflect=permute(MMlow_reflect,(1,2,),(3,4,))

    

    RMup=permute(MMup*MMup_reflect,(3,4,),(1,2,));
    RMlow=MMlow*MMlow_reflect;

    M=RMup*RMlow;

    uM,sM,vM = tsvd(M; trunc=truncdim(chi+20));
    #println(diag(convert(Array,sM)))

    sM=truncate_multiplet(sM,chi,1e-5,trun_tol);
    
    uM_new,sM_new,vM_new=delet_zero_block(uM,sM,vM);
    @assert (norm(uM_new*sM_new*vM_new-uM*sM*vM)/norm(uM*sM*vM))<1e-14
    uM=uM_new;
    sM=sM_new;
    vM=vM_new;
    #println(diag(convert(Array,sM)))


    sM=sM/norm(sM)
    sM_inv=pinv(sM);
    sM_dense=convert(Array,sM)

    # println("svd:")
    # sm_=sort(diag(sM_dense),rev=true)
    # println(sm_/sm_[1])

    # _,sM_test,_ = tsvd(M; trunc=truncdim(chi+1));
    # sm_=sort(diag(convert(Array,sM_test)),rev=true)
    # println(sm_/sm_[1])

    for c1=1:size(sM_dense,1)
        if sM_dense[c1,c1]<trun_tol
            sM_dense[c1,c1]=0;
        end
    end
    #display(sM_dense)
    #display(pinv.(sM_dense))

    #display(sM_inv)
    #display(convert(Array,sM_inv))
    #sM_inv_sqrt=sqrt.(convert(Array,sM_inv))
    #display(space(sM_inv))
    #display(sM_inv_sqrt)
    sM_inv_sqrt=TensorMap(pinv.(sqrt.(sM_dense)),codomain(sM_inv)←domain(sM_inv))

    PM_inv=RMlow*vM'*sM_inv_sqrt;
    PM=sM_inv_sqrt*uM'*RMup;
    PM=permute(PM,(2,3,),(1,));

    @tensor M5tem[:]:=Tset[mod1(direction-1,4)][4,3,1]*AA[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
    @tensor M1tem[:]:=Cset[mod1(direction,4)][1,2]*Tset[mod1(direction,4)][2,3,-2]*PM_inv[1,3,-1];
    @tensor M7tem[:]:=Cset[mod1(direction-1,4)][1,2]*Tset[mod1(direction-2,4)][-1,3,1]* PM[2,3,-2];


    if direction==2 #update T1

        @tensor T1__[:]:=T1__[4,3,1]*AAR__[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
        T1__=T1__/norm(M5tem);
        @tensor T1Z_[:]:=T1Z_[4,3,1]*AARZ_[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
        T1Z_=T1Z_/norm(M5tem);
        @tensor T1_Z[:]:=T1_Z[4,3,1]*AAR_Z[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
        T1_Z=T1_Z/norm(M5tem);
        @tensor T1ZZ[:]:=T1ZZ[4,3,1]*AARZZ[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
        T1ZZ=T1ZZ/norm(M5tem);
    end
    if direction==4 #update T3

        @tensor T3__[:]:=T3__[4,3,1]*AAR__[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
        T3__=T3__/norm(M5tem);
        @tensor T3Z_[:]:=T3Z_[4,3,1]*AARZ_[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
        T3Z_=T3Z_/norm(M5tem);
        @tensor T3_Z[:]:=T3_Z[4,3,1]*AAR_Z[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
        T3_Z=T3_Z/norm(M5tem);
        @tensor T3ZZ[:]:=T3ZZ[4,3,1]*AARZZ[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
        T3ZZ=T3ZZ/norm(M5tem);
    end



    Cset[mod1(direction,4)]=M1tem/norm(M1tem);
    Tset[mod1(direction-1,4)]=M5tem/norm(M5tem);
    Cset[mod1(direction-1,4)]=M7tem/norm(M7tem);
    return Cset,Tset, T1__,T1Z_,T1_Z,T1ZZ,T3__,T3Z_,T3_Z,T3ZZ
end

function init_CTM_string_T1(A,Zp,Z,U_L,U_D,U_R,U_U)
 
    direction=1;
    inds=(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),5);
    A_rotate=permute(A,inds);
    Ap_rotate=A_rotate';

    
    if Zp
        Zg=gauge_operator(space(Ap_rotate,3));
        @tensor Ap_rotate[:]=Zg[-3,1]*Ap_rotate[-1,-2,1,-4,-5];
    end
    if Z
        Zg=gauge_operator(space(A_rotate,3));
        @tensor A_rotate[:]=Zg[-3,1]*A_rotate[-1,-2,1,-4,-5];
    end

    @tensor T1[:]:=Ap_rotate[-1,-3,-5,1,2]*A_rotate[-2,-4,-6,1,2];
    @tensor C1[:]:=Ap_rotate[1,-1,-3,2,3]*A_rotate[1,-2,-4,2,3];


    #fuse legs
    direction=1;
    ul=unitary(fuse(space(C1, 3) ⊗ space(C1, 4)), space(C1, 3) ⊗ space(C1, 4));
    ur=unitary(fuse(space(T1, 5) ⊗ space(T1, 6)), space(T1, 5) ⊗ space(T1, 6));


    direction=1;
    ulp=permute(ul',(3,),(1,2,));
    urp=permute(ur',(3,),(1,2,));

    @tensor T1[:]:=ulp[-1,1,2]*T1[1,2,-2,-3,3,4]*ur[-4,3,4];#put all indices in tone side so that its adjoint has the same index order

    T1=fuse_CTM_legs_string(T1,"T1",U_L,U_D,U_R,U_U);
    return T1
end


function init_CTM_string_T3(A,Zp,Z,U_L,U_D,U_R,U_U)
 
    direction=3;
    inds=(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),5);
    A_rotate=permute(A,inds);
    Ap_rotate=A_rotate';

    
    if Zp
        Zg=gauge_operator(space(Ap_rotate,1));
        @tensor Ap_rotate[:]=Zg[-1,1]*Ap_rotate[1,-2,-3,-4,-5];
    end
    if Z
        Zg=gauge_operator(space(A_rotate,1));
        @tensor A_rotate[:]=Zg[-1,1]*A_rotate[1,-2,-3,-4,-5];
    end

    @tensor T3[:]:=Ap_rotate[-1,-3,-5,1,2]*A_rotate[-2,-4,-6,1,2];
    @tensor C3[:]:=Ap_rotate[1,-1,-3,2,3]*A_rotate[1,-2,-4,2,3];


    #fuse legs
    direction=3;
    ul=unitary(fuse(space(C3, 3) ⊗ space(C3, 4))', space(C3, 3) ⊗ space(C3, 4));
    ur=unitary(fuse(space(T3, 5) ⊗ space(T3, 6))', space(T3, 5) ⊗ space(T3, 6));

    direction=3;
    ulp=permute(ul',(3,),(1,2,));
    urp=permute(ur',(3,),(1,2,));

    @tensor T3[:]:=ulp[-1,1,2]*T3[1,2,-2,-3,3,4]*ur[-4,3,4];#put all indices in tone side so that its adjoint has the same index order
    T3=fuse_CTM_legs_string(T3,"T3",U_L,U_D,U_R,U_U);

    return T3
    
end