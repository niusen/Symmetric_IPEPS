using LinearAlgebra
using TensorKit


function build_double_layer(A,operator)
    #display(space(A))
    A=permute(A,(1,2,),(3,4,5));
    U_L=unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    U_D=unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    U_R=inv(U_L);
    U_U=inv(U_D);
    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uM,sM,vM=tsvd(A);
    uM=uM*sM

    uM=permute(uM,(1,2,3,),())
    V=space(vM,1);
    U=unitary(fuse(V' ⊗ V), V' ⊗ V);
    @tensor double_LD[:]:=uM'[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vM=permute(vM,(1,2,3,4,),());
    if operator==[]
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
        @tensor double_RU[:]:=vM'[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];
    else
        @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
        @tensor double_RU[:]:=vM'[3,-2,-4,1]*operator[2,1]*double_RU[-1,3,-3,-5,2];
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

    
    return AA_fused, U_L,U_D,U_R,U_U
end


function fuse_CTM_legs(CTM,U_L,U_D,U_R,U_U)
    #fuse CTM legs
    Tset=CTM["Tset"];

    #T4
    T4=permute(Tset[4],(1,4,),(2,3,));
    T4=T4*U_R;
    T4=permute(T4,(1,3,2,),());
    Tset[4]=T4
    #display(space(T4))

    #T3
    T3=permute(Tset[3],(1,4,),(2,3,));
    T3=T3*U_U;
    T3=permute(T3,(1,3,2,),());
    Tset[3]=T3
    #display(space(T3))

    #T2
    T2=permute(Tset[2],(2,3,),(1,4,));
    T2=U_L*T2;
    T2=permute(T2,(2,1,3,),());
    Tset[2]=T2
    #display(space(T2))

    #T1
    T1=permute(Tset[1],(2,3,),(1,4,));
    T1=U_D*T1;
    T1=permute(T1,(2,1,3,),());
    Tset[1]=T1
    #display(space(T1))

    CTM["Tset"]=Tset;
    return CTM
end

function spectrum_conv_check(ss_old,C_new)
    _,ss_new,_=svd(permute(C_new,(1,),(2,)));
    ss_new=convert(Array,ss_new);
    ss_new=sort(diag(ss_new), rev=true);
    ss_old=ss_old/ss_old[1];
    ss_new=ss_new/ss_new[1];
    #display(ss_new)
    if length(ss_old)>length(ss_new)
        dss=copy(ss_old);
        siz=length(ss_new)
    elseif length(ss_old)<=length(ss_new)
        dss=copy(ss_new);
        siz=length(ss_old)
    end
    dss[1:siz]=ss_old[1:siz]-ss_new[1:siz]
    # println("spectra diff:")
    # println(ss_old);
    # println(ss_new)
    er=norm(dss);
    return er,ss_new
end

function CTMRG(A,chi,conv_check,tol,init,CTM_ite_nums, CTM_trun_tol,CTM_ite_info=true,CTM_conv_info=false,projector_strategy="4x4",CTM_trun_svd=false,svd_lanczos_tol=1e-10)

    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)
    #initial corner transfer matrix
        #initial corner transfer matrix
    if isempty(init["CTM"])
        CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A,init["init_type"],CTM_ite_info);
        AA_memory=Base.summarysize(AA_fused)/1024/1024;
        println("Memory cost of double layer tensor: "*string(AA_memory)*" Mb.")
    else
        _, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A,init["init_type"],CTM_ite_info);
        CTM=deepcopy(init["CTM"]);
    end

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
    

    AA_rotated=rotate_AA(AA_fused);

    if CTM_ite_info
        println("start CTM iterations:")
    end
    ite_num=0;
    ite_err=1;
    for ci=1:CTM_ite_nums
        ite_num=ci;
        #direction_order=[1,2,3,4];
        #direction_order=[4,1,2,3];
        direction_order=[3,4,1,2];
        for direction in direction_order
            if CTM_trun_svd
                Cset,Tset=CTM_ite_trunSVD(Cset, Tset, AA_rotated[direction], chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,svd_lanczos_tol);
            else
                Cset,Tset=CTM_ite(Cset, Tset, AA_rotated[direction], chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy);
            end
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
            ite_err=er;
            if CTM_ite_info
                println("CTMRG iteration: "*string(ci)*", CTMRG err: "*string(er));flush(stdout);
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
    if CTM_conv_info
        return CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err
    else
        return CTM, AA_fused, U_L,U_D,U_R,U_U
    end

end
function rotate_AA(AA_fused)
    AA_rotated=Vector(undef,4);
    for direction=1:4
        AA_rotated[direction]=permute(AA_fused, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
    end
    return AA_rotated
end

function CTM_ite(Cset, Tset, AA, chi, direction, trun_tol,CTM_ite_info,projector_strategy)

    
    M1=Cset[mod1(direction,4)];
    M2=Tset[mod1(direction,4)];
    M3=Tset[mod1(direction-1,4)];
    #M4=AA;
    M5=M3;
    #M6=M4;
    M7=Cset[mod1(direction-1,4)];
    M8=Tset[mod1(direction-2,4)];

    @tensor MMup[:]:=M1[1,2]*M2[2,3,-3]*M3[-1,4,1]*AA[4,-2,-4,3];
    @tensor MMlow[:]:=M5[1,3,-1]*AA[3,4,-4,-2]*M7[2,1]*M8[-3,4,2];

    MMup=permute(MMup,(1,2,),(3,4,))
    MMlow=permute(MMlow,(1,2,),(3,4,))

    if projector_strategy=="4x4"
        M1_=Cset[mod1(direction+1,4)];
        M2_=Tset[mod1(direction,4)];
        M3_=Tset[mod1(direction+1,4)];
        #M4_=M4;
        M5_=M3_;
        #M6_=M4_;
        M7_=Cset[mod1(direction-2,4)];
        M8_=Tset[mod1(direction-2,4)];

        @tensor MMup_reflect[:]:=M2_[-1,3,1]* M1_[1,2]* AA[-2,-4,4,3]* M3_[2,4,-3];
        @tensor MMlow_reflect[:]:=M5_[-4,-3,2]*M8_[1,-2,-1]*M7_[2,1];
        @tensor MMlow_reflect[:]:=MMlow_reflect[-1,1,2,-3]*AA[-2,1,2,-4];

        MMup_reflect=permute(MMup_reflect,(1,2,),(3,4,))
        MMlow_reflect=permute(MMlow_reflect,(1,2,),(3,4,))
    end

    
    if projector_strategy=="4x4"
        RMup=permute(MMup*MMup_reflect,(3,4,),(1,2,));
        RMlow=MMlow*MMlow_reflect;
        M=RMup*RMlow;
    elseif projector_strategy=="4x2"
        RMup=permute(MMup,(3,4,),(1,2,));
        RMlow=MMlow;
        M=RMup*RMlow;
    end

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

    # println(norm(M5tem))
    # println(norm(M1tem))
    # println(norm(M7tem))

    Cset[mod1(direction,4)]=M1tem/norm(M1tem);
    Tset[mod1(direction-1,4)]=M5tem/norm(M5tem);
    Cset[mod1(direction-1,4)]=M7tem/norm(M7tem);
    return Cset,Tset
end

function HR(x,M1,M2,M3,M5,M7,M8,M1p,M2p,M3p,M5p,M7p,M8p,AA)
    x=deepcopy(x);
    @tensor x[:]:=x[2,3,-3]*AA[-2,5,4,3]*M5p[2,4,1]*M7p[1,6]*M8p[6,5,-1];
    @tensor x[:]:=x[2,3,-3]*AA[5,4,3,-2]*M5[6,5,-1]*M7[1,6]*M8[2,4,1];
    @tensor x[:]:=x[2,3,-3]*AA[4,3,-2,5]*M1[1,6]*M2[6,5,-1]*M3[2,4,1];
    @tensor x[:]:=x[2,3,-3]*AA[3,-2,5,4]*M1p[1,6]*M2p[2,4,1]*M3p[6,5,-1];
    #println(norm(x))
    return x
end

function HR_conj(x,M1,M2,M3,M5,M7,M8,M1p,M2p,M3p,M5p,M7p,M8p,AA)
    # x=deepcopy(x)';
    # @tensor x[:]:=x[2,3,-3]*AA[-2,3,4,5]*M1p[6,1]*M2p[-1,5,6]*M3p[1,4,2];
    # @tensor x[:]:=x[2,3,-3]*AA[5,-2,3,4]*M1[6,1]*M2[1,4,2]*M3[-1,5,6];
    # @tensor x[:]:=x[2,3,-3]*AA[4,5,-2,3]*M5[1,4,2]*M7[6,1]*M8[-1,5,6];
    # @tensor x[:]:=x[2,3,-3]*AA[3,4,5,-2]*M5p[-1,5,6]*M7p[6,1]*M8p[1,4,2];
    # return permute(x',(1,2,3,))
    x=deepcopy(x);
    @tensor x[:]:=x[2,3,-3]*AA'[-2,3,4,5]*M1p'[6,1]*M2p'[-1,5,6]*M3p'[1,4,2];
    @tensor x[:]:=x[2,3,-3]*AA'[5,-2,3,4]*M1'[6,1]*M2'[1,4,2]*M3'[-1,5,6];
    @tensor x[:]:=x[2,3,-3]*AA'[4,5,-2,3]*M5'[1,4,2]*M7'[6,1]*M8'[-1,5,6];
    @tensor x[:]:=x[2,3,-3]*AA'[3,4,5,-2]*M5p'[-1,5,6]*M7p'[6,1]*M8p'[1,4,2];
    #println(norm(x))
    return x
end

function CTM_ite_trunSVD(Cset, Tset, AA, chi, direction, trun_tol,CTM_ite_info,projector_strategy,svd_lanczos_tol)
    AA=AA/norm(AA);

    M1=Cset[mod1(direction,4)];M1=M1/norm(M1);
    M2=Tset[mod1(direction,4)];M2=M2/norm(M2);
    M3=Tset[mod1(direction-1,4)];M3=M3/norm(M3);
    #M4=AA;
    M5=M3;
    #M6=M4;
    M7=Cset[mod1(direction-1,4)];M7=M7/norm(M7);
    M8=Tset[mod1(direction-2,4)];M8=M8/norm(M8);

    M1_=Cset[mod1(direction+1,4)];M1_=M1_/norm(M1_);
    M2_=Tset[mod1(direction,4)];M2_=M2_/norm(M2_);
    M3_=Tset[mod1(direction+1,4)];M3_=M3_/norm(M3_);
    #M4_=M4;
    M5_=M3_;
    #M6_=M4_;
    M7_=Cset[mod1(direction-2,4)];M7_=M7_/norm(M7_);
    M8_=Tset[mod1(direction-2,4)];M8_=M8_/norm(M8_);


    ##############
    # JLDnm="test.jld2";
    # init___=Dict([("M1", M1), ("M2",M2),("M3",M3),("M5",M5),("M7",M7),("M8",M8),("M1_", M1_), ("M2_",M2_),("M3_",M3_),("M5_",M5_),("M7_",M7_),("M8_",M8_),("AA",AA)]);
    # save(JLDnm, "init",init___);

    spins=Vector(undef,0);
    S_set=Vector(undef,0);
    U_set=Vector(undef,0);
    V_set=Vector(undef,0);

    Vspace=fuse(space(M3_,3)*space(AA,2));
    AA=AA*dim(Vspace);

    #######
    S_max=normalize_AA(M1,M2,M3,M5,M7,M8,M1_,M2_,M3_,M5_,M7_,M8_,AA);
    AA=AA/(S_max^(1/4));#four AA are multiplied in a 4x4 cluster 
    #######

    for cs=1:length(Vspace.dims.keys)
        spin=Vspace.dims.keys[cs].j;
        #println(spin)
        Random.seed!(1234)
        vr_init=permute(TensorMap(randn, space(M3_,3)*space(AA,2),SU₂Space(spin=>1)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.
    
    
        n_keep=min(Vspace.dims.values[cs],Int(round(chi/(2*spin+1))));
    
        HR_svd_R(x)=HR(x,M1,M2,M3,M5,M7,M8,M1_,M2_,M3_,M5_,M7_,M8_,AA);
        HR_svd_R_conj(x)=HR_conj(x,M1,M2,M3,M5,M7,M8,M1_,M2_,M3_,M5_,M7_,M8_,AA);
    
        vr_temp=deepcopy(vr_init);
        ite_test=4;
        for cccc=1:ite_test
            vr_temp=HR(vr_temp,M1,M2,M3,M5,M7,M8,M1_,M2_,M3_,M5_,M7_,M8_,AA);
        end
        norm_coe=norm(vr_temp)/norm(vr_init);
        norm_coe=norm_coe^(1/ite_test);
        if norm_coe<1e-14
            continue;
        end
    
    #@suppress begin
        S,U,V,info=svdsolve((HR_svd_R,HR_svd_R_conj), vr_init, n_keep,:LR, krylovdim=n_keep*3,tol=svd_lanczos_tol);
        S_set=vcat(S_set,S);U_set=vcat(U_set,U);V_set=vcat(V_set,V);
        spins=vcat(spins,spin*ones(length(S)));
        #@assert info.converged >= minimum([n_eff,dim(full_space,sec)])
        # println(S)
        # if ((S[1]/S_set[1])<1e-16)
        #     println("truncate at spin= "*string(spin))
        #     break;
        # end
    #end
        
        

    end
    

    multiplet_tol=1e-5;
    trun_tol=1e-12;
    pos=truncated_svd_truncate(spins,S_set,chi,multiplet_tol,trun_tol);
    spins=spins[pos];
    S_set=S_set[pos];
    U_set=U_set[pos];
    V_set=V_set[pos];
    
    Um,Sm,Vm=group_svd_components(U_set,S_set,V_set,spins);
    TT_=Um*Sm*Vm';
    Vm=Vm';


    ##############



    @tensor MMup[:]:=M1[1,2]*M2[2,3,-3]*M3[-1,4,1]*AA[4,-2,-4,3];
    @tensor MMlow[:]:=M5[1,3,-1]*AA[3,4,-4,-2]*M7[2,1]*M8[-3,4,2];

    @tensor MMup_reflect[:]:=M2_[-1,3,1]* M1_[1,2]* AA[-2,-4,4,3]* M3_[2,4,-3];
    @tensor MMlow_reflect[:]:=M5_[-4,-3,2]*M8_[1,-2,-1]*M7_[2,1];
    @tensor MMlow_reflect[:]:=MMlow_reflect[-1,1,2,-3]*AA[-2,1,2,-4];

    MMup=permute(MMup,(1,2,),(3,4,))
    MMlow=permute(MMlow,(1,2,),(3,4,))

    MMup_reflect=permute(MMup_reflect,(1,2,),(3,4,))
    MMlow_reflect=permute(MMlow_reflect,(1,2,),(3,4,))

    RMup=permute(MMup*MMup_reflect,(3,4,),(1,2,));
    RMlow=MMlow*MMlow_reflect;

    M=RMup*RMlow;

    uM,sM,vM = tsvd(M; trunc=truncdim(chi+20));
    sM=truncate_multiplet(sM,chi,1e-5,trun_tol);

    #############
    TT__=uM*sM*vM;
    svd_err=norm(permute(TT_,(1,2,),(3,4,))-TT__)/norm(TT__);
    println("error of truncated svd: "*string(norm(permute(TT_,(1,2,),(3,4,))-TT__)/norm(TT__)));
    #@assert svd_err<0.5
    
    #############
    Um,Sm,Vm=delet_zero_block(Um,Sm,Vm);

    uM_new,sM_new,vM_new=delet_zero_block(uM,sM,vM);
    @assert (norm(uM_new*sM_new*vM_new-uM*sM*vM)/norm(uM*sM*vM))<1e-14
    uM0=uM_new;
    sM0=sM_new;
    vM0=vM_new;



    ######################
    uM=Um;
    sM=Sm;
    vM=Vm;


    sM=sM/norm(sM)
    sM_inv=pinv(sM);
    sM_dense=convert(Array,sM)
    for c1=1:size(sM_dense,1)
        if abs(sM_dense[c1,c1])<trun_tol
            sM_dense[c1,c1]=0;
        end
    end


    sM0=sM0/norm(sM0)
    sM_inv0=pinv(sM0);
    sM_dense0=convert(Array,sM0)
    for c1=1:size(sM_dense0,1)
        if sM_dense0[c1,c1]<trun_tol
            sM_dense0[c1,c1]=0;
        end
    end

    sM_inv_sqrt=TensorMap(pinv.(sqrt.(sM_dense)),codomain(sM_inv)←domain(sM_inv))
    sM_inv_sqrt0=TensorMap(pinv.(sqrt.(sM_dense0)),codomain(sM_inv0)←domain(sM_inv0))



    PM_inv0=RMlow*vM0'*sM_inv_sqrt0;
    PM_inv=RMlow*vM'*sM_inv_sqrt;
    # @tensor PM_inv[:]:=vM'[2,3,-3]*AA[-2,5,4,3]*M5_[2,4,1]*M7_[1,6]*M8_[6,5,-1];
    # @tensor PM_inv[:]:=PM_inv[2,3,-3]*AA[5,4,3,-2]*M5[6,5,-1]*M7[1,6]*M8[2,4,1];
    # @tensor PM_inv[:]:=PM_inv[-1,-2,1]*sM_inv_sqrt[1,-3];
    # PM_inv=permute(PM_inv,(1,2,),(3,));

    # println(norm(PM_inv))
    # println(norm(PM_inv0))



    PM0=sM_inv_sqrt0*uM0'*RMup;
    PM=sM_inv_sqrt*uM'*RMup;
    # @tensor PM[:]:=uM'[-1,2,3]*AA[-3,3,4,5]*M1_[6,1]*M2_[-2,5,6]*M3_[1,4,2];
    # @tensor PM[:]:=PM[-1,2,3]*AA[5,-3,3,4]*M1[6,1]*M2[1,4,2]*M3[-2,5,6];
    # @tensor PM[:]:=sM_inv_sqrt[-1,1]*PM[1,-2,-3];
    # PM=permute(PM,(1,),(2,3,));


    # JLDnm="test.jld2";
    # init___=Dict([("PM",PM),("PM_inv",PM_inv),("PM0",PM0),("PM_inv0",PM_inv0), ("M1", M1), ("M2",M2),("M3",M3),("M5",M5),("M7",M7),("M8",M8),("M1_", M1_), ("M2_",M2_),("M3_",M3_),("M5_",M5_),("M7_",M7_),("M8_",M8_),("AA",AA)]);
    # save(JLDnm, "init",init___);
    
    # println(norm(PM))
    # println(norm(PM0))




    PM=permute(PM,(2,3,),(1,));
    PM0=permute(PM0,(2,3,),(1,));

    @tensor M5tem[:]:=M3[4,3,1]*AA[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
    @tensor M1tem[:]:=M1[1,2]*M2[2,3,-2]*PM_inv[1,3,-1];
    @tensor M7tem[:]:=M7[1,2]*M8[-1,3,1]* PM[2,3,-2];

    @tensor M5tem0[:]:=M3[4,3,1]*AA[3,5,-2,2]* PM_inv0[4,5,-1]* PM0[1,2,-3];
    @tensor M1tem0[:]:=M1[1,2]*M2[2,3,-2]*PM_inv0[1,3,-1];
    @tensor M7tem0[:]:=M7[1,2]*M8[-1,3,1]* PM0[2,3,-2];



    Cset[mod1(direction,4)]=M1tem/norm(M1tem);
    Tset[mod1(direction-1,4)]=M5tem/norm(M5tem);
    Cset[mod1(direction-1,4)]=M7tem/norm(M7tem);

    # Cset[mod1(direction,4)]=M1tem0/norm(M1tem0);
    # Tset[mod1(direction-1,4)]=M5tem0/norm(M5tem0);
    # Cset[mod1(direction-1,4)]=M7tem0/norm(M7tem0);
    return Cset,Tset
end


function init_CTM(chi,A,type,CTM_ite_info)
    if CTM_ite_info
        display("initialize CTM")
    end
    #numind(A)
    #numin(A)
    #numout(A)
    CTM=[];
    Cset=Vector(undef,4);
    Tset=Vector(undef,4);
    #space(A,1)
    if type=="PBC"
        for direction=1:4
            inds=(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),5);
            A_rotate=permute(A,inds);
            Ap_rotate=A_rotate';

            @tensor M[:]:=Ap_rotate[1,-1,-3,2,3]*A_rotate[1,-2,-4,2,3];
            Cset[direction]=M;
            @tensor M[:]:=Ap_rotate[-1,-3,-5,1,2]*A_rotate[-2,-4,-6,1,2];
            Tset[direction]=M;
        end

        #fuse legs
        ul_set=Vector(undef,4);
        ur_set=Vector(undef,4);
        for direction=1:2
            ul_set[direction]=unitary(fuse(space(Cset[direction], 3) ⊗ space(Cset[direction], 4)), space(Cset[direction], 3) ⊗ space(Cset[direction], 4));
            ur_set[direction]=unitary(fuse(space(Tset[direction], 5) ⊗ space(Tset[direction], 6)), space(Tset[direction], 5) ⊗ space(Tset[direction], 6));
        end
        for direction=3:4
            ul_set[direction]=unitary(fuse(space(Cset[direction], 3) ⊗ space(Cset[direction], 4))', space(Cset[direction], 3) ⊗ space(Cset[direction], 4));
            ur_set[direction]=unitary(fuse(space(Tset[direction], 5) ⊗ space(Tset[direction], 6))', space(Tset[direction], 5) ⊗ space(Tset[direction], 6));
        end
        for direction=1:4
            C=Cset[direction];
            ul=ur_set[mod1(direction-1,4)];
            ur=ul_set[direction];
            ulp=permute(ul',(3,),(1,2,));
            urp=permute(ur',(3,),(1,2,));
            #@tensor Cnew[(-1);(-2)]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4]
            @tensor Cnew[:]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4];#put all indices in tone side so that its adjoint has the same index order
            Cset[direction]=Cnew;

            T=Tset[direction];
            ul=ul_set[direction];
            ur=ur_set[direction];
            ulp=permute(ul',(3,),(1,2,));
            urp=permute(ur',(3,),(1,2,));
            #@tensor Tnew[(-1);(-2,-3,-4)]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4]
            @tensor Tnew[:]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4];#put all indices in tone side so that its adjoint has the same index order
            Tset[direction]=Tnew;
        end
    elseif type=="random"
    end
    CTM=Dict([("Cset", Cset), ("Tset", Tset)]);

    AA_fused, U_L,U_D,U_R,U_U=build_double_layer(A,[]);
    CTM=fuse_CTM_legs(CTM,U_L,U_D,U_R,U_U);

    return CTM, AA_fused, U_L,U_D,U_R,U_U


    # #save initial CTM to compare with other codes
    # @time CTM=init_CTM(10,PEPS_tensor,"PBC");
    # matwrite("matfile.mat", Dict(
    # 	"C1" => convert(Array,CTM["Cset"][1]),
    # 	"C2" => convert(Array,CTM["Cset"][2]),
    #     "C3" => convert(Array,CTM["Cset"][3]),
    #     "C4" => convert(Array,CTM["Cset"][4]),
    #     "T1" => convert(Array,CTM["Tset"][1]),
    #     "T2" => convert(Array,CTM["Tset"][2]),
    #     "T3" => convert(Array,CTM["Tset"][3]),
    #     "T4" => convert(Array,CTM["Tset"][4])
    # ); compress = false)

end;



function group_svd_components(U_set,S_set,V_set,spins)
    allspin=sort(unique(spins));
    spin_dim=deepcopy(allspin);
    spin_range=deepcopy(allspin);
    for cs=1:length(allspin)
        spin_range[cs]=findall(abs.(spins.-allspin[cs]).<1e-6)
        spin_dim[cs]=length(spin_range[cs])
    end


    Vtotal=Rep[SU₂](allspin[1]=>spin_dim[1]);
    for cs=2:length(allspin)
        Vtotal=Vtotal⊕ Rep[SU₂](allspin[cs]=>spin_dim[cs]);
    end
    vtem=space(V_set[1]',1);
    if vtem.dual
        Vtotal=Vtotal';
    end
    Um=TensorMap(randn,space(U_set[1],1)*space(U_set[1],2),Vtotal)*(0*im);
    Vm=Um*0;
    Sm=TensorMap(randn,Vtotal,Vtotal)*(0*im);



    for cs=1:length(allspin)
        Range=spin_range[cs];
        U_block=Um.data.values[cs];
        V_block=Vm.data.values[cs];
        S_block=Sm.data.values[cs];
        
        for ccc=1:spin_dim[cs]
            U=U_set[Range[ccc]];
            U_block[:,ccc]=U.data.values[1];

            V=V_set[Range[ccc]];
            V_block[:,ccc]=V.data.values[1];

            S=S_set[Range[ccc]];
            S_block[ccc,ccc]=S;
        end
        # Um.data.values[cs]=U_block*sqrt((2*allspin[cs]+1));
        # Vm.data.values[cs]=V_block*sqrt((2*allspin[cs]+1));
        Um.data.values[cs]=U_block;
        Vm.data.values[cs]=V_block;
        Sm.data.values[cs]=S_block;
    end


    
    return Um,Sm,Vm
end

function normalize_AA(M1,M2,M3,M5,M7,M8,M1_,M2_,M3_,M5_,M7_,M8_,AA)
    HR_svd_R_temp(x)=HR(x,M1,M2,M3,M5,M7,M8,M1_,M2_,M3_,M5_,M7_,M8_,AA);
    HR_svd_R_conj_temp(x)=HR_conj(x,M1,M2,M3,M5,M7,M8,M1_,M2_,M3_,M5_,M7_,M8_,AA);
    vr_init=permute(TensorMap(randn, space(M3_,3)*space(AA,2),SU₂Space(0=>1)), (1,2,3,),());# assume that the dominant eigenvector has total spin zero. If not, it will have three indeces and it's not Hermiitan.

    S_temp,_,_,_=svdsolve((HR_svd_R_temp,HR_svd_R_conj_temp), vr_init, 1,:LR, krylovdim=8);

    return S_temp[1]
end




function truncated_svd_truncate(spins,S_set,chi,multiplet_tol,trun_tol)
    S_set_dense=[];
    for cc=1:length(S_set)
        S_set_dense=vcat(S_set_dense,S_set[cc]*ones(Int(2*spins[cc]+1)));
    end
    sorted=sort(S_set_dense,rev=true);


    if length(sorted)>chi
        value_trun=sorted[chi+1];
    else
        value_trun=0;
    end
    value_max=maximum(sorted);

    for cd=1:length(sorted)
        if ((sorted[cd]/value_max)<trun_tol) | (sorted[cd]<=value_trun) |(abs((sorted[cd]-value_trun)/value_trun)<multiplet_tol)
            sorted[cd]=0;
        end
    end

    pos=findall(sorted.>0);
    value_trun=sorted[pos[end]];
    pos=findall(S_set.>value_trun)

    return pos

end




function truncate_multiplet(s,chi,multiplet_tol,trun_tol)
    #the multiplet is not due to su(2) symmetry
    s_dense=sort(diag(convert(Array,s)),rev=true);

    # println(s_dense/s_dense[1])

    if length(s_dense)>chi
        value_trun=s_dense[chi+1];
    else
        value_trun=0;
    end
    value_max=maximum(s_dense);

    s_Dict=convert(Dict,s);
    
    space_full=space(s,1);
    for sp in sectors(space_full)

        diag_elem=diag(s_Dict[:data][string(sp)]);
        for cd=1:length(diag_elem)
            if ((diag_elem[cd]/value_max)<trun_tol) | (diag_elem[cd]<=value_trun) |(abs((diag_elem[cd]-value_trun)/value_trun)<multiplet_tol)
                diag_elem[cd]=0;
            end
        end
        s_Dict[:data][string(sp)]=diagm(diag_elem);
    end
    s=convert(TensorMap,s_Dict);

    # s_=sort(diag(convert(Array,s)),rev=true);
    # s_=s_/s_[1];
    # print(s_)
    # @assert 1+1==3
    return s
end



function delet_zero_block(U,Σ,V)

    secs=blocksectors(Σ);
    sec_length=Vector{Int}(undef,length(secs))
    U_dict = convert(Dict,U)
    Σ_dict = convert(Dict,Σ)
    V_dict = convert(Dict,V)

    #ProductSpace(Rep[SU₂](0=>3, 1/2=>4, 1=>4, 3/2=>2, 2=>1))

    for cc =1:length(secs)
        c=secs[cc];
        if (size(diag(Σ_dict[:data][string(c)]),1)>0) & (sum(abs.(diag(Σ_dict[:data][string(c)])))>0)
            inds=findall(x->(abs.(x).>0), diag(Σ_dict[:data][string(c)]));
            U_dict[:data][string(c)]=U_dict[:data][string(c)][:,inds];
            Σ_dict[:data][string(c)]=Σ_dict[:data][string(c)][inds,inds];
            V_dict[:data][string(c)]=V_dict[:data][string(c)][inds,:];

            sec_length[cc]=length(inds);
        else
            delete!(U_dict[:data], string(c))
            delete!(V_dict[:data], string(c))
            delete!(Σ_dict[:data], string(c))
            sec_length[cc]=0;
        end
    end

    #define sector string
    sec_str="ProductSpace(Rep[SU₂](" *string(((dim(secs[1])-1)/2)) * "=>" * string(sec_length[1]);
    for cc=2:length(secs)
        sec_str=sec_str*", " * string(((dim(secs[cc])-1)/2)) * "=>" * string(sec_length[cc]);
    end
    sec_str=sec_str*"))"

    U_dict[:domain]=sec_str
    V_dict[:codomain]=sec_str
    Σ_dict[:domain]=sec_str
    Σ_dict[:codomain]=sec_str

    return convert(TensorMap, U_dict), convert(TensorMap, Σ_dict), convert(TensorMap, V_dict)
end