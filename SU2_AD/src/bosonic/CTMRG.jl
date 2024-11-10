using LinearAlgebra:diag,diagm,I 
using TensorKit
using Zygote:@ignore_derivatives

function build_double_layer(A,operator)
    #display(space(A))
    A=permute(A,(1,2,),(3,4,5));
    U_L=@ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1))*(1+0*im);
    U_D=@ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2))*(1+0*im);
    # U_R=(U_L)';
    # U_U=(U_D)';
    U_R=@ignore_derivatives unitary(space(A, 3) ⊗ space(A, 3)', fuse(space(A, 3)' ⊗ space(A, 3)))*(1+0*im);
    U_U=@ignore_derivatives unitary(space(A, 4) ⊗ space(A, 4)', fuse(space(A, 4)' ⊗ space(A, 4)))*(1+0*im);
    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    # #uM,sM,vM=tsvd(A; trunc=truncerr(1e-10));
    # uM,sM,vM=tsvd(A);
    # uM=uM*sM
    U_tem=@ignore_derivatives unitary(fuse(space(A,1)*space(A,2)), space(A,1)*space(A,2))*(1+0*im);
    vM=U_tem*A;
    uM=U_tem';
    @assert(norm(uM*vM-A)/norm(A)<1e-12);

    uM=permute(uM,(1,2,3,),())
    V=space(vM,1);
    U=@ignore_derivatives unitary(fuse(V' ⊗ V), V' ⊗ V)*(1+0*im);
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

    # A=permute(A,(1,2,3,4,5,));
    # if operator==[]
    #     @tensor AA_fused[:]:=A'[2,4,6,8,1]*A[3,5,7,9,1]*U_L[-1,2,3]*U_D[-2,4,5]*U_R[6,7,-3]*U_U[8,9,-4];
    # else
    #     @tensor Ap[:]:=A'[-1,-2,-3,-4,1]*operator[-5,1];
    #     @tensor AA_fused[:]:=Ap[2,4,6,8,1]*A[3,5,7,9,1]*U_L[-1,2,3]*U_D[-2,4,5]*U_R[6,7,-3]*U_U[8,9,-4];
    # end

    return AA_fused, U_L,U_D,U_R,U_U
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

function CTMRG(A,chi,init, CTM0,ctm_setting)
    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)
    ########################
    CTM_trun_tol=ctm_setting.CTM_trun_tol;
    CTM_ite_info=ctm_setting.CTM_ite_info;
    CTM_conv_info=ctm_setting.CTM_conv_info;
    projector_strategy=ctm_setting.projector_strategy;
    CTM_trun_svd=ctm_setting.CTM_trun_svd;
    svd_lanczos_tol=ctm_setting.svd_lanczos_tol;
    CTM_ite_nums=ctm_setting.CTM_ite_nums;
    construct_double_layer=ctm_setting.construct_double_layer;
    #######################
    if (CTM_trun_svd==true) & (projector_strategy=="4x4")
        println("Attention: truncated svd with 4x4 projector could give large error");
    end


    #initial corner transfer matrix

    if init.reconstruct_AA
        AA_fused, U_L,U_D,U_R,U_U=build_double_layer(A,[]);
        AA_memory=@ignore_derivatives Base.summarysize(AA_fused)/1024/1024;
        @ignore_derivatives if CTM_ite_info
            println("Memory cost of double layer tensor: "*string(AA_memory)*" Mb.");flush(stdout);
        end
    else
        AA_fused=auxi_tensors.AA_fused;
        U_L=auxi_tensors.U_L;
        U_D=auxi_tensors.U_D;
        U_R=auxi_tensors.U_R;
        U_U=auxi_tensors.U_U;
    end

    if init.reconstruct_CTM
        CTM= init_CTM(chi,A,init.init_type,CTM_ite_info);
    else
        CTM=deepcopy(CTM0);
    end
        
        

    Cset=CTM.Cset;
    Tset=CTM.Tset;
    ctm_setting.conv_check="singular_value"

    ss_old1=ones(chi)*2;
    ss_old2=ones(chi)*2;
    ss_old3=ones(chi)*2;
    ss_old4=ones(chi)*2;
    d=2;
    rho_old=Matrix(I,d^3,d^3);

    #Iteration

    print_corner=false;
    @ignore_derivatives if print_corner
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

    if construct_double_layer
        AA_rotated=rotate_AA(AA_fused,construct_double_layer);
    else
        AA_rotated=rotate_AA(A,construct_double_layer);
    end

    @ignore_derivatives  if CTM_ite_info
        println("start CTM iterations:")
    end
    ite_num=0;
    ite_err=1;
    for ci=1:CTM_ite_nums
        ite_num=ci;
        #direction_order=[1,2,3,4];
        #direction_order=[4,1,2,3];
        direction_order=[3,4,1,2];
        #direction_order=[1];
        
        for direction in direction_order
            if ctm_setting.grad_checkpoint #use checkpoint to save memory
                Cset,Tset= Zygote.checkpointed(CTM_ite, Cset, Tset, get_Tset(AA_rotated, direction), chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer);
            else
                Cset,Tset=CTM_ite(Cset, Tset, get_Tset(AA_rotated, direction), chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer);
            end
        end

        print_corner=false;
        @ignore_derivatives if print_corner
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
        


        if ctm_setting.conv_check=="singular_value" #check convergence of singular value
            er1,ss_new1=@ignore_derivatives spectrum_conv_check(ss_old1,Cset.C1);
            er2,ss_new2=@ignore_derivatives spectrum_conv_check(ss_old2,Cset.C2);
            er3,ss_new3=@ignore_derivatives spectrum_conv_check(ss_old3,Cset.C3);
            er4,ss_new4=@ignore_derivatives spectrum_conv_check(ss_old4,Cset.C4);

            er=@ignore_derivatives max(er1,er2,er3,er4);
            ite_err=er;
            @ignore_derivatives  if CTM_ite_info
                println("CTMRG iteration: "*string(ci)*", CTMRG err: "*string(er));flush(stdout);
            end
            if er<ctm_setting.CTM_conv_tol
                break;
            end
            ss_old1=ss_new1;
            ss_old2=ss_new2;
            ss_old3=ss_new3;
            ss_old4=ss_new4;
        elseif ctm_setting.conv_check=="density_matrix" #check reduced density matrix

            # ob_opts.SiteNumber=1;
            # CTM_tem.Cset=Cset;
            # CTM_tem.Tset=Tset;
            # rho_new=ob_CTMRG(CTM_tem,A,ob_opts).A;
            # er=sum(sum((abs(rho_old-rho_new))));
            # disp(['CTMRG iteration: ',num2str(ci),' CTMRG err: ',num2str(er)]);
            # if er<ctm_setting.CTM_conv_tol
            #     break;
            # end
            # rho_old=rho_new;
        end

        # if ci==CTM_ite_nums
        #     display(er)
        #     warn("CTMRG does not converge: " * string(er));
        # end
    end

    CTM=CTM_struc(Cset,Tset);

    if CTM_conv_info
        return CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err
    else
        return CTM, AA_fused, U_L,U_D,U_R,U_U
    end

end
function rotate_AA(AA_fused,construct_double_layer)
    #AA_rotated=Vector(undef,4);
    AA_rotated=Tset_struc(AA_fused,AA_fused,AA_fused,AA_fused);
    for direction=1:4
        if construct_double_layer
            AA_rotated_tem=permute(AA_fused, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
            AA_rotated=set_Tset(AA_rotated,AA_rotated_tem,direction);
        else
            AA_rotated_tem=permute(AA_fused, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),5,),());
            AA_rotated=set_Tset(AA_rotated,AA_rotated_tem,direction);
        end
    end
    return AA_rotated
end

function CTM_ite(Cset, Tset, AA, chi, direction, trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer)

    #jldsave("CTM_ite.jld2"; Cset,Tset,AA,chi,direction)
    #if construct_double_layer==false, then AA is single layer tensor
    
    M1=get_Cset(Cset, mod1(direction,4));
    M2=get_Tset(Tset, mod1(direction,4));
    M3=get_Tset(Tset, mod1(direction-1,4));
    #M4=AA;
    M5=M3;
    #M6=M4;
    M7=get_Cset(Cset, mod1(direction-1,4));
    M8=get_Tset(Tset, mod1(direction-2,4));

    if construct_double_layer
        @tensor MMup[:]:=M1[1,2]*M2[2,3,-3]*M3[-1,4,1]*AA[4,-2,-4,3];
        @tensor MMlow[:]:=M5[1,3,-1]*AA[3,4,-4,-2]*M7[2,1]*M8[-3,4,2];
    else
        @tensor MMup[:]:=M1[1,2]*M2[2,6,4,-4]*M3[-1,5,3,1]*AA[3,-3,-6,4,7]*AA'[5,-2,-5,6,7];
        @tensor MMlow[:]:=M5[2,6,4,-1]*AA[4,3,-6,-3,7]*M7[1,2]*M8[-4,5,3,1]*AA'[6,5,-5,-2,7];
    end

    #define permute index that is heavily used
    permut_ind1=[];
    permut_ind2=[];
    if construct_double_layer
        permut_ind1=(1,2,);
        permut_ind2=(3,4,);
    else
        permut_ind1=(1,2,3,);
        permut_ind2=(4,5,6,);
    end

    MMup=permute(MMup,permut_ind1,permut_ind2)
    MMlow=permute(MMlow,permut_ind1,permut_ind2)

    if projector_strategy=="4x4"
        M1_=get_Cset(Cset, mod1(direction+1,4));
        M2_=get_Tset(Tset, mod1(direction,4));
        M3_=get_Tset(Tset, mod1(direction+1,4));
        #M4_=M4;
        M5_=M3_;
        #M6_=M4_;
        M7_=get_Cset(Cset, mod1(direction-2,4));
        M8_=get_Tset(Tset, mod1(direction-2,4));

        if construct_double_layer
            @tensor MMup_reflect[:]:=M2_[-1,3,1]* M1_[1,2]* AA[-2,-4,4,3]* M3_[2,4,-3];
            @tensor MMlow_reflect[:]:=M5_[-4,-3,2]*M8_[1,-2,-1]*M7_[2,1];
            @tensor MMlow_reflect[:]:=MMlow_reflect[-1,1,2,-3]*AA[-2,1,2,-4];
        else
            @tensor MMup_reflect[:]:=M2_[-1,5,3,1]* M1_[1,2]* AA[-3,-6,4,3,7]* M3_[2,6,4,-4]*AA'[-2,-5,6,5,7];
            @tensor MMlow_reflect[:]:=M8_[2,6,4,-1]*AA[-3,4,3,-6,7]*M7_[1,2]*M5_[-4,5,3,1]*AA'[-2,6,5,-5,7];
        end
        MMup_reflect=permute(MMup_reflect,permut_ind1,permut_ind2)
        MMlow_reflect=permute(MMlow_reflect,permut_ind1,permut_ind2)
    end

    
    if projector_strategy=="4x4"
        RMup=permute(MMup*MMup_reflect,permut_ind2,permut_ind1);
        RMlow=MMlow*MMlow_reflect;
    elseif projector_strategy=="4x2"
        RMup=permute(MMup,permut_ind2,permut_ind1);
        RMlow=MMlow;
    end

    #without the below normalization, the gradiant of svd will explode!!!
    #Also we should ignore derivative of this step, otherwise it seems that the normalization factor will accumulate and the grad explode again!!!
    
    RMlow_norm=norm(RMlow);
    RMlow= RMlow/RMlow_norm;

    RMup_norm=norm(RMup);
    RMup= RMup/RMup_norm;

    # RMlow=@ignore_derivatives RMlow/norm(RMlow);
    # RMup=@ignore_derivatives RMup/norm(RMup);

    # norm_low=@ignore_derivatives norm(RMlow);
    # norm_up=@ignore_derivatives norm(RMup);
    
    # norm_total=@ignore_derivatives sqrt(10^round(log10(norm_low*norm_up)))
    # RMlow=RMlow/norm_total;#gradient of this step can't be ignored, otherwise final grad will be oncorrect
    # RMup=RMup/norm_total;

    # RMlow=normalize_no_grad(RMlow);
    # RMup=normalize_no_grad(RMup);

    # RMup=show_grad(RMup);
    # RMlow=show_grad(RMlow);

    M=RMup*RMlow;

    #M=@ignore_derivatives M/norm(M);

    #M=show_grad(M);
    
    #jldsave("hard_tensor.jld2"; M)
    
    if isa(space(M,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})#Z2 symmetry
        chi_extra=3;
    elseif isa(space(M,1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}) #U1 symmetry
        chi_extra=4;
    elseif isa(space(M,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}) #SU(2) symmetry
        chi_extra=20;
    elseif isa(space(M,1), GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}) #U1 x SU(2)
        chi_extra=20;
    elseif isa(space(M,1), ComplexSpace)
        chi_extra=1;
    end


    if CTM_trun_svd
        uM,sM,vM, M=truncated_svd_method(M,chi+chi_extra,svd_lanczos_tol,construct_double_layer);

        # TT_=uM*sM*vM;
        # uM0,sM0,vM0 = tsvd(M; trunc=truncdim(chi+chi_extra));
        # TT=uM0*sM0*vM0;
        # println("error of truncated svd: "*string(norm(TT_-TT)/norm(TT)))
    else
        #uM,sM,vM = tsvd(M; trunc=truncdim(chi+20));
        uM,sM,vM = my_tsvd(M; trunc=truncdim(chi+chi_extra));
        #uM,sM,vM = tsvd(M);
    end

    # uM=show_grad(uM);
    # sM=show_grad(sM);
    # vM=show_grad(vM);
    
    # println(sM.data.values)
    sM_norm=norm(sM);
    sM=sM/sM_norm;
    # println(sM.data.values)treat_svd_results
    multiplet_tol=1e-5;

    sM_inv_sqrt=sdiag_inv_sqrt(sM);
    #sM_inv_sqrt=@ignore_derivatives unitary(space(sM,1),space(sM,1));


    PM_inv=RMlow*vM'*sM_inv_sqrt;
    PM=sM_inv_sqrt*uM'*RMup;


    # PM_inv=@ignore_derivatives unitary(space(RMlow,1)*space(RMlow,2), fuse(space(RMlow,1)*space(RMlow,2)))
    # PM=@ignore_derivatives unitary(fuse(space(RMup,3)*space(RMup,4)), space(RMup,3)'*space(RMup,4)')

    # println(space(PM_inv_))
    # println(space(PM_inv))

    # println(space(PM_))
    # println(space(PM))

    PM=permute(PM,(permut_ind1.+1),(1,));
    #println([norm(PM),norm(PM_inv)])



    if construct_double_layer
        @tensor M5tem[:]:=get_Tset(Tset, mod1(direction-1,4))[4,3,1]*AA[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
        @tensor M1tem[:]:=get_Cset(Cset, mod1(direction,4))[1,2]*get_Tset(Tset, mod1(direction,4))[2,3,-2]*PM_inv[1,3,-1];
        @tensor M7tem[:]:=get_Cset(Cset, mod1(direction-1,4))[1,2]*get_Tset(Tset, mod1(direction-2,4))[-1,3,1]* PM[2,3,-2];
    else
        @tensor M5tem[:]:=get_Tset(Tset, mod1(direction-1,4))[1,5,3,7]*AA[3,2,-3,8,6]* PM_inv[1,4,2,-1]* PM[7,9,8,-4]*AA'[5,4,-2,9,6];
        @tensor M1tem[:]:=get_Cset(Cset, mod1(direction,4))[1,2]*get_Tset(Tset, mod1(direction,4))[2,3,4,-2]*PM_inv[1,3,4,-1];
        @tensor M7tem[:]:=get_Cset(Cset, mod1(direction-1,4))[1,2]*get_Tset(Tset, mod1(direction-2,4))[-1,3,4,1]* PM[2,3,4,-2];
    end

    
    norm_M1=norm(M1tem);
    C1_tem= M1tem/norm_M1; #somehow I must ignore grad of such normalization, otherwise error will occur in the checkpoint punction

    T_norm=norm(M5tem);
    T4_tem= M5tem/T_norm;
    
    norm_M7=norm(M7tem);
    C4_tem= M7tem/norm_M7;

    
    

    Cset=set_Cset(Cset, C1_tem,mod1(direction,4));
    Cset=set_Cset(Cset, C4_tem, mod1(direction-1,4));
    Tset=set_Tset(Tset, T4_tem, mod1(direction-1,4));
    return Cset,Tset

    # if direction==1
    #     return C1_tem, Cset.C2, Cset.C3, C4_tem, Tset.T1, Tset.T2, Tset.T3, T4_tem
    # elseif direction==2
    #     return C4_tem, C1_tem, Cset.C3, Cset.C4, T4_tem, Tset.T2, Tset.T3, Tset.T4
    # elseif direction==3
    #     return Cset.C1, C4_tem, C1_tem, Cset.C4, Tset.T1, T4_tem, Tset.T3, Tset.T4
    # elseif direction==4
    #     return Cset.C1, Cset.C2, C4_tem, C1_tem, Tset.T1, Tset.T2, T4_tem, Tset.T4
    # end
    # if direction==1
    #     return C1_tem, Cset.C2, Cset.C3, C4_tem, Tset
    # elseif direction==2
    #     return C4_tem, C1_tem, Cset.C3, Cset.C4, Tset
    # elseif direction==3
    #     return Cset.C1, C4_tem, C1_tem, Cset.C4, Tset
    # elseif direction==4
    #     return Cset.C1, Cset.C2, C4_tem, C1_tem, Tset
    # end
end






function init_CTM(chi,A,type,CTM_ite_info)
    @ignore_derivatives  if CTM_ite_info
        display("initialize CTM")
    end
    #numind(A)
    #numin(A)
    #numout(A)
    CTM=[];
    #Cset=Vector{TensorMap}(undef,4);
    #Tset=Vector{TensorMap}(undef,4);
    Cset=Cset_struc(A,A,A,A)
    Tset=Tset_struc(A,A,A,A)
    # Cset=(A,A,A,A)
    # Tset=(A,A,A,A)
    #space(A,1)
    
    if type=="PBC"
        U_L=@ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1))*(1+0*im);
        U_D=@ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2))*(1+0*im);
        # U_R=(U_L)';
        # U_U=(U_D)';
        U_R=@ignore_derivatives unitary(space(A, 3) ⊗ space(A, 3)', fuse(space(A, 3)' ⊗ space(A, 3)))*(1+0*im);
        U_U=@ignore_derivatives unitary(space(A, 4) ⊗ space(A, 4)', fuse(space(A, 4)' ⊗ space(A, 4)))*(1+0*im);

        @tensor C1[:]:=A'[2,4,6,3,1]*A[2,5,7,3,1]*U_D[-1,4,5]*U_R[6,7,-2];
        @tensor C2[:]:=A'[4,6,3,2,1]*A[5,7,3,2,1]*U_L[-1,4,5]*U_D[-2,6,7];
        @tensor C3[:]:=A'[6,3,2,4,1]*A[7,3,2,5,1]*U_U[4,5,-1]*U_L[-2,6,7];
        @tensor C4[:]:=A'[2,3,6,4,1]*A[2,3,7,5,1]*U_R[6,7,-1]*U_U[4,5,-2];
    
        @tensor T4[:]:=A'[2,3,5,7,1]*A[2,4,6,8,1]*U_D[-1,3,4]*U_R[5,6,-2]*U_U[7,8,-3];
        @tensor T1[:]:=A'[3,5,7,2,1]*A[4,6,8,2,1]*U_L[-1,3,4]*U_D[-2,5,6]*U_R[7,8,-3];
        @tensor T2[:]:=A'[5,7,2,3,1]*A[6,8,2,4,1]*U_U[3,4,-1]*U_L[-2,5,6]*U_D[-3,7,8];
        @tensor T3[:]:=A'[7,2,3,5,1]*A[8,2,4,6,1]*U_R[3,4,-1]*U_U[5,6,-2]*U_L[-3,7,8];
    
        Cset=Cset_struc(C1,C2,C3,C4);
        Tset=Tset_struc(T1,T2,T3,T4);
    
        CTM=CTM_struc(Cset,Tset)

    elseif type=="random"
    end

    CTM=CTM_struc(Cset, Tset);
    return CTM
end










# function truncate_multiplet(s,chi,multiplet_tol,trun_tol)
#     #the multiplet is not due to su(2) symmetry
#      s_dense=@ignore_derivatives sort(abs.(diag(convert(Array,s))),rev=true);


#     # println(s_dense/s_dense[1])

#     if length(s_dense)>chi
#         value_trun=s_dense[chi+1];
#     else
#         value_trun=0;
#     end
#     value_max=maximum(s_dense);

#     s_Dict=convert(Dict,s);
    
#     space_full=space(s,1);
#     for sp in sectors(space_full)

#         diag_elem=abs.(diag(s_Dict[:data][string(sp)]));
#         for cd=1:length(diag_elem)
#             if ((diag_elem[cd]/value_max)<trun_tol) | (diag_elem[cd]<=value_trun) |(abs((diag_elem[cd]-value_trun)/value_trun)<multiplet_tol)
#                 diag_elem[cd]=0;
#             end
#         end
#         s_Dict[:data][string(sp)]=diagm(diag_elem);
#     end
#     s=convert(TensorMap,s_Dict);

#     return s
# end

function truncate_multiplet(s,chi,multiplet_tol,trun_tol)
    #the multiplet is not due to su(2) symmetry
     s_dense=@ignore_derivatives sort(abs.(diag(convert(Array,s))),rev=true);


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

        diag_elem=abs.(diag(s_Dict[:data][string(sp)]));
        for cd=1:length(diag_elem)
            if ((diag_elem[cd]/value_max)<trun_tol) | (diag_elem[cd]<=value_trun) |(abs((diag_elem[cd]-value_trun)/value_trun)<multiplet_tol)
                diag_elem[cd]=0;
            end
        end
        s_Dict[:data][string(sp)]=diagm(diag_elem);
    end
    s=convert(TensorMap,s_Dict);

    s_dense=convert(Array,s);
    s_dense=unique(diag(s_dense));
    s_dense=sort(s_dense);
    if s_dense[1]==0 #return the minimal nonzero element
        return s_dense[2]
    else
        return s_dense[1]
    end

end




function group_svd_components(U_set,S_set,V_set,spins,VL,VR)
    VL=fuse(VL);
    VR=fuse(VR);
    allspin=sort(unique(spins));
    spin_dim=deepcopy(allspin);
    spin_range=deepcopy(allspin);
    for cs=1:length(allspin)
        spin_range[cs]=findall(abs.(spins.-allspin[cs]).<1e-6)
        spin_dim[cs]=length(spin_range[cs])
    end

    if typeof(VL)==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
        @assert length(spin_dim)==2
        Vtotal=Rep[ℤ₂](Int(allspin[1])=>spin_dim[1],Int(allspin[2])=>spin_dim[2]);
    else
        Vtotal=Rep[SU₂](allspin[1]=>spin_dim[1]);
        for cs=2:length(allspin)
            Vtotal=Vtotal⊕ Rep[SU₂](allspin[cs]=>spin_dim[cs]);
        end
    end

    Um=TensorMap(randn,VL,Vtotal)*(0*im);
    Vm=TensorMap(randn,Vtotal,VR)*(0*im);
    Sm=TensorMap(randn,Vtotal,Vtotal)*(0);


    for cs=1:length(allspin)
        Range=spin_range[cs];
        U_block=Um.data.values[cs]*0;
        V_block=Vm.data.values[cs]*0;
        S_block=Sm.data.values[cs]*0;
        
        for ccc=1:spin_dim[cs]
            U=U_set[Range[ccc]];
            
            U_block[:,ccc]=U;

            V=V_set[Range[ccc]];
            V_block[ccc,:]=V';

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



function truncated_svd_truncate(spins,S_set,chi)
    S_set_dense=[];
    for cc=1:length(S_set)
        S_set_dense=vcat(S_set_dense,S_set[cc]*ones(Int(2*spins[cc]+1)));
    end
    sorted=sort(S_set_dense,rev=true);



    if length(sorted)>chi
        value_trun=sorted[chi+1];
    else
        value_trun=sorted[end];
    end

    for cd=1:length(sorted)
        if  (sorted[cd]<=value_trun) 
            sorted[cd]=0;
        end
    end


    pos=findall(sorted.>0);
    value_trun=sorted[pos[end]];
    pos=findall(S_set.>value_trun)

    return pos

end


function normalize_AA(mm)
    S_temp,_,_,_=svdsolve(mm, 1,:LR, krylovdim=8);

    return S_temp[1]
end

function truncated_svd_method(M,chi,svd_lanczos_tol,construct_double_layer)
    # JLDnm="test.jld2";
    # init___=Dict([("M",M),("chi",chi),("trun_tol",trun_tol)]);
    # save(JLDnm, "init",init___);

    spins=Vector(undef,0);
    S_set=Vector(undef,0);
    U_set=Vector(undef,0);
    V_set=Vector(undef,0);

    #######
    S_max=normalize_AA(M.data.values[1]);
    M=M/S_max;

    #######

    
    for cs=1:length(M.data.keys)
        if typeof(space(M,1))==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
            spin=M.data.keys[cs].n;
            #println(spin)
            mm=M.data.values[cs];
            Random.seed!(1234)
            vr_init=randn(size(mm,2))+im*randn(size(mm,2));
            n_keep=min(size(mm,1),size(mm,2),Int(round(chi/(2*spin+1))));
        
    
            vr_temp=deepcopy(vr_init);
            ite_test=4;
            for cccc=1:ite_test
                vr_temp=mm*vr_temp;
                vr_temp=mm'*vr_temp;
            end
            norm_coe=norm(vr_temp)/norm(vr_init);
            norm_coe=norm_coe^(1/ite_test);
            if norm_coe<1e-14
                continue;
            end
        
            S,U,V,info=svdsolve(mm, n_keep,:LR, krylovdim=n_keep*3,tol=svd_lanczos_tol);
            S_set=vcat(S_set,S);U_set=vcat(U_set,U);V_set=vcat(V_set,V);
            spins=vcat(spins,spin*ones(length(S)));

        else
            spin=M.data.keys[cs].j;
            #println(spin)
            mm=M.data.values[cs];
            Random.seed!(1234)
            vr_init=randn(size(mm,2))+im*randn(size(mm,2));
            n_keep=min(size(mm,1),size(mm,2),Int(round(chi/(2*spin+1))));
        

            vr_temp=deepcopy(vr_init);
            ite_test=4;
            for cccc=1:ite_test
                vr_temp=mm*vr_temp;
                vr_temp=mm'*vr_temp;
            end
            norm_coe=norm(vr_temp)/norm(vr_init);
            norm_coe=norm_coe^(1/ite_test);
            if norm_coe<1e-14
                continue;
            end
        
        #@suppress begin
            S,U,V,info=svdsolve(mm, n_keep,:LR, krylovdim=n_keep*3,tol=svd_lanczos_tol);
            S_set=vcat(S_set,S);U_set=vcat(U_set,U);V_set=vcat(V_set,V);
            spins=vcat(spins,spin*ones(length(S)));
            #@assert info.converged >= minimum([n_eff,dim(full_space,sec)])
            
            # if ((S[1]/S_set[1])<1e-16)
            #     println("truncate at spin= "*string(spin))
            #     break;
            # end
        #end
        end
        
        

    end
    
    # println(S_set)

    pos=truncated_svd_truncate(spins,S_set,chi);

    spins=spins[pos];
    S_set=S_set[pos];
    U_set=U_set[pos];
    V_set=V_set[pos];

    if construct_double_layer
        VL=space(M,1)*space(M,2);
        VR=space(M,3)*space(M,4);
    else
        VL=space(M,1)*space(M,2)*space(M,3);
        VR=space(M,4)*space(M,5)*space(M,6);
    end

    Um,Sm,Vm=group_svd_components(U_set,S_set,V_set,spins,VL,VR);

    if construct_double_layer
        U1=@ignore_derivatives unitary(space(M,1)*space(M,2),space(Um,1));
        Um=U1*Um;
        U2=@ignore_derivatives unitary(space(Vm,2)',space(M,3)'*space(M,4)');
        Vm=Vm*U2;
    else
        U1=@ignore_derivatives unitary(space(M,1)*space(M,2)*space(M,3),space(Um,1));
        Um=U1*Um;
        U2=@ignore_derivatives unitary(space(Vm,2)',space(M,4)'*space(M,5)'*space(M,6)');
        Vm=Vm*U2;
    end

    
    
    return Um,Sm,Vm, M

end











# function sdiag_inv_sqrt(S::AbstractTensorMap)
#     toret = similar(S);
    
#     if sectortype(S) == Trivial
#         copyto!(toret.data,LinearAlgebra.diagm(LinearAlgebra.diag(S.data).^(-1/2)));
#     else
#         for (k,b) in blocks(S)
#             copyto!(blocks(toret)[k],(LinearAlgebra.diagm(LinearAlgebra.diag(b).^(-1/2))));
#         end
#     end
#     toret
# end

function sdiag_inv_sqrt(S::AbstractTensorMap)
    toret = similar(S);
    global chi,multiplet_tol,projector_trun_tol
    s_min=truncate_multiplet(S,chi,multiplet_tol,projector_trun_tol);
    if sectortype(S) == Trivial
        b=S.data;
        newdata=(diagm(diag(b).^(-1/2))).*(diagm(diag(b).>=(s_min)));
        copyto!(toret.data,newdata);
    else
        for (k,b) in blocks(S)
            
            copyto!(blocks(toret)[k],(diagm(diag(b).^(-1/2))).*(diagm(diag(b).>=(s_min))));
        end
    end
    toret
end


function ChainRulesCore.rrule(::typeof(sdiag_inv_sqrt),S::AbstractTensorMap)
    toret = sdiag_inv_sqrt(S);
    toret,c̄ -> (ChainRulesCore.NoTangent(),-1/2*_elementwise_mult(c̄,toret'^3))
end


function _elementwise_mult(a::AbstractTensorMap,b::AbstractTensorMap)
    dst = similar(a);
    for (k,block) in blocks(dst)
        copyto!(block,blocks(a)[k].*blocks(b)[k]);
    end
    dst
end