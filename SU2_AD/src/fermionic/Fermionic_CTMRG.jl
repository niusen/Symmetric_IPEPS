using LinearAlgebra:diag,I 
using TensorKit
using Zygote:@ignore_derivatives

function build_double_layer_swap(Ap,A)
    # println(space(Ap))
    # println(space(A))

    gate=@ignore_derivatives swap_gate(Ap,1,4); 
    @tensor Ap[:]:=Ap[1,-2,-3,2,-5]*gate[-1,-4,1,2];  
    gate=@ignore_derivatives swap_gate(Ap,2,3); 
    @tensor Ap[:]:=Ap[-1,1,2,-4,-5]*gate[-2,-3,1,2];  
    gate=@ignore_derivatives parity_gate(Ap,4); 
    @tensor Ap[:]:=Ap[-1,-2,-3,1,-5]*gate[-4,1];
    gate=@ignore_derivatives parity_gate(Ap,2); 
    @tensor Ap[:]:=Ap[-1,1,-3,-4,-5]*gate[-2,1];
    
    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5));    

    U_L=@ignore_derivatives unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    U_D=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    U_R=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # uMp,sMp,vMp=tsvd(Ap);
    # uMp=uMp*sMp;
    # uM,sM,vM=tsvd(A);
    # uM=uM*sM;

    U_tem=@ignore_derivatives unitary(fuse(space(A,1)*space(A,2)), space(A,1)*space(A,2))*(1+0*im);
    vM=U_tem*A;
    uM=U_tem';
    U_temp=@ignore_derivatives unitary(fuse(space(Ap,1)*space(Ap,2)), space(Ap,1)*space(Ap,2))*(1+0*im);
    vMp=U_temp*Ap;
    uMp=U_temp';

    uMp=permute(uMp,(1,2,3,),())
    uM=permute(uM,(1,2,3,),())
    Vp=@ignore_derivatives space(uMp,3);
    V=@ignore_derivatives space(vM,1);
    U=@ignore_derivatives unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

    @tensor double_LD[:]:=uMp[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vMp=permute(vMp,(1,2,3,4,),());
    vM=permute(vM,(1,2,3,4,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];

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

    ##########################
    AA_fused=permute(AA_fused,(1,2,3,4,));

    P_odd_Lp,_=@ignore_derivatives projector_parity(space(U_L',1));
    P_odd_Up,_=@ignore_derivatives projector_parity(space(U_U',2));
    P_odd_U,_=@ignore_derivatives projector_parity(space(U_U',3));

    @tensor isom_Lp[:]:=U_L[-1,4,3]*P_odd_Lp'[4,1]*P_odd_Lp[1,2]*U_L'[2,3,-2];
    @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2];
    @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6];
    @tensor AA_Lp_U[:]:=AA_fused[1,-2,-3,4]*isom_Lp[-1,1]*isom_U[-4,4];
    AA_fused=AA_fused-2*AA_Lp_U;
    @tensor AA_Up_U[:]:=AA_fused[-1,-2,-3,4]*isom_Up_U[-4,4];
    AA_fused=AA_fused-2*AA_Up_U;

    # @tensor U_LU[:]:=U_L'[-1,-2,-5]*U_U'[-6,-3,-4];
    # gate1=@ignore_derivatives swap_gate(U_LU,1,4);
    # gate2=@ignore_derivatives swap_gate(U_LU,3,4);
    # @tensor U_LU[:]:=U_LU[1,-2,-3,2,-5,-6]*gate1[-1,-4,1,2];
    # @tensor U_LU[:]:=U_LU[-1,-2,1,2,-5,-6]*gate2[-3,-4,1,2];
    # @tensor U_LU[:]:=U_LU[1,2,3,4,-3,-4]*U_L[-1,1,2]*U_U[3,4,-2];
    # @tensor AA_fused[:]:=AA_fused[1,-2,-3,2]*U_LU[-1,-4,1,2];

    P_odd_Dp,_=@ignore_derivatives projector_parity(space(U_D',1));
    P_odd_D,_=@ignore_derivatives projector_parity(space(U_D',2));
    P_odd_R,_=@ignore_derivatives projector_parity(space(U_R',3));
    @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2];
    @tensor isom_R[:]:=U_R[3,4,-1]*P_odd_R'[4,1]*P_odd_R[1,2]*U_R'[-2,3,2];
    @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2];
    @tensor AA_Dp_D[:]:=AA_fused[-1,2,-3,-4]*isom_Dp_D[-2,2];
    AA_fused=AA_fused-2*AA_Dp_D;
    @tensor AA_Dp_R[:]:=AA_fused[-1,2,3,-4]*isom_Dp[-2,2]*isom_R[-3,3];
    AA_fused=AA_fused-2*AA_Dp_R;

    # @tensor U_DR[:]:=U_D'[-1,-2,-5]*U_R'[-6,-3,-4];
    # gate1=@ignore_derivatives swap_gate(U_DR,1,2);
    # gate2=@ignore_derivatives swap_gate(U_DR,1,4);
    # @tensor U_DR[:]:=U_DR[1,2,-3,-4,-5,-6]*gate1[-1,-2,1,2];
    # @tensor U_DR[:]:=U_DR[1,-2,-3,2,-5,-6]*gate2[-1,-4,1,2];

    # @tensor U_DR[:]:=U_DR[1,2,3,4,-3,-4]*U_D[-1,1,2]*U_R[3,4,-2];
    # @tensor AA_fused[:]:=AA_fused[-1,1,2,-4]*U_DR[-2,-3,1,2];

    return AA_fused, U_L,U_D,U_R,U_U
end


# function build_double_layer_swap_old(Ap,A)
#     # println(space(Ap))
#     # println(space(A))
    

#     gate=@ignore_derivatives swap_gate(Ap,1,4); 
#     @tensor Ap[:]:=Ap[1,-2,-3,2,-5]*gate[-1,-4,1,2];  
#     gate=@ignore_derivatives swap_gate(Ap,2,3); 
#     @tensor Ap[:]:=Ap[-1,1,2,-4,-5]*gate[-2,-3,1,2];  
#     gate=@ignore_derivatives parity_gate(Ap,4); 
#     @tensor Ap[:]:=Ap[-1,-2,-3,1,-5]*gate[-4,1];
#     gate=@ignore_derivatives parity_gate(Ap,2); 
#     @tensor Ap[:]:=Ap[-1,1,-3,-4,-5]*gate[-2,1];
    



#     Ap=permute(Ap,(1,2,),(3,4,5))
#     A=permute(A,(1,2,),(3,4,5));
    

#     U_L=@ignore_derivatives unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
#     U_D=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
#     U_R=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
#     U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));


#     # uMp,sMp,vMp=tsvd(Ap);
#     # uMp=uMp*sMp;
#     # uM,sM,vM=tsvd(A);
#     # uM=uM*sM;

#     U_tem=@ignore_derivatives unitary(fuse(space(A,1)*space(A,2)), space(A,1)*space(A,2))*(1+0*im);
#     vM=U_tem*A;
#     uM=U_tem';
#     U_temp=@ignore_derivatives unitary(fuse(space(Ap,1)*space(Ap,2)), space(Ap,1)*space(Ap,2))*(1+0*im);
#     vMp=U_temp*Ap;
#     uMp=U_temp';

#     uMp=permute(uMp,(1,2,3,),())
#     uM=permute(uM,(1,2,3,),())
#     Vp=@ignore_derivatives space(uMp,3);
#     V=@ignore_derivatives space(vM,1);
#     U=@ignore_derivatives unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

#     @tensor double_LD[:]:=uMp[-1,-2,1]*U'[1,-3,-4];
#     @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

#     vMp=permute(vMp,(1,2,3,4,),());
#     vM=permute(vM,(1,2,3,4,),());

#     @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
#     @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];

#     #display(space(double_RU))

#     double_LD=permute(double_LD,(1,2,),(3,4,5,));
#     double_LD=U_L*double_LD;
#     double_LD=permute(double_LD,(2,3,),(1,4,));
#     double_LD=U_D*double_LD;
#     double_LD=permute(double_LD,(2,1,),(3,));
#     #display(space(double_LD))
#     double_RU=permute(double_RU,(1,4,5,),(2,3,));
#     double_RU=double_RU*U_R;
#     double_RU=permute(double_RU,(1,4,),(2,3,));
#     double_RU=double_RU*U_U;
#     double_LD=permute(double_LD,(1,2,),(3,));
#     double_RU=permute(double_RU,(1,),(2,3,));
#     AA_fused=double_LD*double_RU;


#     ##########################
#     @tensor U_LU[:]:=U_L'[-1,-2,-5]*U_U'[-6,-3,-4];
#     gate1=@ignore_derivatives swap_gate(U_LU,1,4);
#     gate2=@ignore_derivatives swap_gate(U_LU,3,4);
#     @tensor U_LU[:]:=U_LU[1,-2,-3,2,-5,-6]*gate1[-1,-4,1,2];
#     @tensor U_LU[:]:=U_LU[-1,-2,1,2,-5,-6]*gate2[-3,-4,1,2];
#     @tensor U_LU[:]:=U_LU[1,2,3,4,-3,-4]*U_L[-1,1,2]*U_U[3,4,-2];
#     @tensor AA_fused[:]:=AA_fused[1,-2,-3,2]*U_LU[-1,-4,1,2];#this line is slow


#     @tensor U_DR[:]:=U_D'[-1,-2,-5]*U_R'[-6,-3,-4];
#     gate1=@ignore_derivatives swap_gate(U_DR,1,2);
#     gate2=@ignore_derivatives swap_gate(U_DR,1,4);
#     @tensor U_DR[:]:=U_DR[1,2,-3,-4,-5,-6]*gate1[-1,-2,1,2];
#     @tensor U_DR[:]:=U_DR[1,-2,-3,2,-5,-6]*gate2[-1,-4,1,2];

#     @tensor U_DR[:]:=U_DR[1,2,3,4,-3,-4]*U_D[-1,1,2]*U_R[3,4,-2];
#     @tensor AA_fused[:]:=AA_fused[-1,1,2,-4]*U_DR[-2,-3,1,2];#this line is slow

#     return AA_fused, U_L,U_D,U_R,U_U
# end


function init_CTM_swap(chi,A,AA_fused, U_L,U_D,U_R,U_U,type,CTM_ite_info)
    @ignore_derivatives  if CTM_ite_info
        display("initialize CTM")
    end

    CTM=[];

    Cset=Cset_struc(A,A,A,A)
    Tset=Tset_struc(A,A,A,A)

    
    if type=="PBC"

        @tensor C1[:]:=AA_fused[1,-1,-2,3]*U_L'[2,2,1]*U_U'[3,4,4];
        @tensor C2[:]:=AA_fused[-1,-2,3,1]*U_U'[1,2,2]*U_R'[3,4,4];
        @tensor C3[:]:=AA_fused[-2,3,1,-1]*U_R'[1,2,2]*U_D'[4,4,3];
        @tensor C4[:]:=AA_fused[1,3,-1,-2]*U_L'[2,2,1]*U_D'[4,4,3];

        @tensor T4[:]:=AA_fused[1,-1,-2,-3]*U_L'[2,2,1];
        @tensor T1[:]:=AA_fused[-1,-2,-3,1]*U_U'[1,2,2];
        @tensor T2[:]:=AA_fused[-2,-3,1,-1]*U_R'[1,2,2];
        @tensor T3[:]:=AA_fused[-3,1,-1,-2]*U_D'[2,2,1];

        Cset=Cset_struc(C1,C2,C3,C4);
        Tset=Tset_struc(T1,T2,T3,T4);
    
        CTM=CTM_struc(Cset,Tset)

    elseif type=="random"
        C1=TensorMap(randn,space(AA_fused,2),space(AA_fused,3)')
        C2=TensorMap(randn,space(AA_fused,1),space(AA_fused,2)')
        C3=TensorMap(randn,space(AA_fused,4),space(AA_fused,1)')
        C4=TensorMap(randn,space(AA_fused,3),space(AA_fused,4)')

        T4=TensorMap(randn,space(AA_fused,2)*space(AA_fused,3),space(AA_fused,4)')
        T1=TensorMap(randn,space(AA_fused,1)*space(AA_fused,2),space(AA_fused,3)')
        T2=TensorMap(randn,space(AA_fused,4)*space(AA_fused,1),space(AA_fused,2)')
        T3=TensorMap(randn,space(AA_fused,3)*space(AA_fused,4),space(AA_fused,1)')

        Cset=Cset_struc(C1,C2,C3,C4);
        Tset=Tset_struc(T1,T2,T3,T4);
    
        CTM=CTM_struc(Cset,Tset)
    end

    CTM=CTM_struc(Cset, Tset);
    return CTM
end

function fermi_CTMRG(A,chi,init, CTM0,ctm_setting)
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
        AA_fused, U_L,U_D,U_R,U_U=build_double_layer_swap(A',A);
        #AA_fused, U_L,U_D,U_R,U_U=build_double_layer(A,[]);
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
        CTM=init_CTM_swap(chi,A,AA_fused, U_L,U_D,U_R,U_U,init.init_type,CTM_ite_info)
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
