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

function CTMRG(A,chi,init,ctm_setting)
    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)
    ########################
    CTM_trun_tol=ctm_setting.CTM_trun_tol;
    CTM_ite_info=ctm_setting.CTM_ite_info;
    CTM_conv_info=ctm_setting.CTM_conv_info;
    projector_strategy=ctm_setting.projector_strategy;
    CTM_trun_svd=ctm_setting.CTM_trun_svd;
    svd_lanczos_tol=ctm_setting.svd_lanczos_tol;
    CTM_ite_nums=ctm_setting.CTM_ite_nums;
    #######################
    if (CTM_trun_svd==true) & (projector_strategy=="4x4")
        println("Attention: truncated svd with 4x4 projector could give large error");
    end


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
    ctm_setting.conv_check="singular_value"

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
            Cset,Tset=CTM_ite(Cset, Tset, AA_rotated[direction], chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol);
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
        


        if ctm_setting.conv_check=="singular_value" #check convergence of singular value
            er1,ss_new1=spectrum_conv_check(ss_old1,Cset[1]);
            er2,ss_new2=spectrum_conv_check(ss_old2,Cset[2]);
            er3,ss_new3=spectrum_conv_check(ss_old3,Cset[3]);
            er4,ss_new4=spectrum_conv_check(ss_old4,Cset[4]);

            er=maximum([er1,er2,er3,er4]);
            ite_err=er;
            if CTM_ite_info
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

function CTM_ite(Cset, Tset, AA, chi, direction, trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol)

    
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
    elseif projector_strategy=="4x2"
        RMup=permute(MMup,(3,4,),(1,2,));
        RMlow=MMlow;
    end

    M=RMup*RMlow;

    if CTM_trun_svd
        uM,sM,vM, M=truncated_svd_method(M,chi+20,svd_lanczos_tol);

        # TT_=uM*sM*vM;
        # uM0,sM0,vM0 = tsvd(M; trunc=truncdim(chi+20));
        # TT=uM0*sM0*vM0;
        # println("error of truncated svd: "*string(norm(TT_-TT)/norm(TT)))
    else
        uM,sM,vM = tsvd(M; trunc=truncdim(chi+20));
    end
    

    multiplet_tol=1e-5;
    uM,sM,vM,sM_inv_sqrt=treat_svd_results(uM,sM,vM,chi,multiplet_tol,trun_tol);


    PM_inv=RMlow*vM'*sM_inv_sqrt;
    PM=sM_inv_sqrt*uM'*RMup;
    PM=permute(PM,(2,3,),(1,));


    @tensor M5tem[:]:=Tset[mod1(direction-1,4)][4,3,1]*AA[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
    @tensor M1tem[:]:=Cset[mod1(direction,4)][1,2]*Tset[mod1(direction,4)][2,3,-2]*PM_inv[1,3,-1];
    @tensor M7tem[:]:=Cset[mod1(direction-1,4)][1,2]*Tset[mod1(direction-2,4)][-1,3,1]* PM[2,3,-2];



    Cset[mod1(direction,4)]=M1tem/norm(M1tem);
    Tset[mod1(direction-1,4)]=M5tem/norm(M5tem);
    Cset[mod1(direction-1,4)]=M7tem/norm(M7tem);
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









function truncate_multiplet(s,chi,multiplet_tol,trun_tol)
    #the multiplet is not due to su(2) symmetry
    s_dense=sort(abs.(diag(convert(Array,s))),rev=true);

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


function treat_svd_results(uM,sM,vM,chi,multiplet_tol,trun_tol)
    sM=truncate_multiplet(sM,chi,multiplet_tol,trun_tol);
    
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

    return uM,sM,vM,sM_inv_sqrt
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


    Vtotal=Rep[SU₂](allspin[1]=>spin_dim[1]);
    for cs=2:length(allspin)
        Vtotal=Vtotal⊕ Rep[SU₂](allspin[cs]=>spin_dim[cs]);
    end

    Um=TensorMap(randn,VL,Vtotal)*(0*im);
    Vm=TensorMap(randn,Vtotal,VR)*(0*im);
    Sm=TensorMap(randn,Vtotal,Vtotal)*(0*im);


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

function truncated_svd_method(M,chi,svd_lanczos_tol)
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
    
    # println(S_set)

    pos=truncated_svd_truncate(spins,S_set,chi);

    spins=spins[pos];
    S_set=S_set[pos];
    U_set=U_set[pos];
    V_set=V_set[pos];


    VL=space(M,1)*space(M,2);
    VR=space(M,3)*space(M,4);
    Um,Sm,Vm=group_svd_components(U_set,S_set,V_set,spins,VL,VR);

    U1=unitary(space(M,1)*space(M,2),space(Um,1));
    Um=U1*Um;
    U2=unitary(space(Vm,2)',space(M,3)'*space(M,4)');
    Vm=Vm*U2;

    
    
    return Um,Sm,Vm, M

end