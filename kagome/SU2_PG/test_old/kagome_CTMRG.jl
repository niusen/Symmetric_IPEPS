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


function CTMRG(A,chi,conv_check,tol,init)

    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)
    #initial corner transfer matrix
        #initial corner transfer matrix
    if isempty(init["CTM"])
        CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A,init["init_type"]);
    else
        CTM=init;
    end

    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    conv_check="singular_value"

    ss_old=ones(chi)*2;
    d=2;
    rho_old=Matrix(I,d^3,d^3);

    #Iteration
    CTM_ite_nums=50;
    display("start CTM iterations:")
    for ci=1:CTM_ite_nums
        direction_order=[3,4,1,2];
        for direction in direction_order
            Cset,Tset=CTM_ite(Cset, Tset, AA_fused, chi, direction);
        end
        if conv_check=="singular_value" #check convergence of singular value
            _,ss_new,_=svd(permute(Cset[1],(1,),(2,)));
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
            er=norm(dss);
            println("CTMRG iteration: "*string(ci)*", CTMRG err: "*string(er));
            if er<tol
                break;
            end
            ss_old=ss_new;
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
    return CTM, AA_fused, U_L,U_D,U_R,U_U

end

function CTM_ite(Cset, Tset, AA_fused, chi, direction)

    AA=permute(AA_fused, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());

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
    _, RMup=leftorth(permute(MMup*MMup_reflect,(3,4,),(1,2,)));
    _, RMlow=leftorth(permute(MMlow*MMlow_reflect,(3,4,),(1,2,)));


    RMlow=permute(RMlow,(2,3,),(1,));
    M=RMup*RMlow;



    uM,sM,vM = tsvd(M; trunc=truncdim(chi));

    println(diag(convert(Array,sM)))

    #uM,sM,vM = tsvd(M; trunc=trunbelow(1e-34)); #this looks to truncate absolute value, not relative value
    #display(convert(Array,sM))
    #display(convert(Array,inv(sM)))
    sM=sM/norm(sM)
    sM_inv=inv(sM);
    sM_dense=convert(Array,sM)
    #display(sM_dense)
    trun_tol=1e-12
    for c1=1:size(sM_dense,1)
        if sM_dense[c1,c1]<trun_tol
            sM_dense[c1,c1]=0;
        end
    end
    #display(sM_dense)
    #display(pinv.(sM_dense))

    #display(sM_inv)
    #display(convert(Array,sM_inv))
    sM_inv_sqrt=sqrt.(convert(Array,sM_inv))
    #display(space(sM_inv))
    #display(sM_inv_sqrt)
    sM_inv_sqrt=TensorMap(pinv.(sqrt.(sM_dense)),codomain(sM_inv)←domain(sM_inv))

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

function init_CTM(chi,A,type)

    display("initialize CTM")
    #numind(A)
    #numin(A)
    #numout(A)
    CTM=[];
    Cset=Vector(undef,4);
    Tset=Vector(undef,4);
    #space(A,1)
    @time if type=="PBC"
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