using LinearAlgebra
using TensorKit

function CTMRG(A,chi,conv_check,tol,init)
    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)

    #initial corner transfer matrix
    if isempty(init["CTM"])
        CTM=init_CTM(chi,A,init["init_type"]);
    else
        CTM=init;
    end
    Cset=CTM["Cset"];
    Tset=CTM["Tset"];


    ss_old=ones(chi)*2;
    d=2;
    rho_old=Matrix(I,d^3,d^3);

    #Iteration
    CTM_ite_nums=50;
    display("start CTM iterations:")
    for ci=1:CTM_ite_nums
        #told=Tset{4};
        #pold=Pset{4};
        direction_order=[1,2,3,4];
        for direction in direction_order
            #disp(['direction ', num2str(direction)])
            @time Cset, Tset=CTM_ite(Cset, Tset, A, chi, direction);
        end

        if conv_check=="singular_value" #check convergence of singular value
            _,ss_new,_=svd(permute(Cset[1],(1,),(2,)));
            ss_new=convert(Array,ss_new);
            ss_new=sort(diag(ss_new), rev=true);
            ss_old=ss_old/ss_old[1];
            ss_new=ss_new/ss_new[1];
            display(ss_new)
            if length(ss_old)>length(ss_new)
                dss=copy(ss_old);
                siz=length(ss_new)
            elseif length(ss_old)<=length(ss_new)
                dss=copy(ss_new);
                siz=length(ss_old)
            end
            dss[1:siz]=ss_old[1:siz]-ss_new[1:siz]
            er=norm(dss);
            display("CTMRG iteration: "*string(ci)*", CTMRG err: "*string(er));
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
    return CTM
end

function init_CTM(chi,A,type)
    display("initialize CTM")
    #numind(A)
    #numin(A)
    #numout(A)
    CTM=[]
    Cset=Vector(undef,4)
    Tset=Vector(undef,4)
    #space(A,1)
    if type=="PBC"
        for direction=1:4    
            inds=(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),5);
            A_rotate=permute(A,inds)
            Ap_rotate=A_rotate'

            @tensor M[:]:=Ap_rotate[1,-1,-3,2,3]*A_rotate[1,-2,-4,2,3];
            Cset[direction]=M;
            @tensor M[:]:=Ap_rotate[-1,-3,-5,1,2]*A_rotate[-2,-4,-6,1,2];
            Tset[direction]=M
        end

        #fuse legs
        ul_set=Vector(undef,4)
        ur_set=Vector(undef,4)
        for direction=1:2
            ul_set[direction]=unitary(fuse(space(Cset[direction], 3) ⊗ space(Cset[direction], 4)), space(Cset[direction], 3) ⊗ space(Cset[direction], 4))
            ur_set[direction]=unitary(fuse(space(Tset[direction], 5) ⊗ space(Tset[direction], 6)), space(Tset[direction], 5) ⊗ space(Tset[direction], 6))
        end
        for direction=3:4
            ul_set[direction]=unitary(fuse(space(Cset[direction], 3) ⊗ space(Cset[direction], 4))', space(Cset[direction], 3) ⊗ space(Cset[direction], 4))
            ur_set[direction]=unitary(fuse(space(Tset[direction], 5) ⊗ space(Tset[direction], 6))', space(Tset[direction], 5) ⊗ space(Tset[direction], 6))
        end
        for direction=1:4
            C=Cset[direction]
            ul=ur_set[mod1(direction-1,4)]
            ur=ul_set[direction]
            ulp=permute(ul',(3,),(1,2,))
            urp=permute(ur',(3,),(1,2,))
            #@tensor Cnew[(-1);(-2)]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4]
            @tensor Cnew[:]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4]#put all indices in tone side so that its adjoint has the same index order 
            Cset[direction]=Cnew

            T=Tset[direction]
            ul=ul_set[direction]
            ur=ur_set[direction]
            ulp=permute(ul',(3,),(1,2,))
            urp=permute(ur',(3,),(1,2,))
            #@tensor Tnew[(-1);(-2,-3,-4)]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4]
            @tensor Tnew[:]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4]#put all indices in tone side so that its adjoint has the same index order 
            Tset[direction]=Tnew
        end
    elseif type=="random"
    end
    CTM=Dict([("Cset", Cset), ("Tset", Tset)])
    return CTM
end;



function CTM_ite(Cset, Tset, A, chi, direction)
    #direction=1, 2, 3, 4:  left, up, right, down move

    A=permute(A, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),(5,));
    
    M1=Cset[mod1(direction,4)];#mod1(direction,4)
    M2=Tset[mod1(direction,4)];#mod1(direction,4)
    M3=Tset[mod1(direction-1,4)];#mod1(direction-1,4)
    M4=permute(A,(1,2,3,4,5,),());
    M5=Tset[mod1(direction-1,4)];#mod1(direction-1,4)
    M6=permute(A,(1,2,3,4,5,),());
    M7=Cset[mod1(direction-1,4)];#mod1(direction-1,4)
    M8=Tset[mod1(direction-2,4)];#mod1(direction-2,4)
    
    M1_reflect=Cset[mod1(direction+1,4)];
    M2_reflect=Tset[mod1(direction,4)];
    M3_reflect=Tset[mod1(direction+1,4)];
    M4_reflect=permute(A,(1,2,3,4,5,),());
    M5_reflect=Tset[mod1(direction+1,4)];
    M6_reflect=permute(A,(1,2,3,4,5,),());
    M7_reflect=Cset[mod1(direction-2,4)];
    M8_reflect=Tset[mod1(direction-2,4)];
    
    @tensor M3M4[:]:=M3[-1,1,2,-6]* M4'[1,-2,-4,-7,3]* M4[2,-3,-5,-8,3];  
    #M3M4=reshape(M3M4,[chi*D^2,D,D,chi*D^2]);
    @tensor M5M6[:]:=M5[-1,1,2,-6]* M6'[1,-2,-4,-7,3]* M6[2,-3,-5,-8,3];  
    #M5M6=reshape(M5M6,[chi*D^2,D,D,chi*D^2]);
    @tensor M1M2[:]:=M1[-1,1]* M2[1,-2,-3,-4];  
    #M1M2=reshape(M1M2, [chi*D^2, chi]);
    @tensor M7M8[:]:=M7[1,-2]* M8[-1,-3,-4,1];  
    #M7M8=reshape(M7M8, [chi, chi*D^2]);
    @tensor M1M2[:]:=M1[-1,1]* M2[1,-2,-3,-4];  
    #M1M2=reshape(M1M2, [chi*D^2, chi]);
    @tensor M7M8[:]:=M7[1,-2]* M8[-1,-3,-4,1];  
    #M7M8=reshape(M7M8, [chi, chi*D^2]);

    @tensor MMup[:]:=M1M2[1,2,3,-4]* M3M4[-1,-2,-3,-5,-6,1,2,3]; 
    MMup=permute(MMup,(1,2,3,),(4,5,6,))
    #MMup=reshape(MMup, [chi*D^2,chi*D^2]);
    @tensor MMlow[:]:=M7M8[-4,1,2,3]* M5M6[1,2,3,-5,-6,-1,-2,-3]; 
    MMlow=permute(MMlow,(1,2,3,),(4,5,6,))
    #MMlow=reshape(MMlow, [chi*D^2,chi*D^2]);

    @tensor MMup_reflect[:]:=M2_reflect[-1,3,4,1]* M1_reflect[1,2]* M4_reflect'[-2,-5,5,3,7]* M4_reflect[-3,-6,6,4,7]* M3_reflect[2,5,6,-4]; 
    MMup_reflect=permute(MMup_reflect,(1,2,3,),(4,5,6,))
    #MMup_reflect=reshape(MMup_reflect,[chi*D^2,chi*D^2]);
    @tensor MMlow_reflect[:]:=M8_reflect[1,3,4,-1]* M7_reflect[2,1]* M6_reflect'[-2,3,5,-5,7]* M6_reflect[-3,4,6,-6,7]* M5_reflect[-4,5,6,2]; 
    MMlow_reflect=permute(MMlow_reflect,(1,2,3,),(4,5,6,))
    #MMlow_reflect=reshape(MMlow_reflect,[chi*D^2,chi*D^2]);
    _, RMup=leftorth(permute(MMup*MMup_reflect,(4,5,6,),(1,2,3,)));
    _, RMlow=leftorth(permute(MMlow*MMlow_reflect,(4,5,6,),(1,2,3,))); 

    RMlow=permute(RMlow,(2,3,4,),(1,));
    
    M=RMup*RMlow;
    uM,sM,vM = tsvd(M; trunc=truncdim(chi));
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
    #sM_inv_sqrt=sqrt.(convert(Array,sM_inv))
    #display(space(sM_inv))
    #display(sM_inv_sqrt)
    sM_inv_sqrt=TensorMap(pinv.(sqrt.(sM_dense)),codomain(sM_inv)←domain(sM_inv))

    PM_inv=RMlow*vM'*sM_inv_sqrt; 
    PM=sM_inv_sqrt*uM'*RMup; 
    PM=permute(PM,(2,3,4,),(1,));
    
    @tensor M5tem[:]:=M3M4[1,2,3,-2,-3,4,5,6]* PM_inv[1,2,3,-1]* PM[4,5,6,-4];
    @tensor M1tem[:]:=M1M2[1,2,3,-2]* PM_inv[1,2,3,-1];
    @tensor M7tem[:]:=M7M8[-1,1,2,3]* PM[1,2,3,-2];
    
    Cset[mod1(direction,4)]=M1tem/norm(M1tem);
    Tset[mod1(direction-1,4)]=M5tem/norm(M5tem);
    Cset[mod1(direction-1,4)]=M7tem/norm(M7tem);
    
    return Cset, Tset
end;




function ob_CTMRG(CTM,A,opts)
    Caset=CTM["Cset"];
    Cbset=CTM["Cset"];
    Taset=CTM["Tset"];
    Tbset=CTM["Tset"];
    B=A;
    
    if opts["SiteNumber"]==1 #single-site density matrix
        C1=Cbset[1];
        C2=Cbset[2];
        C3=Cbset[3];
        C4=Cbset[4];
        T1=Tbset[1];
        T2=Tbset[2];
        T3=Tbset[3];
        T4=Tbset[4];
        
        @tensor rho[:]:=C1[1,2]* C2[16,15]* C3[12,11]* C4[8,7]* T1[2,4,6,16]* T2[15,13,14,12]* T3[11,9,10,8]* T4[7,3,5,1]* A'[3,9,13,4,-1]* A[5,10,14,6,-2];
        @tensor trace_rho[:]:=rho[1,1]
        rho=rho/trace_rho;
        
    elseif opts["SiteNumber"]==2
        if  opts["direction"]=="x"
            # left: A,  right: B 
            M1=Cbset[1];
            M2=Tbset[1];
            M3=Taset[1];
            M4=Caset[2];
            M5=Tbset[4];
            M6=A;
            M7=B;
            M8=Taset[2];
            M9=Cbset[4];
            M10=Tbset[3];
            M11=Taset[3];
            M12=Caset[3];
            @tensor rho[:]:=M1[1,2]*M2[2,5,7,11]*M3[11,14,16,12]*M4[12,21]*M5[3,4,6,1]*M6'[4,9,13,5,-1]*M6[6,10,15,7,-2]*M7'[13,18,23,14,-3]*M7[15,19,24,16,-4]*M8[21,23,24,22]*M9[8,3]*M10[17,9,10,8]*M11[20,18,19,17]*M12[22,20];
            @tensor trace_rho[:]:=rho[1,2,1,2]
            rho=rho/trace_rho;
        elseif opts["direction"]=="y"
            # up: A,  down: B 
            M1=Cbset[1];
            M2=Tbset[1];
            M3=Cbset[2];
            M4=Tbset[4];
            M5=A;
            M6=Tbset[2];
            M7=Taset[4];
            M8=B;
            M9=Taset[2];
            M10=Caset[4];
            M11=Taset[3];
            M12=Caset[3];
            @tensor rho[:]:=M1[8,3]*M2[3,4,6,1]*M3[1,2]*M4[17,9,10,8]*M5'[9,13,5,4,-1]*M5[10,15,7,6,-2]*M6[2,5,7,11]*M7[20,18,19,17]*M8'[18,23,14,13,-3]*M8[19,24,16,15,-4]*M9[11,14,16,12]*M10[22,20]*M11[21,23,24,22]*M12[12,21];
            @tensor trace_rho[:]:=rho[1,2,1,2]
            rho=rho/trace_rho;
        end
    end
    return rho
end;