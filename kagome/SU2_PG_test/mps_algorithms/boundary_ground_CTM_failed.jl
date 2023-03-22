


function CTMRG_boundary_ground(O1,O2,W,conv_check,tol,CTM_init)
    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)

    @tensor OO[:]:=O1[-2,-3,-5,1]*O2[-1,1,-4,-6];
    U_fuse_chichi=unitary(fuse(space(OO,1)⊗ space(OO,2)),space(OO,1)⊗ space(OO,2));
    @tensor OO[:]:=U_fuse_chichi[-1,1,2]*OO[1,2,-2,3,4,-4]*U_fuse_chichi'[3,4,-3];

    #initial corner transfer matrix
        #initial corner transfer matrix
    if isempty(CTM_init)
        CTM=init_CTM_boundary_ground(W,OO);
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
        direction_order=[1,2,3,4];
        for direction in direction_order
            Cset,Tset=CTM_ite_boundary_ground(Cset, Tset, OO, chi, direction);
        end
        AL=left_right_normalize(permute(Tset[3],(3,1,2,),()));
        #dominant eigenvalue of transfer matrix
        E=left_eigenvalue(impo_imps(OO,AL),AL,1)[1]
        display("E="*string(E))

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
    return CTM, OO

end

function CTM_ite_boundary_ground(Cset, Tset, OO, W, direction)

    OO_rotated=permute(OO, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
                
    @tensor MMup[:]:=Cset[mod1(direction,4)][1,2]*Tset[mod1(direction,4)][2,3,-3]*Tset[mod1(direction-1,4)][-1,4,1]*OO_rotated[4,-2,-4,3]; 
    @tensor MMlow[:]:=Tset[mod1(direction-1,4)][1,3,-1]*OO_rotated[3,4,-4,-2]*Cset[mod1(direction-1,4)][2,1]*Tset[mod1(direction-2,4)][-3,4,2]; 

    @tensor MMup_reflect[:]:=Tset[mod1(direction,4)][-1,3,1]* Cset[mod1(direction+1,4)][1,2]* OO_rotated[-2,-4,4,3]* Tset[mod1(direction+1,4)][2,4,-3];
    #@tensor MMlow_reflect[:]:=OO_rotated[-2,4,3,-4]*Tset[mod1(direction+1,4)][-3,3,1]*Tset[mod1(direction-2,4)][2,4,-1]*Cset[mod1(direction-2,4)][1,2]; #this is slow compared to other coners, I don't know why            
    @tensor MMlow_reflect[:]:=Tset[mod1(direction+1,4)][-4,-3,2]*Tset[mod1(direction-2,4)][1,-2,-1]*Cset[mod1(direction-2,4)][2,1]; 
    @tensor MMlow_reflect[:]:=MMlow_reflect[-1,1,2,-3]*OO_rotated[-2,1,2,-4]; 

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

    @tensor M5tem[:]:=Tset[mod1(direction-1,4)][4,3,1]*OO_rotated[3,5,-2,2]* PM_inv[4,5,-1]* PM[1,2,-3];
    @tensor M1tem[:]:=Cset[mod1(direction,4)][1,2]*Tset[mod1(direction,4)][2,3,-2]*PM_inv[1,3,-1];
    @tensor M7tem[:]:=Cset[mod1(direction-1,4)][1,2]*Tset[mod1(direction-2,4)][-1,3,1]* PM[2,3,-2];

    Cset[mod1(direction,4)]=M1tem/norm(M1tem);
    Tset[mod1(direction-1,4)]=M5tem/norm(M5tem);
    Cset[mod1(direction-1,4)]=M7tem/norm(M7tem);


    return Cset,Tset
end

function init_CTM_boundary_ground(W,OO)

    display("initialize CTM")
    #numind(A)
    #numin(A)
    #numout(A)
    CTM=[];
    Cset=Vector(undef,4);
    Tset=Vector(undef,4);
    #space(A,1)
    mps_virtual=SU₂Space(0=>1,1/2=>5);
    Tset[3]=permute(TensorMap(randn, mps_virtual*mps_virtual', space(OO,2)),(1,3,2,),());
    Cset[4]=TensorMap(randn, mps_virtual,mps_virtual);
    Tset[4]=permute(TensorMap(randn, mps_virtual*mps_virtual', space(OO,1)),(1,3,2,),());
    Cset[1]=TensorMap(randn, mps_virtual,mps_virtual);
    Tset[1]=permute(TensorMap(randn, mps_virtual*mps_virtual', space(OO,4)),(1,3,2,),());
    Cset[2]=TensorMap(randn, mps_virtual,mps_virtual);
    Tset[2]=permute(TensorMap(randn, mps_virtual*mps_virtual', space(OO,3)),(1,3,2,),());
    Cset[3]=TensorMap(randn, mps_virtual,mps_virtual);

    CTM=Dict([("Cset", Cset), ("Tset", Tset)]);

    return CTM

    # #save initial CTM to compare with other codes
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