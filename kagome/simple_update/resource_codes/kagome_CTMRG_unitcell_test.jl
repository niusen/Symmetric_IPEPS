using LinearAlgebra
using TensorKit

function convert_cell_posit(Lx,Ly,cx,cy,dx,dy,direction)
    
    if direction==1
        posit=CartesianIndex(mod1(cx+dx,Lx),mod1(cy+dy,Ly));
    elseif direction==2
        posit=CartesianIndex(mod1(cx-dy,Lx),mod1(cy+dx,Ly));
    elseif direction==3
        posit=CartesianIndex(mod1(cx-dx,Lx),mod1(cy-dy,Ly));
    elseif direction==4
        posit=CartesianIndex(mod1(cx+dy,Lx),mod1(cy-dx,Ly));
    end
    return posit
end

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


function fuse_CTM_legs_cell(CTM,U_L_cell,U_D_cell,U_R_cell,U_U_cell)
    #fuse CTM legs
    Tset=CTM["Tset"];
    Lx=size(U_L_cell,1);
    Ly=size(U_L_cell,2);
    for cx=1:Lx
        for cy=1:Ly
            #T4
            T4=permute(Tset[4][cx,cy],(1,4,),(2,3,));
            T4=T4*U_R_cell[cx,cy];
            T4=permute(T4,(1,3,2,),());
            Tset[4][cx,cy]=T4
            #display(space(T4))

            #T3
            T3=permute(Tset[3][cx,cy],(1,4,),(2,3,));
            T3=T3*U_U_cell[cx,cy];
            T3=permute(T3,(1,3,2,),());
            Tset[3][cx,cy]=T3
            #display(space(T3))

            #T2
            T2=permute(Tset[2][cx,cy],(2,3,),(1,4,));
            T2=U_L_cell[cx,cy]*T2;
            T2=permute(T2,(2,1,3,),());
            Tset[2][cx,cy]=T2
            #display(space(T2))

            #T1
            T1=permute(Tset[1][cx,cy],(2,3,),(1,4,));
            T1=U_D_cell[cx,cy]*T1;
            T1=permute(T1,(2,1,3,),());
            Tset[1][cx,cy]=T1
            #display(space(T1))

            CTM["Tset"]=Tset;
        end
    end
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

function CTMRG_cell(A_cell,chi,conv_check,tol,init,CTM_ite_nums, CTM_trun_tol,CTM_ite_info=true,CTM_conv_info=false)
    #Ref: PHYSICAL REVIEW B 98, 235148 (2018)
    #initial corner transfer matrix
        #initial corner transfer matrix
    if isempty(init["CTM"])
        #println("11111")
        CTM, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell=init_CTM_cell(chi,A_cell,init["init_type"],CTM_ite_info);
    else
        #println("22222")
        CTM=deepcopy(init["CTM"]);
        AA_fused_cell=init["AA_fused_cell"];
        U_L_cell=init["U_L_cell"];
        U_R_cell=init["U_R_cell"];
        U_U_cell=init["U_U_cell"];
        U_D_cell=init["U_D_cell"];
    end
    Lx=size(A_cell,1);
    Ly=size(A_cell,2);
    ss_old1_cell=Matrix(undef,Lx,Ly);
    ss_old2_cell=Matrix(undef,Lx,Ly);
    ss_old3_cell=Matrix(undef,Lx,Ly);
    ss_old4_cell=Matrix(undef,Lx,Ly);
    ss_new1_cell=Matrix(undef,Lx,Ly);
    ss_new2_cell=Matrix(undef,Lx,Ly);
    ss_new3_cell=Matrix(undef,Lx,Ly);
    ss_new4_cell=Matrix(undef,Lx,Ly);
    er1_cell=Matrix(undef,Lx,Ly);
    er2_cell=Matrix(undef,Lx,Ly);
    er3_cell=Matrix(undef,Lx,Ly);
    er4_cell=Matrix(undef,Lx,Ly);


    Cset=CTM["Cset"];
    Tset=CTM["Tset"];
    conv_check="singular_value"

    for cx=1:Lx
        for cy=1:Ly
            ss_old1_cell[cx,cy]=ones(chi)*2;
            ss_old2_cell[cx,cy]=ones(chi)*2;
            ss_old3_cell[cx,cy]=ones(chi)*2;
            ss_old4_cell[cx,cy]=ones(chi)*2;
        end
    end
    d=2;
    rho_old=Matrix(I,d^3,d^3);

    #Iteration

    print_corner=false;
    C1_spec_cell=Matrix(undef,Lx,Ly);
    C2_spec_cell=Matrix(undef,Lx,Ly);
    C3_spec_cell=Matrix(undef,Lx,Ly);
    C4_spec_cell=Matrix(undef,Lx,Ly);
    if print_corner
        for cx=1:Lx
            for cy=1:Ly
                println("cell position: "*string([cx,cy]))
                println("corner 4:")
                C4_spec=svdvals(convert(Array,Cset[4][cx,cy]));
                C4_spec_cell[cx,cy]=C4_spec/C4_spec[1];
                println(C4_spec);
                println("corner 1:")
                C1_spec=svdvals(convert(Array,Cset[1][cx,cy]));
                C1_spec_cell[cx,cy]=C1_spec/C1_spec[1];
                println(C1_spec);
                println("corner 3:")
                C3_spec=svdvals(convert(Array,Cset[3][cx,cy]));
                C3_spec_cell[cx,cy]=C3_spec/C3_spec[1];
                println(C3_spec);
                println("corner 2:")
                C2_spec=svdvals(convert(Array,Cset[2][cx,cy]));
                C2_spec_cell[cx,cy]=C2_spec/C2_spec[1];
                println(C2_spec);
            end
        end
        println("CTM init finished")
    end
    


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
            if init["init_type"]=="PBC"
                Cset,Tset=CTM_ite_cell(Cset, Tset, AA_fused_cell, chi, direction,CTM_trun_tol,CTM_ite_info);
            elseif init["init_type"]=="single_layer_random"
                Cset,Tset=CTM_ite_cell(Cset, Tset, A_cell, chi, direction,CTM_trun_tol,CTM_ite_info);
            end
        end

        print_corner=false;
        if print_corner
            for cx=1:Lx
                for cy=1:Ly
                    println("cell position: "*string([cx,cy]))
                    println("corner 4:")
                    C4_spec=svdvals(convert(Array,Cset[4][cx,cy]));
                    C4_spec_cell[cx,cy]=C4_spec/C4_spec[1];
                    println(C4_spec);
                    println("corner 1:")
                    C1_spec=svdvals(convert(Array,Cset[1][cx,cy]));
                    C1_spec_cell[cx,cy]=C1_spec/C1_spec[1];
                    println(C1_spec);
                    println("corner 3:")
                    C3_spec=svdvals(convert(Array,Cset[3][cx,cy]));
                    C3_spec_cell[cx,cy]=C3_spec/C3_spec[1];
                    println(C3_spec);
                    println("corner 2:")
                    C2_spec=svdvals(convert(Array,Cset[2][cx,cy]));
                    C2_spec_cell[cx,cy]=C2_spec/C2_spec[1];
                    println(C2_spec);
                end
            end
            println("next iteration:")
        end
        


        if conv_check=="singular_value" #check convergence of singular value
            for cx=1:Lx
                for cy=1:Ly
                    er1,ss_new1=spectrum_conv_check(ss_old1_cell[cx,cy],Cset[1][cx,cy]);
                    er2,ss_new2=spectrum_conv_check(ss_old2_cell[cx,cy],Cset[2][cx,cy]);
                    er3,ss_new3=spectrum_conv_check(ss_old3_cell[cx,cy],Cset[3][cx,cy]);
                    er4,ss_new4=spectrum_conv_check(ss_old4_cell[cx,cy],Cset[4][cx,cy]);

                    er1_cell[cx,cy]=er1;
                    er2_cell[cx,cy]=er2;
                    er3_cell[cx,cy]=er3;
                    er4_cell[cx,cy]=er4;
                    ss_new1_cell[cx,cy]=ss_new1;
                    ss_new2_cell[cx,cy]=ss_new2;
                    ss_new3_cell[cx,cy]=ss_new3;
                    ss_new4_cell[cx,cy]=ss_new4;

                end
            end

            er=maximum([maximum(er1_cell[:]),maximum(er2_cell[:]),maximum(er3_cell[:]),maximum(er4_cell[:])]);
            ite_err=er;
            if CTM_ite_info
                println("CTMRG iteration: "*string(ci)*", CTMRG err: "*string(er));flush(stdout);
            end
            if er<tol
                break;
            end
            ss_old1_cell=ss_new1_cell;
            ss_old2_cell=ss_new2_cell;
            ss_old3_cell=ss_new3_cell;
            ss_old4_cell=ss_new4_cell;
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
        return CTM, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err
    else
        return CTM, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell
    end

end

function CTM_ite_cell(Cset, Tset, AA_fused_cell, chi, direction, trun_tol,CTM_ite_info)
    #println(direction)
    Lx=size(AA_fused_cell,1);
    Ly=size(AA_fused_cell,2);
    PM_cell=Matrix(undef,Lx,Ly);
    PM_inv_cell=Matrix(undef,Lx,Ly);
    M1tem_cell=Matrix(undef,Lx,Ly);
    M5tem_cell=Matrix(undef,Lx,Ly);
    M7tem_cell=Matrix(undef,Lx,Ly);
    
    #coordinate of C1 tensor: (cx,cy)
    AA_rotated_cell=Matrix(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            AA=permute(AA_fused_cell[cx,cy], (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
            AA_rotated_cell[cx,cy]=AA;
        end
    end
    for cx=1:Lx
        for cy=1:Ly
            AA=AA_rotated_cell[convert_cell_posit(Lx,Ly,cx,cy,1,1,direction)];
            C1=Cset[mod1(direction,4)][convert_cell_posit(Lx,Ly,cx,cy,0,0,direction)];
            T2=Tset[mod1(direction,4)][convert_cell_posit(Lx,Ly,cx,cy,1,0,direction)];
            T4=Tset[mod1(direction-1,4)][convert_cell_posit(Lx,Ly,cx,cy,0,1,direction)];
            @tensor MMup[:]:=C1[1,2]*T2[2,3,-3]*T4[-1,4,1]*AA[4,-2,-4,3];

            AA=AA_rotated_cell[convert_cell_posit(Lx,Ly,cx,cy,1,2,direction)];
            T4=Tset[mod1(direction-1,4)][convert_cell_posit(Lx,Ly,cx,cy,0,2,direction)];
            C4=Cset[mod1(direction-1,4)][convert_cell_posit(Lx,Ly,cx,cy,0,3,direction)];
            T3=Tset[mod1(direction-2,4)][convert_cell_posit(Lx,Ly,cx,cy,1,3,direction)];
            # println(norm(C4))
            # println(norm(T3))
            @tensor MMlow[:]:=T4[1,3,-1]*AA[3,4,-4,-2]*C4[2,1]*T3[-3,4,2];


            


            AA=AA_rotated_cell[convert_cell_posit(Lx,Ly,cx,cy,2,1,direction)];
            T1=Tset[mod1(direction,4)][convert_cell_posit(Lx,Ly,cx,cy,2,0,direction)];
            C2=Cset[mod1(direction+1,4)][convert_cell_posit(Lx,Ly,cx,cy,3,0,direction)];
            T2=Tset[mod1(direction+1,4)][convert_cell_posit(Lx,Ly,cx,cy,3,1,direction)];
            @tensor MMup_reflect[:]:=T1[-1,3,1]* C2[1,2]* AA[-2,-4,4,3]* T2[2,4,-3];

            AA=AA_rotated_cell[convert_cell_posit(Lx,Ly,cx,cy,2,2,direction)];
            T2=Tset[mod1(direction+1,4)][convert_cell_posit(Lx,Ly,cx,cy,3,2,direction)];
            T3=Tset[mod1(direction-2,4)][convert_cell_posit(Lx,Ly,cx,cy,2,3,direction)];
            C3=Cset[mod1(direction-2,4)][convert_cell_posit(Lx,Ly,cx,cy,3,3,direction)];
            @tensor MMlow_reflect[:]:=T2[-4,-3,2]*T3[1,-2,-1]*C3[2,1];
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

            # _,sM0,_ = tsvd(M);
            # println("111111")
            # aa=sort(diag(convert(Array,sM0)),rev=true);
            # println(aa/aa[1])

            sM=truncate_multiplet(sM,chi,1e-5,trun_tol);
            
            uM_new,sM_new,vM_new=delet_zero_block(uM,sM,vM);
            # println(norm(M))
            # println(norm(uM_new*sM_new*vM_new-uM*sM*vM))
            # println(norm(uM*sM*vM))
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

            # if (direction==1)&(cx==1)&(cy==1)
            #     aa=sort(diag(sM_dense),rev=true);
            #     println(aa/aa[1])
            #     #display(pinv.(sM_dense))
            # end

            #display(sM_inv)
            #display(convert(Array,sM_inv))
            #sM_inv_sqrt=sqrt.(convert(Array,sM_inv))
            #display(space(sM_inv))
            #display(sM_inv_sqrt)
            sM_inv_sqrt=TensorMap(pinv.(sqrt.(sM_dense)),codomain(sM_inv)←domain(sM_inv))

            PM_inv=RMlow*vM'*sM_inv_sqrt;
            PM=sM_inv_sqrt*uM'*RMup;
            PM=permute(PM,(2,3,),(1,));

            PM_cell[convert_cell_posit(Lx,Ly,cx,cy,0,2,direction)]=PM;
            PM_inv_cell[convert_cell_posit(Lx,Ly,cx,cy,0,1,direction)]=PM_inv;


        end
    end

    for cx=1:Lx
        for cy=1:Ly
            #println([cx,cy])
            AA=AA_rotated_cell[convert_cell_posit(Lx,Ly,cx,cy,1,2,direction)];
            T4=Tset[mod1(direction-1,4)][convert_cell_posit(Lx,Ly,cx,cy,0,2,direction)];
            T1=Tset[mod1(direction,4)][convert_cell_posit(Lx,Ly,cx,cy,1,0,direction)];
            T3=Tset[mod1(direction-2,4)][convert_cell_posit(Lx,Ly,cx,cy,1,3,direction)];
            C1=Cset[mod1(direction,4)][convert_cell_posit(Lx,Ly,cx,cy,0,0,direction)];
            C4=Cset[mod1(direction-1,4)][convert_cell_posit(Lx,Ly,cx,cy,0,3,direction)];
            # println(space(AA))
            # println(space(T4))
            # println(space(PM_inv_cell[convert_cell_posit(Lx,Ly,cx,cy,0,2,direction)]))
            # println(space(PM_cell[convert_cell_posit(Lx,Ly,cx,cy,0,2,direction)]))
            @tensor M5tem[:]:=T4[4,3,1]*AA[3,5,-2,2]* PM_inv_cell[convert_cell_posit(Lx,Ly,cx,cy,0,2,direction)][4,5,-1]* PM_cell[convert_cell_posit(Lx,Ly,cx,cy,0,2,direction)][1,2,-3];
            @tensor M1tem[:]:=C1[1,2]*T1[2,3,-2]*PM_inv_cell[convert_cell_posit(Lx,Ly,cx,cy,0,0,direction)][1,3,-1];
            @tensor M7tem[:]:=C4[1,2]*T3[-1,3,1]* PM_cell[convert_cell_posit(Lx,Ly,cx,cy,0,3,direction)][2,3,-2];

            M5tem=M5tem/norm(M5tem);
            M1tem=M1tem/norm(M1tem);
            M7tem=M7tem/norm(M7tem);

            M5tem_cell[convert_cell_posit(Lx,Ly,cx,cy,1,2,direction)]=M5tem;
            M1tem_cell[convert_cell_posit(Lx,Ly,cx,cy,1,0,direction)]=M1tem;
            M7tem_cell[convert_cell_posit(Lx,Ly,cx,cy,1,3,direction)]=M7tem;

                # println(norm(M5tem))
                # println(norm(M1tem))
                # println(norm(M7tem))

        end
    end
    for cx=1:Lx
        for cy=1:Ly
            Cset[mod1(direction,4)][cx,cy]=M1tem_cell[cx,cy];
            Tset[mod1(direction-1,4)][cx,cy]=M5tem_cell[cx,cy];
            Cset[mod1(direction-1,4)][cx,cy]=M7tem_cell[cx,cy];
        end
    end
    return Cset,Tset
end

function init_CTM_cell(chi,A_cell,type,CTM_ite_info)
    if CTM_ite_info
        display("initialize CTM")
    end
    Lx=size(A_cell,1);
    Ly=size(A_cell,2);
    #numind(A)
    #numin(A)
    #numout(A)
    CTM=[];
    Cset=Vector(undef,4);
    Tset=Vector(undef,4);
    #space(A,1)
    if type=="PBC"
        for direction=1:4
            C_cell=Matrix(undef,Lx,Ly);
            T_cell=Matrix(undef,Lx,Ly);
            for cx=1:Lx
                for cy=1:Ly
                    A=A_cell[cx,cy];
                    inds=(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),5);
                    A_rotate=permute(A,inds);
                    Ap_rotate=A_rotate';

                    @tensor M[:]:=Ap_rotate[1,-1,-3,2,3]*A_rotate[1,-2,-4,2,3];
                    C_cell[cx,cy]=M;
                    @tensor M[:]:=Ap_rotate[-1,-3,-5,1,2]*A_rotate[-2,-4,-6,1,2];
                    T_cell[cx,cy]=M;
                end
            end
            Cset[direction]=C_cell;
            Tset[direction]=T_cell;
        end

        #fuse legs
        ul_set=Vector(undef,4);
        ur_set=Vector(undef,4);
        for direction=1:2
            ul_cell=Matrix(undef,Lx,Ly);
            ur_cell=Matrix(undef,Lx,Ly);
            for cx=1:Lx
                for cy=1:Ly
                    C=Cset[direction][convert_cell_posit(Lx,Ly,cx,cy,-1,0,direction)];
                    T=Tset[direction][convert_cell_posit(Lx,Ly,cx,cy,0,0,direction)];
                    ul_cell[cx,cy]=unitary(fuse(space(C, 3) ⊗ space(C, 4)), space(C, 3) ⊗ space(C, 4));
                    ur_cell[cx,cy]=unitary(fuse(space(T, 5) ⊗ space(T, 6)), space(T, 5) ⊗ space(T, 6));
                end
            end
            ul_set[direction]=ul_cell;
            ur_set[direction]=ur_cell;
        end
        for direction=3:4
            ul_cell=Matrix(undef,Lx,Ly);
            ur_cell=Matrix(undef,Lx,Ly);
            for cx=1:Lx
                for cy=1:Ly
                    C=Cset[direction][convert_cell_posit(Lx,Ly,cx,cy,-1,0,direction)];
                    T=Tset[direction][convert_cell_posit(Lx,Ly,cx,cy,0,0,direction)];

                    ul_cell[cx,cy]=unitary(fuse(space(C, 3) ⊗ space(C, 4))', space(C, 3) ⊗ space(C, 4));
                    ur_cell[cx,cy]=unitary(fuse(space(T, 5) ⊗ space(T, 6))', space(T, 5) ⊗ space(T, 6));
                end
            end
            ul_set[direction]=ul_cell;
            ur_set[direction]=ur_cell;
        end
        for direction=1:4
            for cx=1:Lx
                for cy=1:Ly
                    C=Cset[direction][convert_cell_posit(Lx,Ly,cx,cy,-1,0,direction)];
                    T=Tset[direction][convert_cell_posit(Lx,Ly,cx,cy,0,0,direction)];

                    ul=ur_set[mod1(direction-1,4)][convert_cell_posit(Lx,Ly,cx,cy,-1,1,direction)];
                    ur=ul_set[direction][cx,cy];
                    ulp=permute(ul',(3,),(1,2,));
                    urp=permute(ur',(3,),(1,2,));
                    #@tensor Cnew[(-1);(-2)]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4]
                    @tensor Cnew[:]:=ulp[-1,1,2]*C[1,2,3,4]*ur[-2,3,4];#put all indices in tone side so that its adjoint has the same index order
                    Cset[direction][convert_cell_posit(Lx,Ly,cx,cy,-1,0,direction)]=Cnew;

                    ul=ul_set[direction][cx,cy];
                    ur=ur_set[direction][cx,cy];
                    ulp=permute(ul',(3,),(1,2,));
                    urp=permute(ur',(3,),(1,2,));
                    #@tensor Tnew[(-1);(-2,-3,-4)]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4]
                    @tensor Tnew[:]:=ulp[-1,1,2]*T[1,2,-2,-3,3,4]*ur[-4,3,4];#put all indices in tone side so that its adjoint has the same index order
                    Tset[direction][convert_cell_posit(Lx,Ly,cx,cy,0,0,direction)]=Tnew;

                end
            end
        end

        AA_fused_cell=Matrix(undef,Lx,Ly);
        U_L_cell=Matrix(undef,Lx,Ly);
        U_D_cell=Matrix(undef,Lx,Ly);
        U_R_cell=Matrix(undef,Lx,Ly);
        U_U_cell=Matrix(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                CTM=Dict([("Cset", Cset), ("Tset", Tset)]);
                AA_fused, U_L,U_D,U_R,U_U=build_double_layer(A_cell[cx,cy],[]);
                AA_fused_cell[cx,cy]=AA_fused;
                U_L_cell[cx,cy]=U_L;
                U_R_cell[cx,cy]=U_R;
                U_U_cell[cx,cy]=U_U;
                U_D_cell[cx,cy]=U_D;
            end
        end
        CTM=fuse_CTM_legs_cell(CTM,U_L_cell,U_D_cell,U_R_cell,U_U_cell);

        return CTM, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell
    elseif type=="random"
    elseif type=="single_layer_random"
        #For single layer CTMRG algorithm.
        V_init=Rep[SU₂](0=>1, 1/2=>1, 1=>1, 3/2=>1);
        V_init=Rep[SU₂](0=>5, 1/2=>5,1=>2);
        V_init=Rep[SU₂](0=>5,1/2=>5);
        Lx=size(A_cell,1);
        Ly=size(A_cell,2);

        CTM=[];
        Cset=Vector(undef,4);
        Tset=Vector(undef,4);

        for direction=1:4
            C_cell=Matrix(undef,Lx,Ly);
            T_cell=Matrix(undef,Lx,Ly);
            for cx=1:Lx
                for cy=1:Ly
                    #T tensor
                    A=A_cell[convert_cell_posit(Lx,Ly,cx,cy,0,1,direction)];
                    inds=(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4));
                    A_rotate=permute(A,(mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),));
                    T=TensorMap(randn, V_init,space(A_rotate,4)⊗ V_init);
                    T_cell[cx,cy]=permute(T,(1,2,3,))
                    C=TensorMap(randn, V_init, V_init);
                    C_cell[cx,cy]=permute(C,(1,2,));

                end
            end
            Tset[direction]=T_cell;
            Cset[direction]=C_cell;
            CTM=Dict([("Cset", Cset), ("Tset", Tset)]);
        end

        return CTM, nothing,nothing,nothing,nothing,nothing

    elseif type=="single_layer_PBC"
        #For single layer CTMRG algorithm.
        Lx=size(A_cell,1);
        Ly=size(A_cell,2);
        @assert Lx==2
        @assert Ly==2

        CTM=[];
        Cset=Vector(undef,4);
        Tset=Vector(undef,4);


        cx=1;cy=1;

        ###########
        direction=1;
        C_cell=Matrix(undef,Lx,Ly);
        T_cell=Matrix(undef,Lx,Ly);
        #T tensor
        A=A_cell[cx,cy];
        B=A_cell[mod1(cx+1,Lx),cy];
        U1=unitary(fuse(space(A,3)*space(A,4)),space(A,3)*space(A,4))
        @tensor T[:]:=A[-1,-2,1,2]*U1[-3,1,2];
        T_cell[cx,cy]=T;
        @tensor T[:]:=B[1,-2,-4,2]*U1'[1,2,-1];
        T_cell[mod1(cx+1,Lx),cy]=T;
        #C tensor
        U2=unitary(fuse(space(A,1)*space(A,2)),space(A,1)*space(A,2))
        @tensor C[:]:=A[1,2,3,4]*U1[-2,3,4]*U2[-1,1,2];
        C_cell[cx,cy]=C;    
        Tset[direction]=T_cell;
        Cset[direction]=C_cell;

        ###########
        direction=2;
        C_cell=Matrix(undef,Lx,Ly);
        T_cell=Matrix(undef,Lx,Ly);
        #T tensor
        A=A_cell[mod1(cx+1,Lx),cy];
        B=A_cell[mod1(cx+1,Lx),mod1(cy-1,Ly)];
        U3=unitary(fuse(space(A,3)*space(A,2)) ,space(A,3)*space(A,2))
        @tensor T[:]:=A[-2,2,1,-1]*U3[-3,1,2];
        T_cell[mod1(cx+1,Lx),cy]=T;
        @tensor T[:]:=B[-2,-3,1,2]*U3'[1,2,-1];
        T_cell[mod1(cx+1,Lx),mod1(cy-1,Ly)]=T;
        #C tensor
        @tensor C[:]:=A[1,4,3,2]*U1'[1,2,-1]*U3[-2,3,4];
        C_cell[mod1(cx+1,Lx),cy]=C;    
        Tset[direction]=T_cell;
        Cset[direction]=C_cell;

        ###########
        direction=3;
        C_cell=Matrix(undef,Lx,Ly);
        T_cell=Matrix(undef,Lx,Ly);
        #T tensor
        A=A_cell[mod1(cx+1,Lx),mod1(cy-1,Ly)];
        B=A_cell[cx,mod1(cy-1,Ly)];
        U4=unitary(fuse(space(A,1)*space(A,2)) ,space(A,1)*space(A,2))
        @tensor T[:]:=A[1,2,-1,-2]*U4[-3,1,2];
        T_cell[mod1(cx+1,Lx),mod1(cy-1,Ly)]=T;
        @tensor T[:]:=B[-3,2,1,-2]*U4'[1,2,-1];
        T_cell[cx,mod1(cy-1,Ly)]=T;
        #C tensor
        @tensor C[:]:=A[1,2,3,4]*U4[-1,1,2]*U3'[3,4,-2];
        C_cell[mod1(cx+1,Lx),mod1(cy-1,Ly)]=C;    
        Tset[direction]=T_cell;
        Cset[direction]=C_cell;

        ###########
        direction=4;
        C_cell=Matrix(undef,Lx,Ly);
        T_cell=Matrix(undef,Lx,Ly);
        #T tensor
        A=A_cell[cx,mod1(cy-1,Ly)];
        B=A_cell[cx,cy];
        @tensor T[:]:=A[1,-1,-2,2]*U2'[1,2,-3];
        T_cell[cx,mod1(cy-1,Ly)]=T;
        @tensor T[:]:=B[1,2,-2,-3]*U2[-1,1,2];
        T_cell[cx,cy]=T;
        #C tensor
        @tensor C[:]:=A[3,2,1,4]*U4'[1,2,-1]*U2'[3,4,-2];
        C_cell[cx,mod1(cy-1,Ly)]=C;    
        Tset[direction]=T_cell;
        Cset[direction]=C_cell;



        CTM=Dict([("Cset", Cset), ("Tset", Tset)]);
        

        return CTM, nothing,nothing,nothing,nothing,nothing
    end


    


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
        if (size(diag(Σ_dict[:data][string(c)]),1)>0) & (sum(diag(Σ_dict[:data][string(c)]))>0)
            inds=findall(x->(x>0), diag(Σ_dict[:data][string(c)]));
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