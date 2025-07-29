using LinearAlgebra:diag,I 
using TensorKit
using Statistics

function build_doublelayer_swap_iPESS(B_set,T_set, pos)
    c1=pos[1];
    c2=pos[2];
    B=B_set[c1,c2];
    B_double, U_L,U_M,U_U = build_double_layer_swap_Tm(B',B, false);#L M U
    T=T_set[c1,c2];
    T_double, U_D,U_R,U_M = build_double_layer_swap_Bm(T',T, true);#D R M
    @tensor AA[:]:=B_double[-1,1,-4]*T_double[-2,-3,1];#(L M U),(D R M) =>(L,D,R,U)
    return AA, T_double, B_double, U_L,U_D,U_R,U_U
end


function convert_cell_posit(cx,cy,dx,dy,direction)
    global Lx,Ly
    if direction==1
        posit=CartesianIndex(mod1(cx+dx,Lx),mod1(cy+dy,Ly));
    elseif direction==2
        posit=CartesianIndex(mod1(cy-dy,Lx),mod1(cx+dx,Ly));
    elseif direction==3
        posit=CartesianIndex(mod1(cx-dx,Lx),mod1(cy-dy,Ly));
    elseif direction==4
        posit=CartesianIndex(mod1(cy+dy,Lx),mod1(cx-dx,Ly));
    end
    return posit
end

function rotate_AA_direction(AA_fused,direction)
    AA_rotated=permute(AA_fused, (mod1(2-direction,4),mod1(3-direction,4),mod1(4-direction,4),mod1(1-direction,4),),());
    return AA_rotated
end



function Fermionic_CTMRG_cell_iPESS(B_set::Matrix,T_set::Matrix,A_cell::Tuple,chi,init,CTM0, ctm_setting) 
    global Lx,Ly
    global algrithm_CTMRG_settings
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
        double_B_cell=initial_tuple_cell(Lx,Ly);
        double_T_cell=initial_tuple_cell(Lx,Ly);
        U_L_cell=initial_tuple_cell(Lx,Ly);
        U_D_cell=initial_tuple_cell(Lx,Ly);
        U_R_cell=initial_tuple_cell(Lx,Ly);
        U_U_cell=initial_tuple_cell(Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                AA_, T_double, B_double, U_L_,U_D_,U_R_,U_U_=build_doublelayer_swap_iPESS(B_set,T_set, [cx,cy]);

                double_B_cell=fill_tuple(double_B_cell, B_double, cx,cy);
                double_T_cell=fill_tuple(double_T_cell, T_double, cx,cy);
                U_L_cell=fill_tuple(U_L_cell, U_L_, cx,cy);
                U_D_cell=fill_tuple(U_D_cell, U_D_, cx,cy);
                U_R_cell=fill_tuple(U_R_cell, U_R_, cx,cy);
                U_U_cell=fill_tuple(U_U_cell, U_U_, cx,cy);
            end
        end
        AA_memory=@ignore_derivatives (Base.summarysize(double_B_cell)+Base.summarysize(double_T_cell))/1024/1024;
        @ignore_derivatives if CTM_ite_info
            println("Memory cost of double layer iPESS: "*string(AA_memory)*" Mb.");flush(stdout);
        end
    else
        # AA_fused_cell=auxi_tensors.AA_fused_cell;
        # U_L_cell=auxi_tensors.U_L_cell;
        # U_D_cell=auxi_tensors.U_D_cell;
        # U_R_cell=auxi_tensors.U_R_cell;
        # U_U_cell=auxi_tensors.U_U_cell;
    end

    if init.reconstruct_CTM
        CTM_cell= init_CTM_cell(chi,B_set,T_set,  init.init_type,CTM_ite_info);
    else
        CTM_cell=deepcopy(CTM0);
    end
    
    ss_old1_cell= Matrix(undef,Lx,Ly);
    ss_old2_cell= Matrix(undef,Lx,Ly);
    ss_old3_cell= Matrix(undef,Lx,Ly);
    ss_old4_cell= Matrix(undef,Lx,Ly);
    ss_new1_cell= Matrix(undef,Lx,Ly);
    ss_new2_cell= Matrix(undef,Lx,Ly);
    ss_new3_cell= Matrix(undef,Lx,Ly);
    ss_new4_cell= Matrix(undef,Lx,Ly);
    er1_cell= Matrix(undef,Lx,Ly);
    er2_cell= Matrix(undef,Lx,Ly);
    er3_cell= Matrix(undef,Lx,Ly);
    er4_cell= ones(Lx,Ly);


    Cset_cell=CTM_cell.Cset;
    Tset_cell=CTM_cell.Tset;
    conv_check="singular_value"

    @ignore_derivatives for cx=1:Lx
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
    C1_spec_cell=@ignore_derivatives Matrix(undef,Lx,Ly);
    C2_spec_cell=@ignore_derivatives Matrix(undef,Lx,Ly);
    C3_spec_cell=@ignore_derivatives Matrix(undef,Lx,Ly);
    C4_spec_cell=@ignore_derivatives Matrix(undef,Lx,Ly);
    @ignore_derivatives if print_corner
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
    


    @ignore_derivatives if CTM_ite_info
        println("start CTM iterations:")
    end
    ite_num=0;
    ite_err=1;
    err_set=1;
    
    if algrithm_CTMRG_settings.CTM_cell_ite_method== "together_update"
        CTM_ite_cell=CTM_ite_cell_together_update;
    elseif algrithm_CTMRG_settings.CTM_cell_ite_method== "continuous_update"
        CTM_ite_cell=CTM_ite_cell_continuous_update;
    end

    for ci=1:CTM_ite_nums
        ite_num=ci;
        #direction_order=[1,2,3,4];
        #direction_order=[4,1,2,3];
        direction_order=[3,4,1,2];
        @time begin
            for direction in direction_order
                Cset_cell,Tset_cell=CTM_ite_cell(Cset_cell, Tset_cell, double_B_cell,double_T_cell, chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer);
            end
        end

        print_corner=false;
        @ignore_derivatives if print_corner
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
                    er1,ss_new1=@ignore_derivatives spectrum_conv_check(ss_old1_cell[cx,cy],Cset_cell[cx][cy].C1);
                    er2,ss_new2=@ignore_derivatives spectrum_conv_check(ss_old2_cell[cx,cy],Cset_cell[cx][cy].C2);
                    er3,ss_new3=@ignore_derivatives spectrum_conv_check(ss_old3_cell[cx,cy],Cset_cell[cx][cy].C3);
                    er4,ss_new4=@ignore_derivatives spectrum_conv_check(ss_old4_cell[cx,cy],Cset_cell[cx][cy].C4);

                    # println([cx,cy])
                    # println(ss_new1)
                    # println(ss_new2)
                    # println(ss_new3)
                    # println(ss_new4)
                    

                    @ignore_derivatives er1_cell[cx,cy]=er1;
                    @ignore_derivatives er2_cell[cx,cy]=er2;
                    @ignore_derivatives er3_cell[cx,cy]=er3;
                    @ignore_derivatives er4_cell[cx,cy]=er4;
                    @ignore_derivatives ss_new1_cell[cx,cy]=ss_new1;
                    @ignore_derivatives ss_new2_cell[cx,cy]=ss_new2;
                    @ignore_derivatives ss_new3_cell[cx,cy]=ss_new3;
                    @ignore_derivatives ss_new4_cell[cx,cy]=ss_new4;

                end
            end

            er=@ignore_derivatives maximum([maximum(er1_cell[:]),maximum(er2_cell[:]),maximum(er3_cell[:]),maximum(er4_cell[:])]);
            err_set=vcat(err_set,er);

            ite_err=er;
            if CTM_ite_info
                println("CTMRG iteration: "*string(ci)*", CTMRG err: "*string(er));flush(stdout);
            end
            if er<ctm_setting.CTM_conv_tol
                break;
            end

            if ci>30
                err_recent=err_set[end-10:end];
                Std=std(err_recent)/mean(err_recent);
                if (Std<0.001)&(er>1e-4)
                    break;
                end

            end

            ss_old1_cell=ss_new1_cell;
            ss_old2_cell=ss_new2_cell;
            ss_old3_cell=ss_new3_cell;
            ss_old4_cell=ss_new4_cell;
        elseif conv_check=="density_matrix" #check reduced density matrix
        end
    end

    CTM_cell=(Cset=Cset_cell,Tset=Tset_cell);
    if CTM_conv_info
        return CTM_cell, double_B_cell,double_T_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err
    else
        return CTM_cell, double_B_cell,double_T_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell
    end

end

function get_AA_direction(double_B_cell,double_T_cell,direction,pos)
    B_double=double_B_cell[pos[1]][pos[2]];
    T_double=double_T_cell[pos[1]][pos[2]];
    @tensor AA[:]:=B_double[-1,1,-4]*T_double[-2,-3,1];
    return rotate_AA_direction(AA,direction)
end


function build_corner_MMup(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly)
    global Lx,Ly
    Pos=convert_cell_posit(coord_[1],coord_[2],1,1,direction_);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1],coord_[2],0,0,direction_);
    C1=get_Cset(Cset_cell_[Pos[1]][Pos[2]], mod1(direction_,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],1,0,direction_);
    T1=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],0,1,direction_);
    T4=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_-1,4));
    @tensor MMup_[:]:=C1[1,2]*T1[2,3,-3]*T4[-1,4,1]*AA[4,-2,-4,3];
    return MMup_
end

function build_corner_MMlow(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly)
    global Lx,Ly
    Pos=convert_cell_posit(coord_[1],coord_[2],1,2,direction_);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1],coord_[2],0,2,direction_);
    T4=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_-1,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],0,3,direction_);
    C4=get_Cset(Cset_cell_[Pos[1]][Pos[2]], mod1(direction_-1,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],1,3,direction_);
    T3=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_-2,4));
    @tensor MMlow_[:]:=T4[1,3,-1]*AA[3,4,-4,-2]*C4[2,1]*T3[-3,4,2];
    return MMlow_
end


function build_corner_MMup_reflect(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly)
    global Lx,Ly
    Pos=convert_cell_posit(coord_[1],coord_[2],2,1,direction_);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1],coord_[2],2,0,direction_);
    T1=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],3,0,direction_);
    C2=get_Cset(Cset_cell_[Pos[1]][Pos[2]], mod1(direction_+1,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],3,1,direction_);
    T2=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_+1,4));
    @tensor MMup_reflect_[:]:=T1[-1,3,1]* C2[1,2]* AA[-2,-4,4,3]* T2[2,4,-3];
    return MMup_reflect_
end

function build_corner_MMlow_reflect(coord_,direction_,double_B_cell_,double_T_cell_,Cset_cell_,Tset_cell_,Lx,Ly)
    global Lx,Ly
    Pos=convert_cell_posit(coord_[1],coord_[2],2,2,direction_);
    AA=get_AA_direction(double_B_cell_,double_T_cell_,direction_,Pos);
    Pos=convert_cell_posit(coord_[1],coord_[2],3,2,direction_);
    T2=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_+1,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],2,3,direction_);
    T3=get_Tset(Tset_cell_[Pos[1]][Pos[2]], mod1(direction_-2,4));
    Pos=convert_cell_posit(coord_[1],coord_[2],3,3,direction_);
    C3=get_Cset(Cset_cell_[Pos[1]][Pos[2]], mod1(direction_-2,4));
    @tensor MMlow_reflect_[:]:=T2[-4,-3,2]*T3[1,-2,-1]*C3[2,1];
    @tensor MMlow_reflect_[:]:=MMlow_reflect_[-1,1,2,-3]*AA[-2,1,2,-4];
    return MMlow_reflect_
end


function append_data(d1,d2)
    append!(d1,d2)
    d1
end

function CTM_ite_cell_continuous_update(Cset_cell, Tset_cell, double_B_cell,double_T_cell, chi, direction, trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer)
    global Lx,Ly
    #println(direction)    
    #
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """

    # if (direction==1)|(direction==3)
    #     cx_max=Lx;
    #     cy_max=Ly;
    # elseif (direction==2)|(direction==4)
    #     cx_max=Ly;
    #     cy_max=Lx;
    # end
    cx_cy_matrix=[Lx Ly;Ly Lx;Lx Ly;Ly Lx];
    cx_max=cx_cy_matrix[direction,1];
    cy_max=cx_cy_matrix[direction,2];

    for cx=1:cx_max

        PM_cell=initial_tuple_cell(Lx,Ly);
        PM_inv_cell=initial_tuple_cell(Lx,Ly);
        M1tem_cell=initial_tuple_cell(Lx,Ly);
        M5tem_cell=initial_tuple_cell(Lx,Ly);
        M7tem_cell=initial_tuple_cell(Lx,Ly);

        for cy=1:cy_max
            coord=[cx,cy];

            ##########################
            parall_data=[];
            parall_data=@sync @distributed  (append_data)  for ccc=1:4
                BLAS.set_num_threads(5);
                if ccc==1
                    MM=build_corner_MMup(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);
                elseif ccc==2
                    MM=build_corner_MMlow(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);
                elseif ccc==3
                    MM=build_corner_MMup_reflect(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);
                elseif ccc==4
                    MM=build_corner_MMlow_reflect(coord,direction,double_B_cell,double_T_cell,Cset_cell,Tset_cell,Lx,Ly);
                end
                [(myid(), ccc, MM)]
            end

            for ccc=1:4
                if parall_data[ccc][2]==1
                    MMup=parall_data[ccc][3];
                elseif parall_data[ccc][2]==2
                    MMlow=parall_data[ccc][3];
                elseif parall_data[ccc][2]==3
                    MMup_reflect=parall_data[ccc][3];
                elseif parall_data[ccc][2]==4
                    MMlow_reflect=parall_data[ccc][3];
                end
            end
            parall_data=[];
            ############################



            MMup=permute(MMup,(1,2,),(3,4,))

            # _,ss,_=tsvd(MMup)
            # display(convert(Array,ss))

            MMlow=permute(MMlow,(1,2,),(3,4,))
            MMup_reflect=permute(MMup_reflect,(1,2,),(3,4,))
            MMlow_reflect=permute(MMlow_reflect,(1,2,),(3,4,))

            

            RMup=permute(MMup*MMup_reflect,(3,4,),(1,2,));
            RMlow=MMlow*MMlow_reflect;

            RMlow_norm=norm(RMlow);
            RMlow= RMlow/RMlow_norm;
            RMup_norm=norm(RMup);
            RMup= RMup/RMup_norm;

            M=RMup*RMlow;

            if isa(space(M,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})#Z2 symmetry
                chi_extra=3;
            elseif isa(space(M,1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}) #U1 symmetry
                chi_extra=4;
            elseif isa(space(M,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}) #SU(2) symmetry
                chi_extra=20;
            elseif isa(space(M,1), GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}) #U1 x SU(2)
                chi_extra=20;
            end

            #####################################


            # uM,sM,vM = tsvd(M; trunc=truncdim(chi+chi_extra));#for new version Pkgs, tsvd backward is much better

 
            N_blocks=length(M.data.values);
            parall_svd_data=[];
            parall_svd_data=@sync @distributed (append_data) for ccc =1:N_blocks
                BLAS.set_num_threads(6);
                uu,ss,vv = svd(M.data.values[ccc]);
                vv=vv';
                [(myid(), ccc, uu,ss,vv)]
            end

            uu_set=Vector{Array}(undef,N_blocks);
            ss_set=Vector{Array}(undef,N_blocks);
            vv_set=Vector{Array}(undef,N_blocks);
            for ccc=1:N_blocks
                ind=parall_svd_data[ccc][2];
                uu_set[ind]=parall_svd_data[ccc][3];
                ss_set[ind]=parall_svd_data[ccc][4];
                vv_set[ind]=parall_svd_data[ccc][5];
            end
            parall_svd_data=[];
            uM,sM,vM = truncate_block_svd(uu_set,ss_set,vv_set,M,chi+chi_extra);

            #println(norm(uM*sM*vM-M)/norm(M));
            #############################################
        


            sM_norm=norm(sM);
            sM=sM/sM_norm;
            
            sM_inv_sqrt=sdiag_inv_sqrt(sM);

            PM_inv=RMlow*vM'*sM_inv_sqrt;
            PM=sM_inv_sqrt*uM'*RMup;
            PM=permute(PM,(2,3,),(1,));

            Pos=convert_cell_posit(coord[1],coord[2],0,2,direction);
            #PM_cell[Pos[1]][Pos[2]]=PM;
            PM_cell=fill_tuple(PM_cell,PM, Pos[1],Pos[2])
            Pos=convert_cell_posit(coord[1],coord[2],0,1,direction);
            #PM_inv_cell[Pos[1]][Pos[2]]=PM_inv;
            PM_inv_cell=fill_tuple(PM_inv_cell,PM_inv, Pos[1],Pos[2])

        end

        for cy=1:cy_max
            coord=[cx,cy];

            Pos=convert_cell_posit(coord[1],coord[2],1,2,direction);
            AA=get_AA_direction(double_B_cell,double_T_cell,direction,Pos);;
            Pos=convert_cell_posit(coord[1],coord[2],0,2,direction);
            T4=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));
            Pos=convert_cell_posit(coord[1],coord[2],1,0,direction);
            T1=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],1,3,direction);
            T3=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-2,4));
            Pos=convert_cell_posit(coord[1],coord[2],0,0,direction);
            C1=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],0,3,direction);
            C4=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));


            Posa=convert_cell_posit(coord[1],coord[2],0,2,direction);
            Posb=convert_cell_posit(coord[1],coord[2],0,2,direction);
            @tensor M5tem[:]:=T4[4,3,1]*AA[3,5,-2,2]*PM_inv_cell[Posa[1]][Posa[2]][4,5,-1]*PM_cell[Posb[1]][Posb[2]][1,2,-3];
            Pos=convert_cell_posit(coord[1],coord[2],0,0,direction);
            @tensor M1tem[:]:=C1[1,2]*T1[2,3,-2]*PM_inv_cell[Pos[1]][Pos[2]][1,3,-1];
            Pos=convert_cell_posit(coord[1],coord[2],0,3,direction);
            @tensor M7tem[:]:=C4[1,2]*T3[-1,3,1]*PM_cell[Pos[1]][Pos[2]][2,3,-2];

            M5tem=M5tem/norm(M5tem);
            M1tem=M1tem/norm(M1tem);
            M7tem=M7tem/norm(M7tem);

            Pos=convert_cell_posit(coord[1],coord[2],1,2,direction);
            #M5tem_cell[Pos[1]][Pos[2]]=M5tem;
            M5tem_cell=fill_tuple(M5tem_cell, M5tem, Pos[1],Pos[2]);
            Pos=convert_cell_posit(coord[1],coord[2],1,0,direction);
            #M1tem_cell[Pos[1]][Pos[2]]=M1tem;
            M1tem_cell=fill_tuple(M1tem_cell, M1tem, Pos[1],Pos[2]);
            Pos=convert_cell_posit(coord[1],coord[2],1,3,direction);
            #M7tem_cell[Pos[1]][Pos[2]]=M7tem;
            M7tem_cell=fill_tuple(M7tem_cell, M7tem, Pos[1],Pos[2]);
            
        end


        for cy=1:cy_max
            coord=[cx,cy];


            Pos=convert_cell_posit(coord[1],coord[2],1,0,direction);
            #Cset_cell[Pos[1]][Pos[2]][mod1(direction,4)]=M1tem_cell[Pos[1]][Pos[2]];
            Cset_old=Cset_cell[Pos[1]][Pos[2]];
            Cset_new=set_Cset(Cset_old, M1tem_cell[Pos[1]][Pos[2]], mod1(direction,4))
            Cset_cell=fill_tuple(Cset_cell, Cset_new, Pos[1],Pos[2]);

            Pos=convert_cell_posit(coord[1],coord[2],1,2,direction);
            #Tset_cell[Pos[1]][Pos[2]][mod1(direction-1,4)]=M5tem_cell[Pos[1]][Pos[2]];
            Tset_old=Tset_cell[Pos[1]][Pos[2]];
            Tset_new=set_Tset(Tset_old, M5tem_cell[Pos[1]][Pos[2]], mod1(direction-1,4))
            Tset_cell=fill_tuple(Tset_cell, Tset_new, Pos[1],Pos[2]);

            Pos=convert_cell_posit(coord[1],coord[2],1,3,direction);
            #Cset_cell[Pos[1]][Pos[2]][mod1(direction-1,4)]=M7tem_cell[Pos[1]][Pos[2]];
            Cset_old=Cset_cell[Pos[1]][Pos[2]];
            Cset_new=set_Cset(Cset_old, M7tem_cell[Pos[1]][Pos[2]], mod1(direction-1,4))
            Cset_cell=fill_tuple(Cset_cell, Cset_new, Pos[1],Pos[2]);
            
        end
    end
    return Cset_cell,Tset_cell
end



function init_CTM_cell(chi,B_set,T_set, type,CTM_ite_info)
    @ignore_derivatives  if CTM_ite_info
        display("initialize CTM from iPESS")
    end
    global Lx,Ly




    Cset_cell=initial_tuple_cell(Lx,Ly);
    Tset_cell=initial_tuple_cell(Lx,Ly);

    if type=="PBC"
        for cx=1:Lx
            for cy=1:Ly
                AA, _, _, U_L,U_D,U_R,U_U=build_doublelayer_swap_iPESS(B_set,T_set, [cx,cy]);
                small_tensor=B_set[1,1];#the value not important

                CTM_=init_CTM_swap(chi,small_tensor,AA, U_L,U_D,U_R,U_U, type,false);
                Cset_cell=fill_tuple(Cset_cell,CTM_.Cset,cx,cy);
                Tset_cell=fill_tuple(Tset_cell,CTM_.Tset,cx,cy);
            end
        end
        CTM_cell=(Cset=Cset_cell,Tset=Tset_cell);
        return CTM_cell
    elseif type=="random"
    end
end

