using LinearAlgebra:diag,I 
using TensorKit
using Statistics

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

function rotate_AA_cell(AA_fused_cell,construct_double_layer)
    if (Lx==1)&(Ly==1)
        AA_rotated_cell=rotate_AA(AA_fused_cell[1][1],construct_double_layer); 
    elseif (Lx==2)&(Ly==1)
        AA_11=rotate_AA(AA_fused_cell[1][1],construct_double_layer);
        AA_21=rotate_AA(AA_fused_cell[2][1],construct_double_layer);
        
        direction=1;
        AA_set1=((AA_11.T1,), (AA_21.T1,));
        direction=2;
        AA_set2=((AA_11.T2,), (AA_21.T2,));
        direction=3;
        AA_set3=((AA_11.T3,), (AA_21.T3,));
        direction=4;
        AA_set4=((AA_11.T4,), (AA_21.T4,));

        AA_rotated_cell=(T1=AA_set1, T2=AA_set2, T3=AA_set3, T4=AA_set4);
    elseif (Lx==1)&(Ly==2)
        AA_11=rotate_AA(AA_fused_cell[1][1],construct_double_layer);
        AA_12=rotate_AA(AA_fused_cell[1][2],construct_double_layer);
        
        direction=1;
        AA_set1=((AA_11.T1, AA_12.T1),);
        direction=2;
        AA_set2=((AA_11.T2, AA_12.T2),);
        direction=3;
        AA_set3=((AA_11.T3, AA_12.T3),);
        direction=4;
        AA_set4=((AA_11.T4, AA_12.T4),);

        AA_rotated_cell=(T1=AA_set1, T2=AA_set2, T3=AA_set3, T4=AA_set4);
    elseif (Lx==2)&(Ly==2)
        AA_11=rotate_AA(AA_fused_cell[1][1],construct_double_layer);
        AA_12=rotate_AA(AA_fused_cell[1][2],construct_double_layer);
        AA_21=rotate_AA(AA_fused_cell[2][1],construct_double_layer);
        AA_22=rotate_AA(AA_fused_cell[2][2],construct_double_layer);
        
        direction=1;
        AA_set1=((AA_11.T1, AA_12.T1), (AA_21.T1, AA_22.T1));
        direction=2;
        AA_set2=((AA_11.T2, AA_12.T2), (AA_21.T2, AA_22.T2));
        direction=3;
        AA_set3=((AA_11.T3, AA_12.T3), (AA_21.T3, AA_22.T3));
        direction=4;
        AA_set4=((AA_11.T4, AA_12.T4), (AA_21.T4, AA_22.T4));

        AA_rotated_cell=(T1=AA_set1, T2=AA_set2, T3=AA_set3, T4=AA_set4);
    elseif (Lx==3)&(Ly==2)
        AA_11=rotate_AA(AA_fused_cell[1][1],construct_double_layer);
        AA_12=rotate_AA(AA_fused_cell[1][2],construct_double_layer);
        AA_21=rotate_AA(AA_fused_cell[2][1],construct_double_layer);
        AA_22=rotate_AA(AA_fused_cell[2][2],construct_double_layer);
        AA_31=rotate_AA(AA_fused_cell[3][1],construct_double_layer);
        AA_32=rotate_AA(AA_fused_cell[3][2],construct_double_layer);
        
        direction=1;
        AA_set1=((AA_11.T1, AA_12.T1), (AA_21.T1, AA_22.T1), (AA_31.T1, AA_32.T1));
        direction=2;
        AA_set2=((AA_11.T2, AA_12.T2), (AA_21.T2, AA_22.T2), (AA_31.T2, AA_32.T2));
        direction=3;
        AA_set3=((AA_11.T3, AA_12.T3), (AA_21.T3, AA_22.T3), (AA_31.T3, AA_32.T3));
        direction=4;
        AA_set4=((AA_11.T4, AA_12.T4), (AA_21.T4, AA_22.T4), (AA_31.T4, AA_32.T4));

        AA_rotated_cell=(T1=AA_set1, T2=AA_set2, T3=AA_set3, T4=AA_set4);
    elseif (Lx==3)&(Ly==3)
        AA_11=rotate_AA(AA_fused_cell[1][1],construct_double_layer);
        AA_12=rotate_AA(AA_fused_cell[1][2],construct_double_layer);
        AA_13=rotate_AA(AA_fused_cell[1][3],construct_double_layer);
        AA_21=rotate_AA(AA_fused_cell[2][1],construct_double_layer);
        AA_22=rotate_AA(AA_fused_cell[2][2],construct_double_layer);
        AA_23=rotate_AA(AA_fused_cell[2][3],construct_double_layer);
        AA_31=rotate_AA(AA_fused_cell[3][1],construct_double_layer);
        AA_32=rotate_AA(AA_fused_cell[3][2],construct_double_layer);
        AA_33=rotate_AA(AA_fused_cell[3][3],construct_double_layer);
        
        direction=1;
        AA_set1=((AA_11.T1, AA_12.T1, AA_13.T1), (AA_21.T1, AA_22.T1, AA_23.T1), (AA_31.T1, AA_32.T1, AA_33.T1));
        direction=2;
        AA_set2=((AA_11.T2, AA_12.T2, AA_13.T2), (AA_21.T2, AA_22.T2, AA_23.T2), (AA_31.T2, AA_32.T2, AA_33.T2));
        direction=3;
        AA_set3=((AA_11.T3, AA_12.T3, AA_13.T3), (AA_21.T3, AA_22.T3, AA_23.T3), (AA_31.T3, AA_32.T3, AA_33.T3));
        direction=4;
        AA_set4=((AA_11.T4, AA_12.T4, AA_13.T4), (AA_21.T4, AA_22.T4, AA_23.T4), (AA_31.T4, AA_32.T4, AA_33.T4));

        AA_rotated_cell=(T1=AA_set1, T2=AA_set2, T3=AA_set3, T4=AA_set4);
    elseif (Lx==2)&(Ly==3)
        AA_11=rotate_AA(AA_fused_cell[1][1],construct_double_layer);
        AA_12=rotate_AA(AA_fused_cell[1][2],construct_double_layer);
        AA_13=rotate_AA(AA_fused_cell[1][3],construct_double_layer);
        AA_21=rotate_AA(AA_fused_cell[2][1],construct_double_layer);
        AA_22=rotate_AA(AA_fused_cell[2][2],construct_double_layer);
        AA_23=rotate_AA(AA_fused_cell[2][3],construct_double_layer);
        
        direction=1;
        AA_set1=((AA_11.T1, AA_12.T1, AA_13.T1), (AA_21.T1, AA_22.T1, AA_23.T1));
        direction=2;
        AA_set2=((AA_11.T2, AA_12.T2, AA_13.T2), (AA_21.T2, AA_22.T2, AA_23.T2));
        direction=3;
        AA_set3=((AA_11.T3, AA_12.T3, AA_13.T3), (AA_21.T3, AA_22.T3, AA_23.T3));
        direction=4;
        AA_set4=((AA_11.T4, AA_12.T4, AA_13.T4), (AA_21.T4, AA_22.T4, AA_23.T4));

        AA_rotated_cell=(T1=AA_set1, T2=AA_set2, T3=AA_set3, T4=AA_set4);
    elseif (Lx==4)&(Ly==2)
        AA_11=rotate_AA(AA_fused_cell[1][1],construct_double_layer);
        AA_12=rotate_AA(AA_fused_cell[1][2],construct_double_layer);
        AA_21=rotate_AA(AA_fused_cell[2][1],construct_double_layer);
        AA_22=rotate_AA(AA_fused_cell[2][2],construct_double_layer);
        AA_31=rotate_AA(AA_fused_cell[3][1],construct_double_layer);
        AA_32=rotate_AA(AA_fused_cell[3][2],construct_double_layer);
        AA_41=rotate_AA(AA_fused_cell[4][1],construct_double_layer);
        AA_42=rotate_AA(AA_fused_cell[4][2],construct_double_layer);
        
        direction=1;
        AA_set1=((AA_11.T1, AA_12.T1), (AA_21.T1, AA_22.T1), (AA_31.T1, AA_32.T1), (AA_41.T1, AA_42.T1));
        direction=2;
        AA_set2=((AA_11.T2, AA_12.T2), (AA_21.T2, AA_22.T2), (AA_31.T2, AA_32.T2), (AA_41.T2, AA_42.T2));
        direction=3;
        AA_set3=((AA_11.T3, AA_12.T3), (AA_21.T3, AA_22.T3), (AA_31.T3, AA_32.T3), (AA_41.T3, AA_42.T3));
        direction=4;
        AA_set4=((AA_11.T4, AA_12.T4), (AA_21.T4, AA_22.T4), (AA_31.T4, AA_32.T4), (AA_41.T4, AA_42.T4));

        AA_rotated_cell=(T1=AA_set1, T2=AA_set2, T3=AA_set3, T4=AA_set4);
    elseif (Lx==6)&(Ly==3)
        AA_11=rotate_AA(AA_fused_cell[1][1],construct_double_layer);
        AA_12=rotate_AA(AA_fused_cell[1][2],construct_double_layer);
        AA_13=rotate_AA(AA_fused_cell[1][3],construct_double_layer);
        AA_21=rotate_AA(AA_fused_cell[2][1],construct_double_layer);
        AA_22=rotate_AA(AA_fused_cell[2][2],construct_double_layer);
        AA_23=rotate_AA(AA_fused_cell[2][3],construct_double_layer);
        AA_31=rotate_AA(AA_fused_cell[3][1],construct_double_layer);
        AA_32=rotate_AA(AA_fused_cell[3][2],construct_double_layer);
        AA_33=rotate_AA(AA_fused_cell[3][3],construct_double_layer);
        AA_41=rotate_AA(AA_fused_cell[4][1],construct_double_layer);
        AA_42=rotate_AA(AA_fused_cell[4][2],construct_double_layer);
        AA_43=rotate_AA(AA_fused_cell[4][3],construct_double_layer);
        AA_51=rotate_AA(AA_fused_cell[5][1],construct_double_layer);
        AA_52=rotate_AA(AA_fused_cell[5][2],construct_double_layer);
        AA_53=rotate_AA(AA_fused_cell[5][3],construct_double_layer);
        AA_61=rotate_AA(AA_fused_cell[6][1],construct_double_layer);
        AA_62=rotate_AA(AA_fused_cell[6][2],construct_double_layer);
        AA_63=rotate_AA(AA_fused_cell[6][3],construct_double_layer);
        
        direction=1;
        AA_set1=((AA_11.T1, AA_12.T1, AA_13.T1), (AA_21.T1, AA_22.T1, AA_23.T1), (AA_31.T1, AA_32.T1, AA_33.T1), (AA_41.T1, AA_42.T1, AA_43.T1), (AA_51.T1, AA_52.T1, AA_53.T1), (AA_61.T1, AA_62.T1, AA_63.T1));
        direction=2;
        AA_set2=((AA_11.T2, AA_12.T2, AA_13.T2), (AA_21.T2, AA_22.T2, AA_23.T2), (AA_31.T2, AA_32.T2, AA_33.T2), (AA_41.T2, AA_42.T2, AA_43.T2), (AA_51.T2, AA_52.T2, AA_53.T2), (AA_61.T2, AA_62.T2, AA_63.T2));
        direction=3;
        AA_set3=((AA_11.T3, AA_12.T3, AA_13.T3), (AA_21.T3, AA_22.T3, AA_23.T3), (AA_31.T3, AA_32.T3, AA_33.T3), (AA_41.T3, AA_42.T3, AA_43.T3), (AA_51.T3, AA_52.T3, AA_53.T3), (AA_61.T3, AA_62.T3, AA_63.T3));
        direction=4;
        AA_set4=((AA_11.T4, AA_12.T4, AA_13.T4), (AA_21.T4, AA_22.T4, AA_23.T4), (AA_31.T4, AA_32.T4, AA_33.T4), (AA_41.T4, AA_42.T4, AA_43.T4), (AA_51.T4, AA_52.T4, AA_53.T4), (AA_61.T4, AA_62.T4, AA_63.T4));

        AA_rotated_cell=(T1=AA_set1, T2=AA_set2, T3=AA_set3, T4=AA_set4);
    elseif (Lx==6)&(Ly==6)
        AA_11=rotate_AA(AA_fused_cell[1][1],construct_double_layer);
        AA_12=rotate_AA(AA_fused_cell[1][2],construct_double_layer);
        AA_13=rotate_AA(AA_fused_cell[1][3],construct_double_layer);
        AA_14=rotate_AA(AA_fused_cell[1][4],construct_double_layer);
        AA_15=rotate_AA(AA_fused_cell[1][5],construct_double_layer);
        AA_16=rotate_AA(AA_fused_cell[1][6],construct_double_layer);
        AA_21=rotate_AA(AA_fused_cell[2][1],construct_double_layer);
        AA_22=rotate_AA(AA_fused_cell[2][2],construct_double_layer);
        AA_23=rotate_AA(AA_fused_cell[2][3],construct_double_layer);
        AA_24=rotate_AA(AA_fused_cell[2][4],construct_double_layer);
        AA_25=rotate_AA(AA_fused_cell[2][5],construct_double_layer);
        AA_26=rotate_AA(AA_fused_cell[2][6],construct_double_layer);
        AA_31=rotate_AA(AA_fused_cell[3][1],construct_double_layer);
        AA_32=rotate_AA(AA_fused_cell[3][2],construct_double_layer);
        AA_33=rotate_AA(AA_fused_cell[3][3],construct_double_layer);
        AA_34=rotate_AA(AA_fused_cell[3][4],construct_double_layer);
        AA_35=rotate_AA(AA_fused_cell[3][5],construct_double_layer);
        AA_36=rotate_AA(AA_fused_cell[3][6],construct_double_layer);
        AA_41=rotate_AA(AA_fused_cell[4][1],construct_double_layer);
        AA_42=rotate_AA(AA_fused_cell[4][2],construct_double_layer);
        AA_43=rotate_AA(AA_fused_cell[4][3],construct_double_layer);
        AA_44=rotate_AA(AA_fused_cell[4][4],construct_double_layer);
        AA_45=rotate_AA(AA_fused_cell[4][5],construct_double_layer);
        AA_46=rotate_AA(AA_fused_cell[4][6],construct_double_layer);
        AA_51=rotate_AA(AA_fused_cell[5][1],construct_double_layer);
        AA_52=rotate_AA(AA_fused_cell[5][2],construct_double_layer);
        AA_53=rotate_AA(AA_fused_cell[5][3],construct_double_layer);
        AA_54=rotate_AA(AA_fused_cell[5][4],construct_double_layer);
        AA_55=rotate_AA(AA_fused_cell[5][5],construct_double_layer);
        AA_56=rotate_AA(AA_fused_cell[5][6],construct_double_layer);
        AA_61=rotate_AA(AA_fused_cell[6][1],construct_double_layer);
        AA_62=rotate_AA(AA_fused_cell[6][2],construct_double_layer);
        AA_63=rotate_AA(AA_fused_cell[6][3],construct_double_layer);
        AA_64=rotate_AA(AA_fused_cell[6][4],construct_double_layer);
        AA_65=rotate_AA(AA_fused_cell[6][5],construct_double_layer);
        AA_66=rotate_AA(AA_fused_cell[6][6],construct_double_layer);
        
        direction=1;
        AA_set1=((AA_11.T1, AA_12.T1, AA_13.T1, AA_14.T1, AA_15.T1, AA_16.T1), (AA_21.T1, AA_22.T1, AA_23.T1, AA_24.T1, AA_25.T1, AA_26.T1), (AA_31.T1, AA_32.T1, AA_33.T1, AA_34.T1, AA_35.T1, AA_36.T1), (AA_41.T1, AA_42.T1, AA_43.T1, AA_44.T1, AA_45.T1, AA_46.T1), (AA_51.T1, AA_52.T1, AA_53.T1, AA_54.T1, AA_55.T1, AA_56.T1), (AA_61.T1, AA_62.T1, AA_63.T1, AA_64.T1, AA_65.T1, AA_66.T1));
        direction=2;
        AA_set2=((AA_11.T2, AA_12.T2, AA_13.T2, AA_14.T2, AA_15.T2, AA_16.T2), (AA_21.T2, AA_22.T2, AA_23.T2, AA_24.T2, AA_25.T2, AA_26.T2), (AA_31.T2, AA_32.T2, AA_33.T2, AA_34.T2, AA_35.T2, AA_36.T2), (AA_41.T2, AA_42.T2, AA_43.T2, AA_44.T2, AA_45.T2, AA_46.T2), (AA_51.T2, AA_52.T2, AA_53.T2, AA_54.T2, AA_55.T2, AA_56.T2), (AA_61.T2, AA_62.T2, AA_63.T2, AA_64.T2, AA_65.T2, AA_66.T2));
        direction=3;
        AA_set3=((AA_11.T3, AA_12.T3, AA_13.T3, AA_14.T3, AA_15.T3, AA_16.T3), (AA_21.T3, AA_22.T3, AA_23.T3, AA_24.T3, AA_25.T3, AA_26.T3), (AA_31.T3, AA_32.T3, AA_33.T3, AA_34.T3, AA_35.T3, AA_36.T3), (AA_41.T3, AA_42.T3, AA_43.T3, AA_44.T3, AA_45.T3, AA_46.T3), (AA_51.T3, AA_52.T3, AA_53.T3, AA_54.T3, AA_55.T3, AA_56.T3), (AA_61.T3, AA_62.T3, AA_63.T3, AA_64.T3, AA_65.T3, AA_66.T3));
        direction=4;
        AA_set4=((AA_11.T4, AA_12.T4, AA_13.T4, AA_14.T4, AA_15.T4, AA_16.T4), (AA_21.T4, AA_22.T4, AA_23.T4, AA_24.T4, AA_25.T4, AA_26.T4), (AA_31.T4, AA_32.T4, AA_33.T4, AA_34.T4, AA_35.T4, AA_36.T4), (AA_41.T4, AA_42.T4, AA_43.T4, AA_44.T4, AA_45.T4, AA_46.T4), (AA_51.T4, AA_52.T4, AA_53.T4, AA_54.T4, AA_55.T4, AA_56.T4), (AA_61.T4, AA_62.T4, AA_63.T4, AA_64.T4, AA_65.T4, AA_66.T4));

        AA_rotated_cell=(T1=AA_set1, T2=AA_set2, T3=AA_set3, T4=AA_set4);
    end

    return AA_rotated_cell
end

function Fermionic_CTMRG_cell(A_cell::Tuple,chi,init,CTM0, ctm_setting) 
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
        AA_fused_cell=initial_tuple_cell(Lx,Ly);
        U_L_cell=initial_tuple_cell(Lx,Ly);
        U_D_cell=initial_tuple_cell(Lx,Ly);
        U_R_cell=initial_tuple_cell(Lx,Ly);
        U_U_cell=initial_tuple_cell(Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                AA_fused_, U_L_,U_D_,U_R_,U_U_=build_double_layer_swap(A_cell[cx][cy]',A_cell[cx][cy]);
                AA_fused_cell=fill_tuple(AA_fused_cell, AA_fused_, cx,cy);
                U_L_cell=fill_tuple(U_L_cell, U_L_, cx,cy);
                U_D_cell=fill_tuple(U_D_cell, U_D_, cx,cy);
                U_R_cell=fill_tuple(U_R_cell, U_R_, cx,cy);
                U_U_cell=fill_tuple(U_U_cell, U_U_, cx,cy);
            end
        end
        AA_memory=@ignore_derivatives Base.summarysize(AA_fused_cell)/1024/1024;
        @ignore_derivatives if CTM_ite_info
            println("Memory cost of double layer tensor: "*string(AA_memory)*" Mb.");flush(stdout);
        end
    else
        AA_fused_cell=auxi_tensors.AA_fused_cell;
        U_L_cell=auxi_tensors.U_L_cell;
        U_D_cell=auxi_tensors.U_D_cell;
        U_R_cell=auxi_tensors.U_R_cell;
        U_U_cell=auxi_tensors.U_U_cell;
    end

    if init.reconstruct_CTM
        CTM_cell= init_CTM_cell(chi,A_cell,AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,  init.init_type,CTM_ite_info);
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
    
    if construct_double_layer
        AA_rotated_cell=rotate_AA_cell(AA_fused_cell,construct_double_layer);
    else
        AA_rotated_cell=rotate_AA(A_cell,construct_double_layer);
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
        for direction in direction_order

            if ctm_setting.grad_checkpoint #use checkpoint to save memory
                Cset_cell,Tset_cell= Zygote.checkpointed(CTM_ite_cell, Cset_cell, Tset_cell, get_Tset(AA_rotated_cell, direction), chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer);
            else
                Cset_cell,Tset_cell=CTM_ite_cell(Cset_cell, Tset_cell, get_Tset(AA_rotated_cell, direction), chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer);
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
        return CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err
    else
        return CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell
    end

end




function CTM_ite_cell_continuous_update(Cset_cell, Tset_cell, AA_cell, chi, direction, trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer)
    global Lx,Ly
    #println(direction)    
    #
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    if direction in [1,3]
        cx_max=Lx;
        cy_max=Ly;
    elseif direction in [2,4]
        cx_max=Ly;
        cy_max=Lx;
    end

    for cx=1:cx_max

        PM_cell=initial_tuple_cell(Lx,Ly);
        PM_inv_cell=initial_tuple_cell(Lx,Ly);
        M1tem_cell=initial_tuple_cell(Lx,Ly);
        M5tem_cell=initial_tuple_cell(Lx,Ly);
        M7tem_cell=initial_tuple_cell(Lx,Ly);

        for cy=1:cy_max
            coord=[cx,cy];

            Pos=convert_cell_posit(coord[1],coord[2],1,1,direction);
            AA=AA_cell[Pos[1]][Pos[2]];
            Pos=convert_cell_posit(coord[1],coord[2],0,0,direction);
            C1=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],1,0,direction);
            T1=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],0,1,direction);
            T4=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));
            @tensor MMup[:]:=C1[1,2]*T1[2,3,-3]*T4[-1,4,1]*AA[4,-2,-4,3];

            Pos=convert_cell_posit(coord[1],coord[2],1,2,direction);
            AA=AA_cell[Pos[1]][Pos[2]];
            Pos=convert_cell_posit(coord[1],coord[2],0,2,direction);
            T4=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));
            Pos=convert_cell_posit(coord[1],coord[2],0,3,direction);
            C4=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));
            Pos=convert_cell_posit(coord[1],coord[2],1,3,direction);
            T3=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-2,4));
            @tensor MMlow[:]:=T4[1,3,-1]*AA[3,4,-4,-2]*C4[2,1]*T3[-3,4,2];


            
            Pos=convert_cell_posit(coord[1],coord[2],2,1,direction);
            AA=AA_cell[Pos[1]][Pos[2]];
            Pos=convert_cell_posit(coord[1],coord[2],2,0,direction);
            T1=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],3,0,direction);
            C2=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction+1,4));
            Pos=convert_cell_posit(coord[1],coord[2],3,1,direction);
            T2=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction+1,4));
            @tensor MMup_reflect[:]:=T1[-1,3,1]* C2[1,2]* AA[-2,-4,4,3]* T2[2,4,-3];

            Pos=convert_cell_posit(coord[1],coord[2],2,2,direction);
            AA=AA_cell[Pos[1]][Pos[2]];
            Pos=convert_cell_posit(coord[1],coord[2],3,2,direction);
            T2=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction+1,4));
            Pos=convert_cell_posit(coord[1],coord[2],2,3,direction);
            T3=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-2,4));
            Pos=convert_cell_posit(coord[1],coord[2],3,3,direction);
            C3=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction-2,4));
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

            uM,sM,vM = my_tsvd(M; trunc=truncdim(chi+chi_extra));

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
            AA=AA_cell[Pos[1]][Pos[2]];
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



function CTM_ite_cell_together_update(Cset_cell, Tset_cell, AA_cell, chi, direction, trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer)
    global Lx,Ly
    #println(direction)    
    #
    """change of coordinate 
    (1,1)  (2,1)
    (1,2)  (2,2)

    coordinate of C1 tensor: (cx,cy)
    """
    PM_cell=initial_tuple_cell(Lx,Ly);
    PM_inv_cell=initial_tuple_cell(Lx,Ly);
    M1tem_cell=initial_tuple_cell(Lx,Ly);
    M5tem_cell=initial_tuple_cell(Lx,Ly);
    M7tem_cell=initial_tuple_cell(Lx,Ly);

    if direction in [1,3]
        cx_max=Lx;
        cy_max=Ly;
    elseif direction in [2,4]
        cx_max=Ly;
        cy_max=Lx;
    end

    for cx=1:cx_max
        for cy=1:cy_max
            coord=[cx,cy];

            Pos=convert_cell_posit(coord[1],coord[2],1,1,direction);
            AA=AA_cell[Pos[1]][Pos[2]];
            Pos=convert_cell_posit(coord[1],coord[2],0,0,direction);
            C1=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],1,0,direction);
            T1=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],0,1,direction);
            T4=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));
            @tensor MMup[:]:=C1[1,2]*T1[2,3,-3]*T4[-1,4,1]*AA[4,-2,-4,3];

            Pos=convert_cell_posit(coord[1],coord[2],1,2,direction);
            AA=AA_cell[Pos[1]][Pos[2]];
            Pos=convert_cell_posit(coord[1],coord[2],0,2,direction);
            T4=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));
            Pos=convert_cell_posit(coord[1],coord[2],0,3,direction);
            C4=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction-1,4));
            Pos=convert_cell_posit(coord[1],coord[2],1,3,direction);
            T3=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-2,4));
            @tensor MMlow[:]:=T4[1,3,-1]*AA[3,4,-4,-2]*C4[2,1]*T3[-3,4,2];


            
            Pos=convert_cell_posit(coord[1],coord[2],2,1,direction);
            AA=AA_cell[Pos[1]][Pos[2]];
            Pos=convert_cell_posit(coord[1],coord[2],2,0,direction);
            T1=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction,4));
            Pos=convert_cell_posit(coord[1],coord[2],3,0,direction);
            C2=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction+1,4));
            Pos=convert_cell_posit(coord[1],coord[2],3,1,direction);
            T2=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction+1,4));
            @tensor MMup_reflect[:]:=T1[-1,3,1]* C2[1,2]* AA[-2,-4,4,3]* T2[2,4,-3];

            Pos=convert_cell_posit(coord[1],coord[2],2,2,direction);
            AA=AA_cell[Pos[1]][Pos[2]];
            Pos=convert_cell_posit(coord[1],coord[2],3,2,direction);
            T2=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction+1,4));
            Pos=convert_cell_posit(coord[1],coord[2],2,3,direction);
            T3=get_Tset(Tset_cell[Pos[1]][Pos[2]], mod1(direction-2,4));
            Pos=convert_cell_posit(coord[1],coord[2],3,3,direction);
            C3=get_Cset(Cset_cell[Pos[1]][Pos[2]], mod1(direction-2,4));
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

            uM,sM,vM = my_tsvd(M; trunc=truncdim(chi+chi_extra));

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
    end
    for cx=1:cx_max
        for cy=1:cy_max
            coord=[cx,cy];

            Pos=convert_cell_posit(coord[1],coord[2],1,2,direction);
            AA=AA_cell[Pos[1]][Pos[2]];
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
            @tensor M5tem[:]:=T4[4,3,1]*AA[3,5,-2,2]* PM_inv_cell[Posa[1]][Posa[2]][4,5,-1]* PM_cell[Posb[1]][Posb[2]][1,2,-3];
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
    end


    for cx=1:cx_max
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


function init_CTM_cell(chi,A_cell,AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell, type,CTM_ite_info)
    @ignore_derivatives  if CTM_ite_info
        display("initialize CTM")
    end
    global Lx,Ly

    #numind(A)
    #numin(A)
    #numout(A)
    Cset_cell=initial_tuple_cell(Lx,Ly);
    Tset_cell=initial_tuple_cell(Lx,Ly);
    #space(A,1)
    if type=="PBC"
        for cx=1:Lx
            for cy=1:Ly
                CTM_=init_CTM_swap(chi,A_cell[cx][cy],AA_cell[cx][cy], U_L_cell[cx][cy],U_D_cell[cx][cy],U_R_cell[cx][cy],U_U_cell[cx][cy], type,false);
                Cset_cell=fill_tuple(Cset_cell,CTM_.Cset,cx,cy);
                Tset_cell=fill_tuple(Tset_cell,CTM_.Tset,cx,cy);
            end
        end
        CTM_cell=(Cset=Cset_cell,Tset=Tset_cell);
        return CTM_cell
    elseif type=="random"
    end
end


function Fermionic_CTMRG_cell_y_translate(A_cell::Tuple,chi,init,CTM0, ctm_setting) #for computing k in entanglement spectrum
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
        AA_fused_cell=initial_tuple_cell(Lx,Ly);
        U_L_cell=initial_tuple_cell(Lx,Ly);
        U_D_cell=initial_tuple_cell(Lx,Ly);
        U_R_cell=initial_tuple_cell(Lx,Ly);
        U_U_cell=initial_tuple_cell(Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                AA_fused_, U_L_,U_D_,U_R_,U_U_=build_double_layer_swap(A_cell[cx][cy]',A_cell[cx][mod1(cy+1,Ly)]);
                AA_fused_cell=fill_tuple(AA_fused_cell, AA_fused_, cx,cy);
                U_L_cell=fill_tuple(U_L_cell, U_L_, cx,cy);
                U_D_cell=fill_tuple(U_D_cell, U_D_, cx,cy);
                U_R_cell=fill_tuple(U_R_cell, U_R_, cx,cy);
                U_U_cell=fill_tuple(U_U_cell, U_U_, cx,cy);
            end
        end
        AA_memory=@ignore_derivatives Base.summarysize(AA_fused_cell)/1024/1024;
        @ignore_derivatives if CTM_ite_info
            println("Memory cost of double layer tensor: "*string(AA_memory)*" Mb.");flush(stdout);
        end
    else
        AA_fused_cell=auxi_tensors.AA_fused_cell;
        U_L_cell=auxi_tensors.U_L_cell;
        U_D_cell=auxi_tensors.U_D_cell;
        U_R_cell=auxi_tensors.U_R_cell;
        U_U_cell=auxi_tensors.U_U_cell;
    end

    if init.reconstruct_CTM
        CTM_cell= init_CTM_cell(chi,A_cell,AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,  init.init_type,CTM_ite_info);
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
    
    if construct_double_layer
        AA_rotated_cell=rotate_AA_cell(AA_fused_cell,construct_double_layer);
    else
        AA_rotated_cell=rotate_AA(A_cell,construct_double_layer);
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
        for direction in direction_order

            if ctm_setting.grad_checkpoint #use checkpoint to save memory
                Cset_cell,Tset_cell= Zygote.checkpointed(CTM_ite_cell, Cset_cell, Tset_cell, get_Tset(AA_rotated_cell, direction), chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer);
            else
                Cset_cell,Tset_cell=CTM_ite_cell(Cset_cell, Tset_cell, get_Tset(AA_rotated_cell, direction), chi, direction,CTM_trun_tol,CTM_ite_info,projector_strategy,CTM_trun_svd,svd_lanczos_tol,construct_double_layer);
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
        return CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err
    else
        return CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell
    end

end
