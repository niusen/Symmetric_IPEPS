
function sum_log(v::Vector)
    if length(v)==0
        return 0
    else
        return sum(log.(v));
    end
end




function norm_env(psi,psi_double,px,py, mps_set_down_move, norm_coe_down_move, mps_set_up_move, norm_coe_up_move)
    Lx,Ly=size(psi_double);
    I_phy=unitary(space(psi[2,2],5),space(psi[2,2],5));
    global U_phy
    U_s_s=@ignore_derivatives unitary(fuse(U_phy'*U_phy), U_phy' ⊗ U_phy);
    
    if py==1
        mps_up=mps_set_down_move[:,py+2];
        mpo=psi_double[:,py+1];
        mps_down=mps_set_up_move[:,py];
        if px==1
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=Lx-1:-1:2
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=mps_up[1][2,1]*mpo[1][-2,3,1]*envR[2,3,-1]*I_phy[7,8]*U_s_s'[7,8,-3];

        elseif 1<px<Lx
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=2:px-1
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            for cc=Lx-1:-1:px+1
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=mps_up[px][1,4,3]*mpo[px][2,-3,5,3]*envL[1,2,-1]*envR[4,5,-2]*I_phy[7,8]*U_s_s'[7,8,-4];
        elseif px==Lx
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            for cc=2:Lx-1
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            @tensor Norm[:]:=mps_up[Lx][2,1]*mpo[Lx][3,-2,1]*envL[2,3,-1]*I_phy[7,8]*U_s_s'[7,8,-3];
        end

        log_coe=sum_log(norm_coe_down_move[3:Ly-1]);
    elseif 1<py<Ly
        mps_up=mps_set_down_move[:,py+1];
        mpo=psi_double[:,py];
        mps_down=mps_set_up_move[:,py-1];

        if px==1
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=Lx-1:-1:2
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=mps_up[1][2,-3]*mps_down[1][1,-1]*envR[2,-2,1]*I_phy[7,8]*U_s_s'[7,8,-4];

        elseif 1<px<Lx
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=2:px-1
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            for cc=Lx-1:-1:px+1
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=mps_up[px][1,3,-4]*mps_down[px][4,2,-2]*envL[1,-1,4]*envR[3,-3,2]*I_phy[7,8]*U_s_s'[7,8,-5];
        elseif px==Lx
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            for cc=2:Lx-1
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            @tensor Norm[:]:=mps_up[Lx][2,-3]*mps_down[Lx][1,-2]*envL[2,-1,1]*I_phy[7,8]*U_s_s'[7,8,-4];
        end

        log_coe=sum_log(norm_coe_up_move[2:py-1])+sum_log(norm_coe_down_move[py+1:Ly-1]);
    elseif py==Ly
        mps_up=mps_set_down_move[:,py];
        mpo=psi_double[:,py-1];
        mps_down=mps_set_up_move[:,py-2];

        if px==1
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=Lx-1:-1:2
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=mpo[1][1,3,-2]*mps_down[1][2,1]*envR[-1,3,2]*I_phy[7,8]*U_s_s'[7,8,-3];

        elseif 1<px<Lx
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=2:px-1
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            for cc=Lx-1:-1:px+1
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=mpo[px][2,3,4,-3]*mps_down[px][1,5,3]*envL[-1,2,1]*envR[-2,4,5]*I_phy[7,8]*U_s_s'[7,8,-4];

        elseif px==Lx
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            for cc=2:Lx-1
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            @tensor Norm[:]:=mpo[Lx][2,1,-2]*mps_down[Lx][3,1]*envL[-1,2,3]*I_phy[7,8]*U_s_s'[7,8,-3];
        end


        log_coe=sum_log(norm_coe_up_move[2:Ly-2]);
    end

    return Norm,log_coe
end


function Ham_env_sum(psi,psi_double,px,py,x_range,y_range, data_down_move,data_up_move,only_nearest_plaquatte=false)
    Lx,Ly=size(psi);
    I_phy=unitary(space(psi[2,2],5),space(psi[2,2],5));
    T=similar(psi_double[px,py])*0;
    if Rank(T==2)
        @tensor H_env_zero[:]:=T[-1,-2]*I_phy[-3,-4];
    elseif Rank(T==3)
        @tensor H_env_zero[:]:=T[-1,-2,-3]*I_phy[-4,-5];
    elseif Rank(T==4)
        @tensor H_env_zero[:]:=T[-1,-2,-3,-4]*I_phy[-5,-6];
    end

    H_env_set=Matrix{TensorMap}(undef,Lx-1,Ly-1);
    H_coe_set=Matrix{TensorMap}(undef,Lx-1,Ly-1);

    for cx=1:Lx-1
        for cy=1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];

            if (abs(mean(x_range)-px)+abs(mean(y_range)-py)>1)& (only_nearest_plaquatte) #not nearest neighbour plaquatte
                H_env_set[cx,cy]=H_env_zero;
                H_coe_set[cx,cy]=0;
            else 
                term,coe=H_env_term(psi,psi_double,px,py,x_range,y_range, data_down_move,data_up_move);
                H_env_set[cx,cy]=term;
                H_coe_set[cx,cy]=coe;
            end

        end
    end
    return H_env_set,H_coe_set
end


function H_env_distant_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move)
    Lx,Ly=size(psi);
    psi_double_plaquatte=construct_double_layer_open_plaquatte(psi,x_range,y_range);
    if py==1
        log_coe=sum_log(data_down_move["norm_coe"][3:Ly-1]);
    elseif py==Ly
        log_coe=sum_log(data_up_move["norm_coe"][2:Ly-2]);
    elseif 1<py<Ly 
        log_coe=sum_log(data_up_move["norm_coe"][2:py-1])+sum_log(data_down_move["norm_coe"][py+1:Ly-1]);
    end
    
    global J1,J2,Jchi
    h_plaquatte=H_plaquatte(J1,J2,Jchi,x_range,y_range,Lx,Ly);

    if py<=y_range[1]-1 #site is below plaquatte
        
        if py==1 #site is at bottom
            mps_set_down_move_new=reconstruct_boundary_mps_down_move(psi_double_plaquatte, data_down_move["norm_coe"], data_down_move["unitarys_R_set"],data_down_move["unitarys_L_set"], data_down_move["UR_set"],data_down_move["UL_set"],data_down_move["projectors_R_set"],data_down_move["projectors_L_set"], py+2);
            mps_up=mps_set_down_move_new[:,py+2];
            mpo=psi_double_plaquatte[:,py+1]
            mps_down=psi_double_plaquatte[:,py];
            if y_range[1]==2
                term=contract_3Row_plaquatte_plaquatte_site(mps_up,mpo,mps_down,x_range,px,h_plaquatte);
            elseif y_range[1]>2
                term=contract_3Row_plaquatte_mps_site(mps_up,mpo,mps_down,x_range,px,h_plaquatte);
            end
        elseif py>1 #site is above bottom
            mps_set_down_move_new=reconstruct_boundary_mps_down_move(psi_double_plaquatte, data_down_move["norm_coe"], data_down_move["unitarys_R_set"],data_down_move["unitarys_L_set"], data_down_move["UR_set"],data_down_move["UL_set"],data_down_move["projectors_R_set"],data_down_move["projectors_L_set"], py+1);
            mps_up=mps_set_down_move_new[:,py+1];
            mpo=psi_double_plaquatte[:,py];
            mps_down=data_up_move["mps_set"][:,py-1];

            term=contract_3Row_plaquatte_site_mps(mps_up,mpo,mps_down,x_range,px,h_plaquatte)
        end

    elseif py>=y_range[2]+1 #site is above plaquatte
        
        if py==Ly #site is at top
            mps_set_up_move_new=reconstruct_boundary_mps_up_move(psi_double_plaquatte, data_up_move["norm_coe"], data_up_move["unitarys_R_set"],data_up_move["unitarys_L_set"], data_up_move["UR_set"],data_up_move["UL_set"],data_up_move["projectors_R_set"],data_up_move["projectors_L_set"], py-2);
            mps_up=psi_double_plaquatte[:,py]
            mps_up=pi_inverse_rotate_mps_down_move(pi_rotate_mps_down_move(mps_up));
            mpo=psi_double_plaquatte[:,py-1];
            mps_down=mps_set_up_move_new[:,py-2];
            if y_range[2]<Ly-1
                term=contract_3Row_site_mps_plaquatte(mps_up,mpo,mps_down,x_range,px,h_plaquatte);
            elseif y_range[2]==Ly-1
                term=contract_3Row_site_plaquatte_plaquatte(mps_up,mpo,mps_down,x_range,px,h_plaquatte);
            end
            
        elseif py<Ly #site is below top
            mps_set_up_move_new=reconstruct_boundary_mps_up_move(psi_double_plaquatte, data_up_move["norm_coe"], data_up_move["unitarys_R_set"],data_up_move["unitarys_L_set"], data_up_move["UR_set"],data_up_move["UL_set"],data_up_move["projectors_R_set"],data_up_move["projectors_L_set"], py-1);
            mps_up=data_down_move["mps_set"][:,py+1];
            mpo=psi_double_plaquatte[:,py];
            mps_down=mps_set_up_move_new[:,py-1];

            term=contract_3Row_mps_site_plaquatte(mps_up,mpo,mps_down,x_range,px,h_plaquatte)
        end
    end

    #####################################################
    if py==y_range[1] #plaquatte is at left or right side of site
        @assert (px<x_range[1]) | (px>x_range[2]); 
        
        if py==1
            mps_set_down_move_new=reconstruct_boundary_mps_down_move(psi_double_plaquatte, data_down_move["norm_coe"], data_down_move["unitarys_R_set"],data_down_move["unitarys_L_set"], data_down_move["UR_set"],data_down_move["UL_set"],data_down_move["projectors_R_set"],data_down_move["projectors_L_set"], py+2);
            mps_up=mps_set_down_move_new[:,py+2];
            mpo=psi_double_plaquatte[:,py+1];
            mps_down=psi_double_plaquatte[:,py];
            term=contract_3row_LR_plaquatte_23_site_bot(mps_up,mpo,mps_down,x_range,px,h_plaquatte);
        elseif py>1
            mps_set_down_move_new=reconstruct_boundary_mps_down_move(psi_double_plaquatte, data_down_move["norm_coe"], data_down_move["unitarys_R_set"],data_down_move["unitarys_L_set"], data_down_move["UR_set"],data_down_move["UL_set"],data_down_move["projectors_R_set"],data_down_move["projectors_L_set"], py+1);
            mps_up=mps_set_down_move_new[:,py+1];#thick bond
            mpo=psi_double_plaquatte[:,py];
            mps_down=data_up_move["mps_set"][:,py-1];
            term=contract_3row_LR_plaquatte_12(mps_up,mpo,mps_down,x_range,px,h_plaquatte);
        end 
        
    elseif py==y_range[2] #plaquatte is at left or right side of site
        @assert (px<x_range[1]) | (px>x_range[2]); 
        
        if py==Ly
            mps_set_up_move_new=reconstruct_boundary_mps_up_move(psi_double_plaquatte, data_up_move["norm_coe"], data_up_move["unitarys_R_set"],data_up_move["unitarys_L_set"], data_up_move["UR_set"],data_up_move["UL_set"],data_up_move["projectors_R_set"],data_up_move["projectors_L_set"], py-2);
            mps_up=psi_double_plaquatte[:,py]
            mps_up=pi_inverse_rotate_mps_down_move(pi_rotate_mps_down_move(mps_up));
            mpo=psi_double_plaquatte[:,py-1];
            mps_down=mps_set_up_move_new[:,py-2];
            term=contract_3row_LR_plaquatte_12_site_top(mps_up,mpo,mps_down,x_range,px,h_plaquatte);
        elseif py<Ly
            mps_set_up_move_new=reconstruct_boundary_mps_up_move(psi_double_plaquatte, data_up_move["norm_coe"], data_up_move["unitarys_R_set"],data_up_move["unitarys_L_set"], data_up_move["UR_set"],data_up_move["UL_set"],data_up_move["projectors_R_set"],data_up_move["projectors_L_set"], py-1);
            mps_up=data_down_move["mps_set"][:,py+1];
            mpo=psi_double_plaquatte[:,py];
            mps_down=mps_set_up_move_new[:,py-1];
            term=contract_3row_LR_plaquatte_23(mps_up,mpo,mps_down,x_range,px,h_plaquatte);
        end
    end

    
    return term,log_coe
end


function H_env_LU_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move)
    @assert x_range[2]==px;
    @assert y_range[1]==py;
    Lx,Ly=size(psi);
    global J1,J2,Jchi
    h_plaquatte=H_plaquatte(J1,J2,Jchi,x_range,y_range,Lx,Ly);
    
    psi_double_plaquatte=construct_double_layer_open_plaquatte(psi,x_range,y_range);
    if py==1
        log_coe=sum_log(data_down_move["norm_coe"][3:Ly-1]);
    elseif py==Ly
        log_coe=sum_log(data_up_move["norm_coe"][2:Ly-2]);
    elseif 1<py<Ly 
        log_coe=sum_log(data_up_move["norm_coe"][2:py-1])+sum_log(data_down_move["norm_coe"][py+1:Ly-1]);
    end


    if py==1 
        mps_set_down_move_new=reconstruct_boundary_mps_down_move(psi_double_plaquatte, data_down_move["norm_coe"], data_down_move["unitarys_R_set"],data_down_move["unitarys_L_set"], data_down_move["UR_set"],data_down_move["UL_set"],data_down_move["projectors_R_set"],data_down_move["projectors_L_set"], py+2);
        mps_up=mps_set_down_move_new[:,py+2];
        mpo=psi_double_plaquatte[:,py+1];
        mps_down=psi_double_plaquatte[:,py];

        if px==2
            envR=envR_3row(mps_up,mpo,mps_down,px+1);
            @tensor term[:]:=mps_up[px-1][5,1]*mpo[px-1][2,6,1,3]*mps_down[px-1][-1,2,4]*mps_up[px][5,11,7]*mpo[px][6,-3,12,7,8]*envR[11,12,-2]*h_plaquatte[3,8,-4,4];
        elseif 2<px<Lx
            envL=envL_3row(mps_up,mpo,mps_down,px-2);
            envR=envR_3row(mps_up,mpo,mps_down,px+1);
            @tensor term[:]:=envL[1,3,5]*mps_up[px-1][1,11,2]*mpo[px-1][3,4,12,2,6]*mps_down[px-1][5,-1,4,7]*mps_up[px][11,8,9]*mpo[px][12,-3,10,9,13]*h_plaquatte[6,13,-4,7]*envR[8,10,-2];
        elseif px==Lx
            envL=envL_3row(mps_up,mpo,mps_down,px-2);
            @tensor term[:]:=envL[1,3,5]*mps_up[px-1][1,9,2]*mpo[px-1][3,4,10,2,6]*mps_down[px-1][5,-1,4,7]*mps_up[px][9,8]*mpo[px][10,-2,8,11]*h_plaquatte[6,11,-3,7];
        end
    elseif py>1 #three rows
        mps_set_down_move_new=reconstruct_boundary_mps_down_move(psi_double_plaquatte, data_down_move["norm_coe"], data_down_move["unitarys_R_set"],data_down_move["unitarys_L_set"], data_down_move["UR_set"],data_down_move["UL_set"],data_down_move["projectors_R_set"],data_down_move["projectors_L_set"], py+1,false);
        mps_up=mps_set_down_move_new[:,py+1];#thick bond
        mpo=psi_double_plaquatte[:,py];
        mps_down=data_up_move["mps_set"][:,py-1];

        if px==2
            @tensor envL[:]:=mps_up[px-1][-1,1,3]*mpo[px-1][2,-2,1,4]*mps_down[px-1][-3,2]*h_plaquatte[3,-4,-5,4];
            envR=envR_3row(mps_up,mpo,mps_down,px+1);
            @tensor term[:]:=envL[2,-1,5,3,-5]*mps_up[px][2,4,-4,3]*mps_down[px][5,1,-2]*envR[4,-3,1];
        elseif 2<px<Lx
            envL=envL_3row(mps_up,mpo,mps_down,px-2);
            envR=envR_3row(mps_up,mpo,mps_down,px+1);
            @tensor envL[:]:=envL[5,3,1]*mps_up[px-1][5,-1,4,6]*mpo[px-1][3,2,-2,4,7]*mps_down[px-1][1,-3,2]*h_plaquatte[6,-4,-5,7];
            @tensor term[:]:=envL[2,-1,5,3,-5]*mps_up[px][2,4,-4,3]*mps_down[px][5,1,-2]*envR[4,-3,1];
        elseif px==Lx
            envL=envL_3row(mps_up,mpo,mps_down,px-2);
            @tensor envL[:]:=envL[5,3,1]*mps_up[px-1][5,-1,4,6]*mpo[px-1][3,2,-2,4,7]*mps_down[px-1][1,-3,2]*h_plaquatte[6,-4,-5,7];
            @tensor term[:]:=envL[1,-1,3,2,-4]*mps_up[px][1,-3,2]*mps_down[px][3,-2];
        end
        
    end

    return term,log_coe
end

function H_env_RU_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move)
    @assert x_range[1]==px;
    @assert y_range[1]==py;
    Lx,Ly=size(psi);
    global J1,J2,Jchi
    h_plaquatte=H_plaquatte(J1,J2,Jchi,x_range,y_range,Lx,Ly);

    psi_double_plaquatte=construct_double_layer_open_plaquatte(psi,x_range,y_range);
    if py==1
        log_coe=sum_log(data_down_move["norm_coe"][3:Ly-1]);
    elseif 1<py<Ly 
        log_coe=sum_log(data_up_move["norm_coe"][2:py-1])+sum_log(data_down_move["norm_coe"][py+1:Ly-1]);
    end

    
    
    if py==1 #two rows
        mps_set_down_move_new=reconstruct_boundary_mps_down_move(psi_double_plaquatte, data_down_move["norm_coe"], data_down_move["unitarys_R_set"],data_down_move["unitarys_L_set"], data_down_move["UR_set"],data_down_move["UL_set"],data_down_move["projectors_R_set"],data_down_move["projectors_L_set"], py+2);
        mps_up=mps_set_down_move_new[:,py+2];
        mpo=psi_double_plaquatte[:,py+1];
        mps_down=psi_double_plaquatte[:,py];

        if px==1
            envR=envR_3row(mps_up,mpo,mps_down,px+2);
            @tensor term[:]:=mps_up[px][9,8]*mpo[px][-2,10,8,11]*mps_up[px+1][9,1,2]*mpo[px+1][10,4,3,2,6]*mps_down[px+1][-1,5,4,7]*envR[1,3,5]*h_plaquatte[11,6,7,-3];
        elseif 1<px<Lx-1
            envL=envL_3row(mps_up,mpo,mps_down,px-1);
            envR=envR_3row(mps_up,mpo,mps_down,px+2);
            @tensor term[:]:=envL[8,10,-1]*mps_up[px][8,11,9]*mpo[px][10,-3,12,9,13]*mps_up[px+1][11,1,2]*mpo[px+1][12,4,3,2,6]*mps_down[px+1][-2,5,4,7]*h_plaquatte[13,6,7,-4]*envR[1,3,5];
        elseif px==Lx-1
            envL=envL_3row(mps_up,mpo,mps_down,px-1);
            @tensor term[:]:=envL[5,7,-1]*mps_up[px][5,8,6]*mpo[px][7,-3,9,6,10]*mps_up[px+1][8,1]*mpo[px+1][9,2,1,3]*mps_down[px+1][-2,2,4]*h_plaquatte[10,3,4,-4];
        end
    elseif py>1 #three rows
        mps_set_down_move_new=reconstruct_boundary_mps_down_move(psi_double_plaquatte, data_down_move["norm_coe"], data_down_move["unitarys_R_set"],data_down_move["unitarys_L_set"], data_down_move["UR_set"],data_down_move["UL_set"],data_down_move["projectors_R_set"],data_down_move["projectors_L_set"], py+1,false);
        mps_up=mps_set_down_move_new[:,py+1];#thick bond
        mpo=psi_double_plaquatte[:,py];
        mps_down=data_up_move["mps_set"][:,py-1];

        if px==1
            envR=envR_3row(mps_up,mpo,mps_down,px+2);
            @tensor envR[:]:=mps_up[px+1][-1,4,5,-4]*mpo[px+1][-2,3,2,5,-5]*mps_down[px+1][-3,1,3]envR[4,2,1];
            @tensor term[:]:=mps_up[px][3,-3,4]*mps_down[px][5,-1]*envR[3,-2,5,1,2]*h_plaquatte[4,1,2,-4];
        elseif 1<px<Lx-1
            envL=envL_3row(mps_up,mpo,mps_down,px-1);
            envR=envR_3row(mps_up,mpo,mps_down,px+2);
            @tensor envR[:]:=envR[4,2,1]*mps_up[px+1][-1,4,5,6]*mpo[px+1][-2,3,2,5,7]*mps_down[px+1][-3,1,3]*h_plaquatte[-4,6,7,-5];
            @tensor term[:]:=envL[4,-1,1]*mps_up[px][4,2,-4,3]*mps_down[px][1,5,-2]*envR[2,-3,5,3,-5];
        elseif px==Lx-1
            envL=envL_3row(mps_up,mpo,mps_down,px-1);
            @tensor envR[:]:=mps_up[px+1][-1,1,3]*mpo[px+1][-2,2,1,4]*mps_down[px+1][-3,2]*h_plaquatte[-4,3,4,-5];
            @tensor term[:]:=envL[4,-1,1]*mps_up[px][4,2,-4,3]*mps_down[px][1,5,-2]*envR[2,-3,5,3,-5];
        end
        
    end

    return term,log_coe

end

function H_env_LD_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move)
    @assert x_range[2]==px;
    @assert y_range[2]==py;
    Lx,Ly=size(psi);
    global J1,J2,Jchi
    h_plaquatte=H_plaquatte(J1,J2,Jchi,x_range,y_range,Lx,Ly);

    psi_double_plaquatte=construct_double_layer_open_plaquatte(psi,x_range,y_range);
    if py==1
        log_coe=sum_log(data_down_move["norm_coe"][3:Ly-1]);
    elseif py==Ly
        log_coe=sum_log(data_up_move["norm_coe"][2:Ly-2]);
    elseif 1<py<Ly 
        log_coe=sum_log(data_up_move["norm_coe"][2:py-1])+sum_log(data_down_move["norm_coe"][py+1:Ly-1]);
    end


    
    if py==Ly #two rows
        mps_set_up_move_new=reconstruct_boundary_mps_up_move(psi_double_plaquatte, data_up_move["norm_coe"], data_up_move["unitarys_R_set"],data_up_move["unitarys_L_set"], data_up_move["UR_set"],data_up_move["UL_set"],data_up_move["projectors_R_set"],data_up_move["projectors_L_set"], py-2);
        mps_up=psi_double_plaquatte[:,py];
        mps_up=pi_inverse_rotate_mps_down_move(pi_rotate_mps_down_move(mps_up));
        mpo=psi_double_plaquatte[:,py-1];
        mps_down=mps_set_up_move_new[:,py-2];

        if px==2
            envR=envR_3row(mps_up,mpo,mps_down,px+1);
            @tensor term[:]:=mps_up[px-1][-1,1,3]*mpo[px-1][2,6,1,4]*mps_down[px-1][5,2]*mpo[px][6,7,9,-3,8]*mps_down[px][5,10,7]*envR[-2,9,10]*h_plaquatte[3,-4,8,4];
        elseif 2<px<Lx
            envL=envL_3row(mps_up,mpo,mps_down,px-2);
            envR=envR_3row(mps_up,mpo,mps_down,px+1);
            @tensor term[:]:=envL[1,3,5]*mps_up[px-1][1,-1,2,6]*mpo[px-1][3,4,11,2,7]*mps_down[px-1][5,12,4]*mpo[px][11,10,9,-3,13]*mps_down[px][12,8,10]*h_plaquatte[6,-4,13,7]*envR[-2,9,8];
        elseif px==Lx
            envL=envL_3row(mps_up,mpo,mps_down,px-2);
            @tensor term[:]:=envL[1,3,5]*mps_up[px-1][1,-1,2,6]*mpo[px-1][3,4,9,2,7]*mps_down[px-1][5,10,4]*mpo[px][9,8,-2,11]*mps_down[px][10,8]*h_plaquatte[6,-3,11,7];
        end
    elseif py<Ly #three rows
        mps_set_up_move_new=reconstruct_boundary_mps_up_move(psi_double_plaquatte, data_up_move["norm_coe"], data_up_move["unitarys_R_set"],data_up_move["unitarys_L_set"], data_up_move["UR_set"],data_up_move["UL_set"],data_up_move["projectors_R_set"],data_up_move["projectors_L_set"], py-1,false);
        mps_up=data_down_move["mps_set"][:,py+1];
        mpo=psi_double_plaquatte[:,py];
        mps_down=mps_set_up_move_new[:,py-1];#thick bond

        if px==2
            @tensor envL[:]:=mps_up[px-1][-1,2]*mpo[px-1][1,-2,2,3]*mps_down[px-1][-3,1,4]*h_plaquatte[3,-4,-5,4];
            envR=envR_3row(mps_up,mpo,mps_down,px+1);
            @tensor term[:]:=envL[3,-1,1,-5,5]*mps_up[px][3,2,-4]*mps_down[px][1,4,-2,5]*envR[2,-3,4];
        elseif 2<px<Lx
            envL=envL_3row(mps_up,mpo,mps_down,px-2);
            envR=envR_3row(mps_up,mpo,mps_down,px+1);
            @tensor envL[:]:=envL[1,3,5]*mps_up[px-1][1,-1,2]*mpo[px-1][3,4,-2,2,6]*mps_down[px-1][5,-3,4,7]*h_plaquatte[6,-4,-5,7];
            @tensor term[:]:=envL[3,-1,1,-5,5]*mps_up[px][3,2,-4]*mps_down[px][1,4,-2,5]*envR[2,-3,4];
        elseif px==Lx
            envL=envL_3row(mps_up,mpo,mps_down,px-2);
            @tensor envL[:]:=envL[1,3,5]*mps_up[px-1][1,-1,2]*mpo[px-1][3,4,-2,2,6]*mps_down[px-1][5,-3,4,7]*h_plaquatte[6,-4,-5,7];
            @tensor term[:]:=envL[3,-1,1,-4,2]*mps_up[px][3,-3]*mps_down[px][1,-2,2];
        end
        
    end

    return term,log_coe
end

function H_env_RD_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move)
    @assert x_range[1]==px;
    @assert y_range[2]==py;
    Lx,Ly=size(psi);
    global J1,J2,Jchi
    h_plaquatte=H_plaquatte(J1,J2,Jchi,x_range,y_range,Lx,Ly);

    psi_double_plaquatte=construct_double_layer_open_plaquatte(psi,x_range,y_range);
    if py==1
        log_coe=sum_log(data_down_move["norm_coe"][3:Ly-1]);
    elseif py==Ly
        log_coe=sum_log(data_up_move["norm_coe"][2:Ly-2]);
    elseif 1<py<Ly 
        log_coe=sum_log(data_up_move["norm_coe"][2:py-1])+sum_log(data_down_move["norm_coe"][py+1:Ly-1]);
    end
    
    if py==Ly #two rows
        mps_set_up_move_new=reconstruct_boundary_mps_up_move(psi_double_plaquatte, data_up_move["norm_coe"], data_up_move["unitarys_R_set"],data_up_move["unitarys_L_set"], data_up_move["UR_set"],data_up_move["UL_set"],data_up_move["projectors_R_set"],data_up_move["projectors_L_set"], py-2);
        mps_up=psi_double_plaquatte[:,py];
        mps_up=pi_inverse_rotate_mps_down_move(pi_rotate_mps_down_move(mps_up));
        mpo=psi_double_plaquatte[:,py-1];
        mps_down=mps_set_up_move_new[:,py-2];

        if px==1
            envR=envR_3row(mps_up,mpo,mps_down,px+2);
            @tensor term[:]:=mpo[px][8,9,-2,11]*mps_down[px][10,8]*mps_up[px+1][-1,1,2,6]*mpo[px+1][9,4,3,2,7]*mps_down[px+1][10,5,4]*envR[1,3,5]*h_plaquatte[-3,6,7,11];
        elseif 1<px<Lx-1
            envL=envL_3row(mps_up,mpo,mps_down,px-1);
            envR=envR_3row(mps_up,mpo,mps_down,px+2);
            @tensor term[:]:=envL[-1,9,8]*mpo[px][9,10,11,-3,13]*mps_down[px][8,12,10]*mps_up[px+1][-2,1,2,6]*mpo[px+1][11,4,3,2,7]*mps_down[px+1][12,5,4]*h_plaquatte[-4,6,7,13]*envR[1,3,5];
        elseif px==Lx-1
            envL=envL_3row(mps_up,mpo,mps_down,px-1);
            @tensor term[:]:=envL[-1,5,7]*mpo[px][5,6,8,-3,10]*mps_down[px][7,9,6]*mps_up[px+1][-2,1,3]*mpo[px+1][8,2,1,4]*mps_down[px+1][9,2]*h_plaquatte[-4,3,4,10];
        end
    elseif py<Ly #three rows
        mps_set_up_move_new=reconstruct_boundary_mps_up_move(psi_double_plaquatte, data_up_move["norm_coe"], data_up_move["unitarys_R_set"],data_up_move["unitarys_L_set"], data_up_move["UR_set"],data_up_move["UL_set"],data_up_move["projectors_R_set"],data_up_move["projectors_L_set"], py-1,false);
        mps_up=data_down_move["mps_set"][:,py+1];
        mpo=psi_double_plaquatte[:,py];
        mps_down=mps_set_up_move_new[:,py-1];#thick bond

        if px==1
            envR=envR_3row(mps_up,mpo,mps_down,px+2);
            @tensor envR[:]:=mps_up[px+1][-1,4,5]*mpo[px+1][-2,3,2,5,-4]*mps_down[px+1][-3,1,3,-5]*envR[4,2,1];
            @tensor term[:]:=mps_up[px][3,-3]*mps_down[px][5,-1,4]*envR[3,-2,5,1,2]*h_plaquatte[-4,1,2,4];
        elseif 1<px<Lx-1
            envL=envL_3row(mps_up,mpo,mps_down,px-1);
            envR=envR_3row(mps_up,mpo,mps_down,px+2);
            @tensor envR[:]:=envR[1,3,5]*mps_up[px+1][-1,1,2]*mpo[px+1][-2,4,3,2,6]*mps_down[px+1][-3,5,4,7]*h_plaquatte[-4,6,7,-5];
            @tensor term[:]:=envL[3,-1,5]*mps_up[px][3,4,-4]*mps_down[px][5,1,-2,2]*envR[4,-3,1,-5,2];
        elseif px==Lx-1
            envL=envL_3row(mps_up,mpo,mps_down,px-1);
            @tensor envR[:]:=mps_up[px+1][-1,1]*mpo[px+1][-2,2,1,3]*mps_down[px+1][-3,2,4]*h_plaquatte[-4,3,4,-5];
            @tensor term[:]:=envL[3,-1,5]*mps_up[px][3,4,-4]*mps_down[px][5,1,-2,2]*envR[4,-3,1,-5,2];
        end
        
    end

    return term,log_coe

end

function H_env_nearest_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move)
    Lx,Ly=size(psi);
    if (x_range[2]==px) & (y_range[1]==py)
        term,log_coe=H_env_LU_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move);
    elseif (x_range[1]==px) & (y_range[1]==py)
        term,log_coe=H_env_RU_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move);
    elseif (x_range[2]==px) & (y_range[2]==py)
        term,log_coe=H_env_LD_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move);
    elseif (x_range[1]==px) & (y_range[2]==py)
        term,log_coe=H_env_RD_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move);
    end
    return term,log_coe
end

function Ham_env_term(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move)
    Lx,Ly=size(psi_double);
    I_phy=unitary(space(psi[2,2],5),space(psi[2,2],5));

    if (abs(mean(x_range)-px)+abs(mean(y_range)-py)>1) #distant term
        term,coe=H_env_distant_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move);
    else #nearest neighbour plaquatte
        term,coe=H_env_nearest_plaquatte(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move);
    end

    return term,coe
end