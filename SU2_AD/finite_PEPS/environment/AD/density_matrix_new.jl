
function build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open=nothing)
    global Lx,Ly

    if (1<x_range[1])&(x_range[2]<Lx)
        xp="bulk";
    elseif (x_range[1]==1)
        xp="left";
    elseif (x_range[2]==Lx)
        xp="right";
    end

    if (1<y_range[1])&(y_range[2]<Ly)
        yp="bulk";
    elseif (y_range[1]==1)
        yp="bot";
    elseif (y_range[2]==Ly)
        yp="top";
    end

    if xp=="bulk"
        if yp=="bulk"
            rho,U_s_s=build_density_matrix_2x2_bulk_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
        elseif yp=="top"
            rho,U_s_s=build_density_matrix_2x2_top_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
        elseif yp=="bot"
            rho,U_s_s=build_density_matrix_2x2_bot_new(mps_bot_set,mps_top_set,iPEPS_2x2,  VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
        end
    elseif xp=="left"
        if yp=="bulk"
            rho,U_s_s=build_density_matrix_2x2_left_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
        elseif yp=="top"
            rho,U_s_s=build_density_matrix_2x2_left_top_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
        elseif yp=="bot"
            rho,U_s_s=build_density_matrix_2x2_left_bot_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
        end
    elseif xp=="right"
        if yp=="bulk"
            rho,U_s_s=build_density_matrix_2x2_right_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
        elseif yp=="top"
            rho,U_s_s=build_density_matrix_2x2_right_top_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
        elseif yp=="bot"
            rho,U_s_s=build_density_matrix_2x2_right_bot_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
        end
    end

    return rho,U_s_s
end

function build_density_matrix_2x2_bulk_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open)
    global Lx,Ly
    @assert (x_range[1]>1)&(x_range[2]<Lx);
    @assert (y_range[1]>1)&(y_range[2]<Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################

    if iPEPS_2x2==nothing
        AA_LD=psi_double_open[x_range[1],y_range[1]];
        AA_LU=psi_double_open[x_range[1],y_range[2]];
        AA_RD=psi_double_open[x_range[2],y_range[1]];
        AA_RU=psi_double_open[x_range[2],y_range[2]];
        global U_phy
        V_s=U_phy;
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);
        U_s_s=U_s_s';
    else
        AA_LD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,1],false);
        AA_LU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,2],false);
        AA_RD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,1],false);
        AA_RU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,2],false);
    end

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];

    global left_right_env_method;
    if left_right_env_method=="exact"
        @tensor VL[:]:=VL0[1,3,5,7]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2,-5]*AA_LD[5,6,-3,4,-6]*mps_bot[x_range[1]][7,-4,6];
        @tensor VR[:]:=VR0[1,3,5,7]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2,-5]*AA_RD[-3,6,5,4,-6]*mps_bot[x_range[2]][-4,7,6];
    elseif left_right_env_method=="trun"
        @tensor VL[:]:=VL0[1][4,6,7]*VL0[2][7,2,1]*mps_top[x_range[1]][4,-1,5]*AA_LU[6,8,-2,5,-5]*AA_LD[2,3,-3,8,-6]*mps_bot[x_range[1]][1,-4,3];
        @tensor VR[:]:=VR0[1][1,3,8]*VR0[2][8,5,4]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,7,3,2,-5]*AA_RD[-3,6,5,7,-6]*mps_bot[x_range[2]][-4,4,6];
    end
    @tensor rho[:]:=VL[1,2,3,4,-1,-4]*VR[1,2,3,4,-2,-3];
    return rho,U_s_s
end



function build_density_matrix_2x2_left_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open)
    global Lx,Ly
    @assert (x_range[1]==1);
    @assert (y_range[1]>1)&(y_range[2]<Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################

    if iPEPS_2x2==nothing
        AA_LD=psi_double_open[x_range[1],y_range[1]];
        AA_LU=psi_double_open[x_range[1],y_range[2]];
        AA_RD=psi_double_open[x_range[2],y_range[1]];
        AA_RU=psi_double_open[x_range[2],y_range[2]];
        global U_phy
        V_s=U_phy;
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);
        U_s_s=U_s_s';
    else
        AA_LD, U_s_s=build_double_layer_left_open(iPEPS_2x2[1,1],false);
        AA_LU, U_s_s=build_double_layer_left_open(iPEPS_2x2[1,2],false);
        AA_RD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,1],false);
        AA_RU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,2],false);
    end

    @tensor VL0[:]:=mps_top[1][-1,1]*AA_LU[2,-2,1,-5]*AA_LD[3,-3,2,-6]*mps_bot[1][-4,3];

    global left_right_env_method;
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    if left_right_env_method=="exact"
        @tensor VR[:]:=VR0[1,3,5,7]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2,-5]*AA_RD[-3,6,5,4,-6]*mps_bot[x_range[2]][-4,7,6];
    elseif left_right_env_method=="trun"
        @tensor VR[:]:=VR0[1][1,3,8]*VR0[2][8,5,4]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,7,3,2,-5]*AA_RD[-3,6,5,7,-6]*mps_bot[x_range[2]][-4,4,6];
    end
    @tensor rho[:]:=VL0[1,2,3,4,-1,-4]*VR[1,2,3,4,-2,-3];
    return rho,U_s_s
end


function build_density_matrix_2x2_right_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open)
    global Lx,Ly
    @assert (x_range[2]==Lx);
    @assert (y_range[1]>1)&(y_range[2]<Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################
    if iPEPS_2x2==nothing
        AA_LD=psi_double_open[x_range[1],y_range[1]];
        AA_LU=psi_double_open[x_range[1],y_range[2]];
        AA_RD=psi_double_open[x_range[2],y_range[1]];
        AA_RU=psi_double_open[x_range[2],y_range[2]];
        global U_phy
        V_s=U_phy;
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);
        U_s_s=U_s_s';
    else
        AA_LD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,1],false);
        AA_LU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,2],false);
        AA_RD, U_s_s=build_double_layer_right_open(iPEPS_2x2[2,1],false);
        AA_RU, U_s_s=build_double_layer_right_open(iPEPS_2x2[2,2],false);
    end

    @tensor VR0[:]:=mps_top[Lx][-1,1]*AA_RU[-2,2,1,-5]*AA_RD[-3,3,2,-6]*mps_bot[Lx][-4,3];

    global left_right_env_method;
    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    if left_right_env_method=="exact"
        @tensor VL[:]:=VL0[1,3,5,7]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2,-5]*AA_LD[5,6,-3,4,-6]*mps_bot[x_range[1]][7,-4,6];
    elseif left_right_env_method=="trun"
        @tensor VL[:]:=VL0[1][4,6,7]*VL0[2][7,2,1]*mps_top[x_range[1]][4,-1,5]*AA_LU[6,8,-2,5,-5]*AA_LD[2,3,-3,8,-6]*mps_bot[x_range[1]][1,-4,3];
    end
    @tensor rho[:]:=VL[1,2,3,4,-1,-4]*VR0[1,2,3,4,-2,-3];
    return rho,U_s_s
end


function build_density_matrix_2x2_top_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open)
    global Lx,Ly
    @assert (x_range[1]>1)&(x_range[2]<Lx);
    @assert (y_range[2]==Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[Ly];

    ###############################################
    if iPEPS_2x2==nothing
        AA_LD=psi_double_open[x_range[1],y_range[1]];
        AA_LU=psi_double_open[x_range[1],y_range[2]];
        AA_RD=psi_double_open[x_range[2],y_range[1]];
        AA_RU=psi_double_open[x_range[2],y_range[2]];
        global U_phy
        V_s=U_phy;
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);
        U_s_s=U_s_s';
    else
        AA_LD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,1],false);
        AA_LU, U_s_s=build_double_layer_top_open(iPEPS_2x2[1,2],false);
        AA_RD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,1],false);
        AA_RU, U_s_s=build_double_layer_top_open(iPEPS_2x2[2,2],false);
    end

    AA_LU=permute(AA_LU,(1,3,2,4,));
    AA_RU=permute(AA_RU,(1,3,2,4,));

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];

    @tensor VL[:]:=VL0[1,3,5]*AA_LU[1,-1,2,-4]*AA_LD[3,4,-2,2,-5]*mps_bot[x_range[1]][5,-3,4];  
    @tensor VR[:]:=VR0[1,3,5]*AA_RU[-1,1,2,-4]*AA_RD[-2,4,3,2,-5]*mps_bot[x_range[2]][-3,5,4];     

    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR[1,2,3,-2,-3];

    return rho,U_s_s
end

function build_density_matrix_2x2_bot_new(mps_bot_set,mps_top_set,iPEPS_2x2,  VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open)
    global Lx,Ly
    @assert (x_range[1]>1)&(x_range[2]<Lx);
    @assert (y_range[1]==1);

    mps_bot=mps_bot_set[1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################
    if iPEPS_2x2==nothing
        AA_LD=psi_double_open[x_range[1],y_range[1]];
        AA_LU=psi_double_open[x_range[1],y_range[2]];
        AA_RD=psi_double_open[x_range[2],y_range[1]];
        AA_RU=psi_double_open[x_range[2],y_range[2]];
        global U_phy
        V_s=U_phy;
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);
        U_s_s=U_s_s';
    else
        AA_LD, U_s_s=build_double_layer_bot_open(iPEPS_2x2[1,1],false);
        AA_LU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,2],false);
        AA_RD, U_s_s=build_double_layer_bot_open(iPEPS_2x2[2,1],false);
        AA_RU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,2],false);
    end

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];

    @tensor VL[:]:=VL0[1,3,5]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2,-4]*AA_LD[5,-3,4,-5];
    @tensor VR[:]:=VR0[1,3,5]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2,-4]*AA_RD[-3,5,4,-5];
    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR[1,2,3,-2,-3];

    return rho,U_s_s
end


function build_density_matrix_2x2_left_top_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open)
    global Lx,Ly
    @assert (x_range[1]==1);
    @assert (y_range[2]==Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[Ly];

    ###############################################
    if iPEPS_2x2==nothing
        AA_LD=psi_double_open[x_range[1],y_range[1]];
        AA_LU=psi_double_open[x_range[1],y_range[2]];
        AA_RD=psi_double_open[x_range[2],y_range[1]];
        AA_RU=psi_double_open[x_range[2],y_range[2]];
        global U_phy
        V_s=U_phy;
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);
        U_s_s=U_s_s';
    else
        AA_LD, U_s_s=build_double_layer_left_open(iPEPS_2x2[1,1],false);
        AA_LU, U_s_s=build_double_layer_left_top_open(iPEPS_2x2[1,2],false);
        AA_RD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,1],false);
        AA_RU, U_s_s=build_double_layer_top_open(iPEPS_2x2[2,2],false);
    end

    AA_LU=permute(AA_LU,(2,1,3,));
    AA_RU=permute(AA_RU,(1,3,2,4,));

    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    @tensor VL0[:]:=AA_LU[-1,1,-4]*AA_LD[2,-2,1,-5]*mps_bot[1][-3,2];
    @tensor VR[:]:=VR0[1,3,5]*AA_RU[-1,1,2,-4]*AA_RD[-2,4,3,2,-5]*mps_bot[x_range[2]][-3,5,4];     
    @tensor rho[:]:=VL0[1,2,3,-1,-4]*VR[1,2,3,-2,-3];
    return rho,U_s_s
end


function build_density_matrix_2x2_right_top_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open)
    global Lx,Ly
    @assert (x_range[2]==Lx);
    @assert (y_range[2]==Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[Ly];

    ###############################################
    if iPEPS_2x2==nothing
        AA_LD=psi_double_open[x_range[1],y_range[1]];
        AA_LU=psi_double_open[x_range[1],y_range[2]];
        AA_RD=psi_double_open[x_range[2],y_range[1]];
        AA_RU=psi_double_open[x_range[2],y_range[2]];
        global U_phy
        V_s=U_phy;
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);
        U_s_s=U_s_s';
    else
        AA_LD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,1],false);
        AA_LU, U_s_s=build_double_layer_top_open(iPEPS_2x2[1,2],false);
        AA_RD, U_s_s=build_double_layer_right_open(iPEPS_2x2[2,1],false);
        AA_RU, U_s_s=build_double_layer_right_top_open(iPEPS_2x2[2,2],false);
    end

    AA_LU=permute(AA_LU,(1,3,2,4,));

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    @tensor VR0[:]:=AA_RU[-1,1,-4]*AA_RD[-2,2,1,-5]*mps_bot[Lx][-3,2];
    @tensor VL[:]:=VL0[1,3,5]*AA_LU[1,-1,2,-4]*AA_LD[3,4,-2,2,-5]*mps_bot[x_range[1]][5,-3,4];  
    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR0[1,2,3,-2,-3];
    return rho,U_s_s
end



function build_density_matrix_2x2_left_bot_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open)
    global Lx,Ly
    @assert (x_range[1]==1);
    @assert (y_range[1]==1);

    mps_bot=mps_bot_set[1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################
    if iPEPS_2x2==nothing
        AA_LD=psi_double_open[x_range[1],y_range[1]];
        AA_LU=psi_double_open[x_range[1],y_range[2]];
        AA_RD=psi_double_open[x_range[2],y_range[1]];
        AA_RU=psi_double_open[x_range[2],y_range[2]];
        global U_phy
        V_s=U_phy;
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);
        U_s_s=U_s_s';
    else
        AA_LD, U_s_s=build_double_layer_left_bot_open(iPEPS_2x2[1,1],false);
        AA_LU, U_s_s=build_double_layer_left_open(iPEPS_2x2[1,2],false);
        AA_RD, U_s_s=build_double_layer_bot_open(iPEPS_2x2[2,1],false);
        AA_RU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,2],false);
    end

    @tensor VL0[:]:=mps_top[1][-1,1]*AA_LU[2,-2,1,-4]*AA_LD[-3,2,-5];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    @tensor VR[:]:=VR0[1,3,5]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2,-4]*AA_RD[-3,5,4,-5];
    @tensor rho[:]:=VL0[1,2,3,-1,-4]*VR[1,2,3,-2,-3];
    return rho,U_s_s
end




function build_density_matrix_2x2_right_bot_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open)
    global Lx,Ly
    @assert (x_range[2]==Lx);
    @assert (y_range[1]==1);

    mps_bot=mps_bot_set[1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################
    if iPEPS_2x2==nothing
        AA_LD=psi_double_open[x_range[1],y_range[1]];
        AA_LU=psi_double_open[x_range[1],y_range[2]];
        AA_RD=psi_double_open[x_range[2],y_range[1]];
        AA_RU=psi_double_open[x_range[2],y_range[2]];
        global U_phy
        V_s=U_phy;
        V_ss=@ignore_derivatives fuse(V_s' ⊗ V_s);
        U_s_s=@ignore_derivatives unitary(V_ss, V_s' ⊗ V_s);
        U_s_s=U_s_s';
    else
        AA_LD, U_s_s=build_double_layer_bot_open(iPEPS_2x2[1,1],false);
        AA_LU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,2],false);
        AA_RD, U_s_s=build_double_layer_right_bot_open(iPEPS_2x2[2,1],false);
        AA_RU, U_s_s=build_double_layer_right_open(iPEPS_2x2[2,2],false);
    end

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    @tensor VR0[:]:=mps_top[Lx][-1,1]*AA_RU[-2,2,1,-4]*AA_RD[-3,2,-5];
    @tensor VL[:]:=VL0[1,3,5]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2,-4]*AA_LD[5,-3,4,-5];
    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR0[1,2,3,-2,-3];
    return rho,U_s_s
end
