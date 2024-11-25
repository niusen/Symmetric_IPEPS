function truncate_mpo_mps(mpo,mps_old)
    global chi, multiplet_tol
    mps_new,_=apply_mpo(mpo,mps_old);
    mps,trun_errs,_=left_truncate_simple(mps_new, chi, multiplet_tol);
    return mps,trun_errs
end
function truncate_mpo_mps_exact(mpo,mps_old)
    global chi, multiplet_tol
    mps_new,_=apply_mpo(mpo,mps_old);
    # mps,trun_errs,_=left_truncate_simple(mps_new, chi, multiplet_tol);
    return mps_new,nothing
end
# function normalize_rho(rho,U_s_s)
#     @tensor rho[:]:=rho[1,2,3,4]*U_s_s[-1,-5,1]*U_s_s[-2,-6,2]*U_s_s[-3,-7,3]*U_s_s[-4,-8,4];
#     rho=permute(rho,(1,2,3,4,),(5,6,7,8,));
#     Norm=@tensor rho[1,2,3,4,1,2,3,4];
#     rho=rho/Norm;
#     return rho
# end

function build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);

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
            rho,U_s_s, trun_history=build_density_matrix_2x2_bulk(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        elseif yp=="top"
            rho,U_s_s, trun_history=build_density_matrix_2x2_top(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        elseif yp=="bot"
            rho,U_s_s, trun_history=build_density_matrix_2x2_bot(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        end
    elseif xp=="left"
        if yp=="bulk"
            rho,U_s_s, trun_history=build_density_matrix_2x2_left(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        elseif yp=="top"
            rho,U_s_s, trun_history=build_density_matrix_2x2_left_top(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        elseif yp=="bot"
            rho,U_s_s, trun_history=build_density_matrix_2x2_left_bot(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        end
    elseif xp=="right"
        if yp=="bulk"
            rho,U_s_s, trun_history=build_density_matrix_2x2_right(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        elseif yp=="top"
            rho,U_s_s, trun_history=build_density_matrix_2x2_right_top(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        elseif yp=="bot"
            rho,U_s_s, trun_history=build_density_matrix_2x2_right_bot(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
        end
    end

    return rho,U_s_s, trun_history
end

function build_density_matrix_2x2_bulk(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert (x_range[1]>1)&(x_range[2]<Lx);
    @assert (y_range[1]>1)&(y_range[2]<Ly);



    trun_history=[];

    mps_bot=(psi_double[:,1]...,);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:y_range[1]-1
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_bot);
        trun_history=vcat(trun_history,trun_errs);
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:y_range[2]+1
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_top);
        trun_history=vcat(trun_history,trun_errs);
    end

    #convert mps_top to normal order
    mps_top=mps_top[end:-1:1];

    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end

    ###############################################

    AA_LD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,1]);
    AA_LU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,2]);
    AA_RD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,1]);
    AA_RU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,2]);

    mpo_top=(psi_double[:,y_range[2]]...,);
    mpo_bot=(psi_double[:,y_range[1]]...,);

    @tensor VL[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
    for cx=2:x_range[1]-1
        @tensor VL[:]:=VL[1,3,5,7]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mpo_bot[cx][5,6,-3,4]*mps_bot[cx][7,-4,6];
    end

    @tensor VR[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
    for cx=Lx-1:-1:x_range[2]+1
        @tensor VR[:]:=VR[1,3,5,7]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mpo_bot[cx][-3,6,5,4]*mps_bot[cx][-4,7,6];
    end

    @tensor VL[:]:=VL[1,3,5,7]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2,-5]*AA_LD[5,6,-3,4,-6]*mps_bot[x_range[1]][7,-4,6];
    @tensor VR[:]:=VR[1,3,5,7]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2,-5]*AA_RD[-3,6,5,4,-6]*mps_bot[x_range[2]][-4,7,6];


    @tensor rho[:]:=VL[1,2,3,4,-1,-4]*VR[1,2,3,4,-2,-3];

    return rho,U_s_s, trun_history
end



function build_density_matrix_2x2_left(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert (x_range[1]==1);
    @assert (y_range[1]>1)&(y_range[2]<Ly);



    trun_history=[];

    mps_bot=(psi_double[:,1]...,);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:y_range[1]-1
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_bot);
        trun_history=vcat(trun_history,trun_errs);
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:y_range[2]+1
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_top);
        trun_history=vcat(trun_history,trun_errs);
    end

    #convert mps_top to normal order
    mps_top=mps_top[end:-1:1];

    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end


    ###############################################

    AA_LD, U_s_s=build_double_layer_left_open(iPEPS_2x2[1,1]);
    AA_LU, U_s_s=build_double_layer_left_open(iPEPS_2x2[1,2]);
    AA_RD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,1]);
    AA_RU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,2]);

    mpo_top=(psi_double[:,y_range[2]]...,);
    mpo_bot=(psi_double[:,y_range[1]]...,);

    @tensor VL[:]:=mps_top[1][-1,1]*AA_LU[2,-2,1,-5]*AA_LD[3,-3,2,-6]*mps_bot[1][-4,3];


    @tensor VR[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
    for cx=Lx-1:-1:x_range[2]+1
        @tensor VR[:]:=VR[1,3,5,7]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mpo_bot[cx][-3,6,5,4]*mps_bot[cx][-4,7,6];
    end

    @tensor VR[:]:=VR[1,3,5,7]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2,-5]*AA_RD[-3,6,5,4,-6]*mps_bot[x_range[2]][-4,7,6];


    @tensor rho[:]:=VL[1,2,3,4,-1,-4]*VR[1,2,3,4,-2,-3];

    return rho,U_s_s, trun_history
end


function build_density_matrix_2x2_right(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert (x_range[2]==Lx);
    @assert (y_range[1]>1)&(y_range[2]<Ly);



    trun_history=[];

    mps_bot=(psi_double[:,1]...,);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:y_range[1]-1
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_bot);
        trun_history=vcat(trun_history,trun_errs);
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:y_range[2]+1
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_top);
        trun_history=vcat(trun_history,trun_errs);
    end

    #convert mps_top to normal order
    mps_top=mps_top[end:-1:1];

    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end

    ###############################################

    AA_LD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,1]);
    AA_LU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,2]);
    AA_RD, U_s_s=build_double_layer_right_open(iPEPS_2x2[2,1]);
    AA_RU, U_s_s=build_double_layer_right_open(iPEPS_2x2[2,2]);

    mpo_top=(psi_double[:,y_range[2]]...,);
    mpo_bot=(psi_double[:,y_range[1]]...,);

    @tensor VL[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
    for cx=2:x_range[1]-1
        @tensor VL[:]:=VL[1,3,5,7]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mpo_bot[cx][5,6,-3,4]*mps_bot[cx][7,-4,6];
    end

    @tensor VR[:]:=mps_top[Lx][-1,1]*AA_RU[-2,2,1,-5]*AA_RD[-3,3,2,-6]*mps_bot[Lx][-4,3];

    @tensor VL[:]:=VL[1,3,5,7]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2,-5]*AA_LD[5,6,-3,4,-6]*mps_bot[x_range[1]][7,-4,6];

    @tensor rho[:]:=VL[1,2,3,4,-1,-4]*VR[1,2,3,4,-2,-3];

    return rho,U_s_s, trun_history
end


function build_density_matrix_2x2_top(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert (x_range[1]>1)&(x_range[2]<Lx);
    @assert (y_range[2]==Ly);



    trun_history=[];

    mps_bot=(psi_double[:,1]...,);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:y_range[1]-1
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_bot);
        trun_history=vcat(trun_history,trun_errs);
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);

    #convert mps_top to normal order
    mps_top=mps_top[end:-1:1];

    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end

    ###############################################

    AA_LD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,1]);
    AA_LU, U_s_s=build_double_layer_top_open(iPEPS_2x2[1,2]);
    AA_RD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,1]);
    AA_RU, U_s_s=build_double_layer_top_open(iPEPS_2x2[2,2]);

    AA_LU=permute(AA_LU,(1,3,2,4,));
    AA_RU=permute(AA_RU,(1,3,2,4,));

    mpo_bot=(psi_double[:,y_range[1]]...,);

    @tensor VL[:]:=mps_top[1][-1,1]*mpo_bot[1][2,-2,1]*mps_bot[1][-3,2];
    for cx=2:x_range[1]-1
        @tensor VL[:]:=VL[1,3,5]*mps_top[cx][1,-1,2]*mpo_bot[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
    end

    @tensor VR[:]:=mps_top[Lx][-1,1]*mpo_bot[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    for cx=Lx-1:-1:x_range[2]+1
        @tensor VR[:]:=VR[1,3,5]*mps_top[cx][-1,1,2]*mpo_bot[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
    end

    @tensor VL[:]:=VL[1,3,5]*AA_LU[1,-1,2,-4]*AA_LD[3,4,-2,2,-5]*mps_bot[x_range[1]][5,-3,4];  
    @tensor VR[:]:=VR[1,3,5]*AA_RU[-1,1,2,-4]*AA_RD[-2,4,3,2,-5]*mps_bot[x_range[2]][-3,5,4];     


    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR[1,2,3,-2,-3];

    return rho,U_s_s, trun_history
end

function build_density_matrix_2x2_bot(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert (x_range[1]>1)&(x_range[2]<Lx);
    @assert (y_range[1]==1);



    trun_history=[];

    mps_bot=(psi_double[:,1]...,);

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:y_range[2]+1
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_top);
        trun_history=vcat(trun_history,trun_errs);
    end

    #convert mps_top to normal order
    mps_top=(mps_top[end:-1:1]...,);

    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end

    ###############################################

    AA_LD, U_s_s=build_double_layer_bot_open(iPEPS_2x2[1,1]);
    AA_LU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,2]);
    AA_RD, U_s_s=build_double_layer_bot_open(iPEPS_2x2[2,1]);
    AA_RU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,2]);

    mpo_top=(psi_double[:,y_range[2]]...,);

    @tensor VL[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mps_bot[1][-3,2];
    for cx=2:x_range[1]-1
        @tensor VL[:]:=VL[1,3,5]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
    end

    @tensor VR[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    for cx=Lx-1:-1:x_range[2]+1
        @tensor VR[:]:=VR[1,3,5]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
    end

    @tensor VL[:]:=VL[1,3,5]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2,-4]*AA_LD[5,-3,4,-5];
    @tensor VR[:]:=VR[1,3,5]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2,-4]*AA_RD[-3,5,4,-5];


    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR[1,2,3,-2,-3];

    return rho,U_s_s, trun_history
end


function build_density_matrix_2x2_left_top(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert (x_range[1]==1);
    @assert (y_range[2]==Ly);



    trun_history=[];

    mps_bot=(psi_double[:,1]...,);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:y_range[1]-1
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_bot);
        trun_history=vcat(trun_history,trun_errs);
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);

    #convert mps_top to normal order
    mps_top=mps_top[end:-1:1];

    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end

    ###############################################

    AA_LD, U_s_s=build_double_layer_left_open(iPEPS_2x2[1,1]);
    AA_LU, U_s_s=build_double_layer_left_top_open(iPEPS_2x2[1,2]);
    AA_RD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,1]);
    AA_RU, U_s_s=build_double_layer_top_open(iPEPS_2x2[2,2]);

    AA_LU=permute(AA_LU,(2,1,3,));
    AA_RU=permute(AA_RU,(1,3,2,4,));

    mpo_bot=(psi_double[:,y_range[1]]...,);

    @tensor VL[:]:=AA_LU[-1,1,-4]*AA_LD[2,-2,1,-5]*mps_bot[1][-3,2];

    @tensor VR[:]:=mps_top[Lx][-1,1]*mpo_bot[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    for cx=Lx-1:-1:x_range[2]+1
        @tensor VR[:]:=VR[1,3,5]*mps_top[cx][-1,1,2]*mpo_bot[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
    end

    @tensor VR[:]:=VR[1,3,5]*AA_RU[-1,1,2,-4]*AA_RD[-2,4,3,2,-5]*mps_bot[x_range[2]][-3,5,4];     

    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR[1,2,3,-2,-3];

    return rho,U_s_s, trun_history
end


function build_density_matrix_2x2_right_top(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert (x_range[2]==Lx);
    @assert (y_range[2]==Ly);



    trun_history=[];

    mps_bot=(psi_double[:,1]...,);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:y_range[1]-1
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_bot);
        trun_history=vcat(trun_history,trun_errs);
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);

    #convert mps_top to normal order
    mps_top=mps_top[end:-1:1];

    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end

    ###############################################

    AA_LD, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,1]);
    AA_LU, U_s_s=build_double_layer_top_open(iPEPS_2x2[1,2]);
    AA_RD, U_s_s=build_double_layer_right_open(iPEPS_2x2[2,1]);
    AA_RU, U_s_s=build_double_layer_right_top_open(iPEPS_2x2[2,2]);

    AA_LU=permute(AA_LU,(1,3,2,4,));


    mpo_bot=(psi_double[:,y_range[1]]...,);

    @tensor VL[:]:=mps_top[1][-1,1]*mpo_bot[1][2,-2,1]*mps_bot[1][-3,2];
    for cx=2:x_range[1]-1
        @tensor VL[:]:=VL[1,3,5]*mps_top[cx][1,-1,2]*mpo_bot[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
    end

    @tensor VR[:]:=AA_RU[-1,1,-4]*AA_RD[-2,2,1,-5]*mps_bot[Lx][-3,2];

    @tensor VL[:]:=VL[1,3,5]*AA_LU[1,-1,2,-4]*AA_LD[3,4,-2,2,-5]*mps_bot[x_range[1]][5,-3,4];  


    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR[1,2,3,-2,-3];

    return rho,U_s_s, trun_history
end



function build_density_matrix_2x2_left_bot(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert (x_range[1]==1);
    @assert (y_range[1]==1);



    trun_history=[];

    mps_bot=(psi_double[:,1]...,);

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:y_range[2]+1
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_top);
        trun_history=vcat(trun_history,trun_errs);
    end

    #convert mps_top to normal order
    mps_top=mps_top[end:-1:1];

    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end

    ###############################################

    AA_LD, U_s_s=build_double_layer_left_bot_open(iPEPS_2x2[1,1]);
    AA_LU, U_s_s=build_double_layer_left_open(iPEPS_2x2[1,2]);
    AA_RD, U_s_s=build_double_layer_bot_open(iPEPS_2x2[2,1]);
    AA_RU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[2,2]);

    mpo_top=(psi_double[:,y_range[2]]...,);

    @tensor VL[:]:=mps_top[1][-1,1]*AA_LU[2,-2,1,-4]*AA_LD[-3,2,-5];


    @tensor VR[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    for cx=Lx-1:-1:x_range[2]+1
        @tensor VR[:]:=VR[1,3,5]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
    end

    @tensor VR[:]:=VR[1,3,5]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2,-4]*AA_RD[-3,5,4,-5];


    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR[1,2,3,-2,-3];

    return rho,U_s_s, trun_history
end




function build_density_matrix_2x2_right_bot(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol)
    Lx=size(psi,1);
    Ly=size(psi,2);
    @assert (x_range[2]==Lx);
    @assert (y_range[1]==1);



    trun_history=[];

    mps_bot=(psi_double[:,1]...,);

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:y_range[2]+1
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs=Zygote.checkpointed(truncate_mpo_mps, mpo, mps_top);
        trun_history=vcat(trun_history,trun_errs);
    end

    #convert mps_top to normal order
    mps_top=mps_top[end:-1:1];

    for cx=2:Lx-1
        mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
    end

    ###############################################

    AA_LD, U_s_s=build_double_layer_bot_open(iPEPS_2x2[1,1]);
    AA_LU, U_s_s=build_double_layer_bulk_open(iPEPS_2x2[1,2]);
    AA_RD, U_s_s=build_double_layer_right_bot_open(iPEPS_2x2[2,1]);
    AA_RU, U_s_s=build_double_layer_right_open(iPEPS_2x2[2,2]);

    mpo_top=(psi_double[:,y_range[2]]...,);

    @tensor VL[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mps_bot[1][-3,2];
    for cx=2:x_range[1]-1
        @tensor VL[:]:=VL[1,3,5]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
    end

    @tensor VR[:]:=mps_top[Lx][-1,1]*AA_RU[-2,2,1,-4]*AA_RD[-3,2,-5];

    @tensor VL[:]:=VL[1,3,5]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2,-4]*AA_LD[5,-3,4,-5];

    @tensor rho[:]:=VL[1,2,3,-1,-4]*VR[1,2,3,-2,-3];

    return rho,U_s_s, trun_history
end