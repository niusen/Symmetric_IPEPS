
function build_fermi_cluster_2x2(mps_bot_set,mps_top_set,AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly)
    # AA_LU=remove_trivial_boundary(AA_LU,x_range[1],y_range[2],Lx,Ly);
    # AA_RU=remove_trivial_boundary(AA_RU,x_range[2],y_range[2],Lx,Ly);
    # AA_LD=remove_trivial_boundary(AA_LD,x_range[1],y_range[1],Lx,Ly);
    # AA_RD=remove_trivial_boundary(AA_RD,x_range[2],y_range[1],Lx,Ly);


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
            rho=build_fermi_cluster_2x2_bulk(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly);
        elseif yp=="top"
            rho=build_fermi_cluster_2x2_top(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly);
        elseif yp=="bot"
            rho=build_fermi_cluster_2x2_bot(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD,  VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly);
        end
    elseif xp=="left"
        if yp=="bulk"
            rho=build_fermi_cluster_2x2_left(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly);
        elseif yp=="top"
            rho=build_fermi_cluster_2x2_left_top(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly);
        elseif yp=="bot"
            rho=build_fermi_cluster_2x2_left_bot(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly);
        end
    elseif xp=="right"
        if yp=="bulk"
            rho=build_fermi_cluster_2x2_right(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly);
        elseif yp=="top"
            rho=build_fermi_cluster_2x2_right_top(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly);
        elseif yp=="bot"
            rho=build_fermi_cluster_2x2_right_bot(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly);
        end
    end

    return rho
end

function build_fermi_cluster_2x2_bulk(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly)
    
    @assert (x_range[1]>1)&(x_range[2]<Lx);
    @assert (y_range[1]>1)&(y_range[2]<Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################


    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];

    global left_right_env_method;
    if left_right_env_method=="exact"
        @tensor VL[:]:=VL0[1,3,5,7]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2]*AA_LD[5,6,-3,4]*mps_bot[x_range[1]][7,-4,6];
        @tensor VR[:]:=VR0[1,3,5,7]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2]*AA_RD[-3,6,5,4]*mps_bot[x_range[2]][-4,7,6];
    elseif left_right_env_method=="trun"
        @tensor VL[:]:=VL0[1][4,6,7]*VL0[2][7,2,1]*mps_top[x_range[1]][4,-1,5]*AA_LU[6,8,-2,5]*AA_LD[2,3,-3,8]*mps_bot[x_range[1]][1,-4,3];
        @tensor VR[:]:=VR0[1][1,3,8]*VR0[2][8,5,4]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,7,3,2]*AA_RD[-3,6,5,7]*mps_bot[x_range[2]][-4,4,6];
    end
    rho=@tensor VL[1,2,3,4]*VR[1,2,3,4];
    return rho
end



function build_fermi_cluster_2x2_left(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly)
    
    @assert (x_range[1]==1);
    @assert (y_range[1]>1)&(y_range[2]<Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################


    @tensor VL0[:]:=mps_top[1][-1,1]*AA_LU[2,-2,1]*AA_LD[3,-3,2]*mps_bot[1][-4,3];

    global left_right_env_method;
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    if left_right_env_method=="exact"
        @tensor VR[:]:=VR0[1,3,5,7]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2]*AA_RD[-3,6,5,4]*mps_bot[x_range[2]][-4,7,6];
    elseif left_right_env_method=="trun"
        @tensor VR[:]:=VR0[1][1,3,8]*VR0[2][8,5,4]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,7,3,2]*AA_RD[-3,6,5,7]*mps_bot[x_range[2]][-4,4,6];
    end
    rho=@tensor VL0[1,2,3,4]*VR[1,2,3,4];
    return rho
end


function build_fermi_cluster_2x2_right(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly)
    
    @assert (x_range[2]==Lx);
    @assert (y_range[1]>1)&(y_range[2]<Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################

    @tensor VR0[:]:=mps_top[Lx][-1,1]*AA_RU[-2,2,1]*AA_RD[-3,3,2]*mps_bot[Lx][-4,3];

    global left_right_env_method;
    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    if left_right_env_method=="exact"
        @tensor VL[:]:=VL0[1,3,5,7]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2]*AA_LD[5,6,-3,4]*mps_bot[x_range[1]][7,-4,6];
    elseif left_right_env_method=="trun"
        @tensor VL[:]:=VL0[1][4,6,7]*VL0[2][7,2,1]*mps_top[x_range[1]][4,-1,5]*AA_LU[6,8,-2,5]*AA_LD[2,3,-3,8]*mps_bot[x_range[1]][1,-4,3];
    end
    rho=@tensor VL[1,2,3,4]*VR0[1,2,3,4];
    return rho
end


function build_fermi_cluster_2x2_top(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly)
    
    @assert (x_range[1]>1)&(x_range[2]<Lx);
    @assert (y_range[2]==Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[Ly];

    ###############################################

    AA_LU=permute(AA_LU,(1,3,2,));
    AA_RU=permute(AA_RU,(1,3,2,));

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];

    @tensor VL[:]:=VL0[1,3,5]*AA_LU[1,-1,2]*AA_LD[3,4,-2,2]*mps_bot[x_range[1]][5,-3,4];  
    @tensor VR[:]:=VR0[1,3,5]*AA_RU[-1,1,2]*AA_RD[-2,4,3,2]*mps_bot[x_range[2]][-3,5,4];     

    rho=@tensor VL[1,2,3]*VR[1,2,3];

    return rho
end

function build_fermi_cluster_2x2_bot(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD,  VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly)
    
    @assert (x_range[1]>1)&(x_range[2]<Lx);
    @assert (y_range[1]==1);

    mps_bot=mps_bot_set[1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];

    @tensor VL[:]:=VL0[1,3,5]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2]*AA_LD[5,-3,4];
    @tensor VR[:]:=VR0[1,3,5]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2]*AA_RD[-3,5,4];
    rho=@tensor VL[1,2,3]*VR[1,2,3];

    return rho
end


function build_fermi_cluster_2x2_left_top(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly)
    
    @assert (x_range[1]==1);
    @assert (y_range[2]==Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[Ly];

    ###############################################

    AA_LU=permute(AA_LU,(2,1,));
    AA_RU=permute(AA_RU,(1,3,2,));

    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    @tensor VL0[:]:=AA_LU[-1,1]*AA_LD[2,-2,1]*mps_bot[1][-3,2];
    @tensor VR[:]:=VR0[1,3,5]*AA_RU[-1,1,2]*AA_RD[-2,4,3,2]*mps_bot[x_range[2]][-3,5,4];     
    rho=@tensor VL0[1,2,3]*VR[1,2,3];
    return rho
end


function build_fermi_cluster_2x2_right_top(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly)
    
    @assert (x_range[2]==Lx);
    @assert (y_range[2]==Ly);

    mps_bot=mps_bot_set[y_range[1]-1];
    mps_top=mps_top_set[Ly];

    ###############################################

    AA_LU=permute(AA_LU,(1,3,2,));

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    @tensor VR0[:]:=AA_RU[-1,1]*AA_RD[-2,2,1]*mps_bot[Lx][-3,2];
    @tensor VL[:]:=VL0[1,3,5]*AA_LU[1,-1,2]*AA_LD[3,4,-2,2]*mps_bot[x_range[1]][5,-3,4];  
    rho=@tensor VL[1,2,3]*VR0[1,2,3];
    return rho
end



function build_fermi_cluster_2x2_left_bot(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly)
    
    @assert (x_range[1]==1);
    @assert (y_range[1]==1);

    mps_bot=mps_bot_set[1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################

    @tensor VL0[:]:=mps_top[1][-1,1]*AA_LU[2,-2,1]*AA_LD[-3,2];
    VR0=VR_set_set[y_range[1]][x_range[2]+1];
    @tensor VR[:]:=VR0[1,3,5]*mps_top[x_range[2]][-1,1,2]*AA_RU[-2,4,3,2]*AA_RD[-3,5,4];
    rho=@tensor VL0[1,2,3]*VR[1,2,3];
    return rho
end




function build_fermi_cluster_2x2_right_bot(mps_bot_set,mps_top_set, AA_LU,AA_RU,AA_LD,AA_RD, VL_set_set,VR_set_set, x_range,y_range,  Lx,Ly)
    
    @assert (x_range[2]==Lx);
    @assert (y_range[1]==1);

    mps_bot=mps_bot_set[1];
    mps_top=mps_top_set[y_range[2]+1];

    ###############################################

    VL0=VL_set_set[y_range[1]][x_range[1]-1];
    @tensor VR0[:]:=mps_top[Lx][-1,1]*AA_RU[-2,2,1]*AA_RD[-3,2];
    @tensor VL[:]:=VL0[1,3,5]*mps_top[x_range[1]][1,-1,2]*AA_LU[3,4,-2,2]*AA_LD[5,-3,4];
    rho=@tensor VL[1,2,3]*VR0[1,2,3];
    return rho
end
