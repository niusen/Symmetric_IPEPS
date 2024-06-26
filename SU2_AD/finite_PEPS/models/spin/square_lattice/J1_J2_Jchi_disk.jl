

function  H_plaquatte(J1,J2,Jchi,x_range,y_range,Lx,Ly)


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
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="top"
            J_12=J1;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1;
            J_41=J1/2;
        end
    elseif xp=="left"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1;
        elseif yp=="top"
            J_12=J1;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1;
            J_41=J1;
        end
    elseif xp=="right"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="top"
            J_12=J1;
            J_23=J1;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1;
            J_34=J1;
            J_41=J1/2;
        end
    end


    J_13=J2;
    J_24=J2;

    J_123=Jchi;
    J_234=Jchi;
    J_341=Jchi;
    J_412=Jchi;

    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");
    Id=unitary(space(H_Heisenberg,1),space(H_Heisenberg,1));
    U_phy=Rep[SU₂](1/2=>1)';
    U_ss=unitary(fuse(U_phy' ⊗ U_phy), U_phy' ⊗ U_phy);
    U_ss=permute(U_ss',(3,1,2,));

    @tensor op_12[:]:=H_Heisenberg[1,2,3,4]*U_ss[-1,1,3]*U_ss[-2,2,4]*Id[5,6]*U_ss[-3,5,6]*Id[7,8]*U_ss[-4,7,8];
    @tensor op_13[:]:=H_Heisenberg[1,2,3,4]*U_ss[-1,1,3]*U_ss[-3,2,4]*Id[5,6]*U_ss[-2,5,6]*Id[7,8]*U_ss[-4,7,8];
    @tensor op_14[:]:=H_Heisenberg[1,2,3,4]*U_ss[-1,1,3]*U_ss[-4,2,4]*Id[5,6]*U_ss[-3,5,6]*Id[7,8]*U_ss[-2,7,8];
    @tensor op_23[:]:=H_Heisenberg[1,2,3,4]*U_ss[-2,1,3]*U_ss[-3,2,4]*Id[5,6]*U_ss[-1,5,6]*Id[7,8]*U_ss[-4,7,8];
    @tensor op_24[:]:=H_Heisenberg[1,2,3,4]*U_ss[-2,1,3]*U_ss[-4,2,4]*Id[5,6]*U_ss[-3,5,6]*Id[7,8]*U_ss[-1,7,8];
    @tensor op_34[:]:=H_Heisenberg[1,2,3,4]*U_ss[-3,1,3]*U_ss[-4,2,4]*Id[5,6]*U_ss[-1,5,6]*Id[7,8]*U_ss[-2,7,8];


    @tensor op_123[:]:=H123chiral[1,3,5,2,4,6]*U_ss[-1,1,2]*U_ss[-2,3,4]*U_ss[-3,5,6]*Id[7,8]*U_ss[-4,7,8];
    @tensor op_234[:]:=H123chiral[1,3,5,2,4,6]*U_ss[-2,1,2]*U_ss[-3,3,4]*U_ss[-4,5,6]*Id[7,8]*U_ss[-1,7,8];
    @tensor op_341[:]:=H123chiral[1,3,5,2,4,6]*U_ss[-3,1,2]*U_ss[-4,3,4]*U_ss[-1,5,6]*Id[7,8]*U_ss[-2,7,8];
    @tensor op_412[:]:=H123chiral[1,3,5,2,4,6]*U_ss[-4,1,2]*U_ss[-1,3,4]*U_ss[-2,5,6]*Id[7,8]*U_ss[-3,7,8];

    h_plaquatte=J_12*op_12+J_13*op_13+J_41*op_14+J_23*op_23+J_24*op_24+J_34*op_34+J_123*op_123+J_234*op_234+J_341*op_341+J_412*op_412;
    #u,s,v=tsvd(h_plaquatte,(1,2,),(3,4,));
    
    return h_plaquatte
end



function H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho)
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
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="top"
            J_12=J1;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1;
            J_41=J1/2;
        end
    elseif xp=="left"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1;
        elseif yp=="top"
            J_12=J1;
            J_23=J1/2;
            J_34=J1/2;
            J_41=J1;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1/2;
            J_34=J1;
            J_41=J1;
        end
    elseif xp=="right"
        if yp=="bulk"
            J_12=J1/2;
            J_23=J1;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="top"
            J_12=J1;
            J_23=J1;
            J_34=J1/2;
            J_41=J1/2;
        elseif yp=="bot"
            J_12=J1/2;
            J_23=J1;
            J_34=J1;
            J_41=J1/2;
        end
    end

    E_12=@tensor rho[1,2,5,6,3,4,5,6]*H_Heisenberg[1,2,3,4];
    E_34=@tensor rho[5,6,1,2,5,6,3,4]*H_Heisenberg[1,2,3,4];
    E_41=@tensor rho[1,5,6,2,3,5,6,4]*H_Heisenberg[1,2,3,4];
    E_23=@tensor rho[5,1,2,6,5,3,4,6]*H_Heisenberg[1,2,3,4];

    E_24=@tensor rho[5,1,6,2,5,3,6,4]*H_Heisenberg[1,2,3,4];
    E_13=@tensor rho[1,5,2,6,3,5,4,6]*H_Heisenberg[1,2,3,4];
    #println([E_12,E_34,E_41,E_23,E_24,E_13])

    E_123=@tensor rho[1,2,3,7,4,5,6,7]*H123chiral[1,2,3,4,5,6];
    E_234=@tensor rho[7,1,2,3,7,4,5,6]*H123chiral[1,2,3,4,5,6];
    E_341=@tensor rho[3,7,1,2,6,7,4,5]*H123chiral[1,2,3,4,5,6];
    E_412=@tensor rho[2,3,7,1,5,6,7,4]*H123chiral[1,2,3,4,5,6];

    E=J_12*E_12+J_23*E_23+J_34*E_34+J_41*E_41+J2*(E_24+E_13)+Jchi*(E_123+E_234+E_341+E_412);

    return E
end



function energy_disk_(psi)
    global chi, multiplet_tol
    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end

    Lx=size(psi,1);
    Ly=size(psi,2);
    J1=parameters["J1"];
    J2=parameters["J2"];
    Jchi=parameters["Jchi"];

    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """

    Lx=size(psi,1);
    Ly=size(psi,2);

    psi_double=construct_double_layer(psi,psi);
    
@time begin
    ########################################
    #construct top and bot environment

    trun_history=[];
    mps_bot_set=initial_tuple(Ly);
    mps_top_set=initial_tuple(Ly);

    mps_bot=(psi_double[:,1]...,);
    mps_bot_set=vector_update(mps_bot_set,mps_bot,1);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:Ly-2
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
        end
        return mps_top
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),Ly);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:3
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
        mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
        trun_history=vcat(trun_history,trun_errs);
    end

    #global trun_history
    # println(trun_history)
    ########################################
end
@time begin
    #construct left anf right environment
    
    VL_set_set=initial_tuple(Ly);
    VR_set_set=initial_tuple(Ly);

    cy=1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+2];
    mpo_top=(psi_double[:,cy+1]...,);
    mps_bot=mps_bot_set[cy];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);

    for cy=2:Ly-2
        VL_set=initial_tuple(Lx);
        VR_set=initial_tuple(Lx);
        mps_top=mps_top_set[cy+2];
        mpo_top=(psi_double[:,cy+1]...,);
        mpo_bot=(psi_double[:,cy]...,);
        mps_bot=mps_bot_set[cy-1];
        if left_right_env_method=="exact"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            VL_set=vector_update(VL_set,vl,1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl[1,3,5,7]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mpo_bot[cx][5,6,-3,4]*mps_bot[cx][7,-4,6];
                VL_set=vector_update(VL_set,vl,cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            VR_set=vector_update(VR_set,vr,Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr[1,3,5,7]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mpo_bot[cx][-3,6,5,4]*mps_bot[cx][-4,7,6];
                VR_set=vector_update(VR_set,vr,cx);
            end
        elseif left_right_env_method=="trun"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            vl_up,vl_dn=split_vl_or_vr(vl);
            VL_set=vector_update(VL_set,(vl_up,vl_dn,),1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl_up[4,6,7]*vl_dn[7,2,1]*mps_top[cx][4,-1,5]*mpo_top[cx][6,8,-2,5]*mpo_bot[cx][2,3,-3,8]*mps_bot[cx][1,-4,3];
                vl_up,vl_dn=split_vl_or_vr(vl);
                VL_set=vector_update(VL_set,(vl_up,vl_dn,),cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            vr_up,vr_dn=split_vl_or_vr(vr);
            VR_set=vector_update(VR_set,(vr_up,vr_dn,),Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr_up[1,3,8]*vr_dn[8,5,4]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,7,3,2]*mpo_bot[cx][-3,6,5,7]*mps_bot[cx][-4,4,6];
                vr_up,vr_dn=split_vl_or_vr(vr);
                VR_set=vector_update(VR_set,(vr_up,vr_dn,),cx);
            end
        end
        VL_set_set=vector_update(VL_set_set,VL_set,cy);
        VR_set_set=vector_update(VR_set_set,VR_set,cy);
    end

    cy=Ly-1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+1];
    mpo_bot=(psi_double[:,cy]...,);
    mps_bot=mps_bot_set[cy-1];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_bot[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_bot[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_bot[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_bot[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);

    ########################################
end

    E_total=0;
    E_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*1.0;

    for cx=1:Lx-1
        for cy=1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            @time rho_plaquatte,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
            rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
            E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives E_set[cx,cy]=E;
            E_total=E_total+E;
        end
    end

    return E_total,E_set
end

function energy_disk_old(A0,psi,psi_double,px,py)
    global chi, multiplet_tol

    Lx=size(psi,1);
    Ly=size(psi,2);
    J1=parameters["J1"];
    J2=parameters["J2"];
    Jchi=parameters["Jchi"];

    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """

    Lx=size(psi,1);
    Ly=size(psi,2);

    psi=matrix_update(psi,px,py,A0);

    ########################################
    if (px in 2:Lx-1)&(py in 2:Ly-1)
        AA,_=build_double_layer_bulk(psi[px,py],psi[px,py],[]);
    elseif (px==1)&(py in 2:Ly-1)
        AA,_=build_double_layer_left(psi[px,py],psi[px,py],[]);
    elseif (px==Lx)&(py in 2:Ly-1)
        AA,_=build_double_layer_right(psi[px,py],psi[px,py],[]);
    elseif (px in 2:Lx-1)&(py==1)
        AA,_=build_double_layer_bot(psi[px,py],psi[px,py],[]);
    elseif (px in 2:Lx-1)&(py==Ly)
        AA,_=build_double_layer_top(psi[px,py],psi[px,py],[]);
    elseif (px==1)&(py==1)
        AA,_=build_double_layer_left_bot(psi[px,py],psi[px,py],[]);
    elseif (px==1)&(py==Ly)
        AA,_=build_double_layer_left_top(psi[px,py],psi[px,py],[]);
    elseif (px==Lx)&(py==1)
        AA,_=build_double_layer_right_bot(psi[px,py],psi[px,py],[]);
    elseif (px==Lx)&(py==Ly)
        AA,_=build_double_layer_right_top(psi[px,py],psi[px,py],[]);
    end

    psi_double=matrix_update(psi_double,px,py,AA);
    ########################################
    
    

    E_total=0;
    E_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*1.0;

    for cx=1:Lx-1
        for cy=1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            rho_plaquatte,U_s_s,trun_history=build_density_matrix_2x2(iPEPS_2x2, psi_double,x_range,y_range, chi, multiplet_tol);
            rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
            E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives E_set[cx,cy]=E;
            E_total=E_total+E;
        end
    end

    return E_total
end

function update_prepare_local(psi,psi_double,px,py,A0)
    Lx,Ly=size(psi);
    psi=matrix_update(psi,px,py,A0);

    if psi_double==nothing
    else
        if (px in 2:Lx-1)&(py in 2:Ly-1)
            AA,_=build_double_layer_bulk(psi[px,py],psi[px,py],[]);
        elseif (px==1)&(py in 2:Ly-1)
            AA,_=build_double_layer_left(psi[px,py],psi[px,py],[]);
        elseif (px==Lx)&(py in 2:Ly-1)
            AA,_=build_double_layer_right(psi[px,py],psi[px,py],[]);
        elseif (px in 2:Lx-1)&(py==1)
            AA,_=build_double_layer_bot(psi[px,py],psi[px,py],[]);
        elseif (px in 2:Lx-1)&(py==Ly)
            AA,_=build_double_layer_top(psi[px,py],psi[px,py],[]);
        elseif (px==1)&(py==1)
            AA,_=build_double_layer_left_bot(psi[px,py],psi[px,py],[]);
        elseif (px==1)&(py==Ly)
            AA,_=build_double_layer_left_top(psi[px,py],psi[px,py],[]);
        elseif (px==Lx)&(py==1)
            AA,_=build_double_layer_right_bot(psi[px,py],psi[px,py],[]);
        elseif (px==Lx)&(py==Ly)
            AA,_=build_double_layer_right_top(psi[px,py],psi[px,py],[]);
        end
        psi_double=matrix_update(psi_double,px,py,AA);
    end
    return psi,psi_double
end


function energy_disk_new(A0,psi,psi_double,px,py,update_type)
    global chi, multiplet_tol

    #     global chi, multiplet_tol

    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end

    Lx=size(psi,1);
    Ly=size(psi,2);
    J1=parameters["J1"];
    J2=parameters["J2"];
    Jchi=parameters["Jchi"];

    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """

    Lx=size(psi,1);
    Ly=size(psi,2);


    ############################
    if update_type=="local"
        psi,psi_double=update_prepare_local(psi,psi_double,px,py,A0);
    elseif update_type=="bond"
        psi,psi_double=set_bond(psi,psi_double,px,py,A0);
    elseif update_type=="global"
        psi_double=construct_double_layer(psi, nothing);
    end

    ########################################
    #construct top and bot environment

    trun_history=[];
    mps_bot_set=initial_tuple(Ly);
    mps_top_set=initial_tuple(Ly);

    mps_bot=(psi_double[:,1]...,);
    mps_bot_set=vector_update(mps_bot_set,mps_bot,1);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:Ly-2
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs,_=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
        end
        return mps_top
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),Ly);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:3
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs,_=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
        mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
        trun_history=vcat(trun_history,trun_errs);
    end
    #global trun_history
    #println(trun_history)
    ########################################
    #construct left anf right environment
    VL_set_set=initial_tuple(Ly);
    VR_set_set=initial_tuple(Ly);

    cy=1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+2];
    mpo_top=(psi_double[:,cy+1]...,);
    mps_bot=mps_bot_set[cy];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);

    for cy=2:Ly-2
        VL_set=initial_tuple(Lx);
        VR_set=initial_tuple(Lx);
        mps_top=mps_top_set[cy+2];
        mpo_top=(psi_double[:,cy+1]...,);
        mpo_bot=(psi_double[:,cy]...,);
        mps_bot=mps_bot_set[cy-1];
        if left_right_env_method=="exact"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            VL_set=vector_update(VL_set,vl,1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl[1,3,5,7]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mpo_bot[cx][5,6,-3,4]*mps_bot[cx][7,-4,6];
                VL_set=vector_update(VL_set,vl,cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            VR_set=vector_update(VR_set,vr,Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr[1,3,5,7]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mpo_bot[cx][-3,6,5,4]*mps_bot[cx][-4,7,6];
                VR_set=vector_update(VR_set,vr,cx);
            end
        elseif left_right_env_method=="trun"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            vl_up,vl_dn=split_vl_or_vr(vl);
            VL_set=vector_update(VL_set,(vl_up,vl_dn,),1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl_up[4,6,7]*vl_dn[7,2,1]*mps_top[cx][4,-1,5]*mpo_top[cx][6,8,-2,5]*mpo_bot[cx][2,3,-3,8]*mps_bot[cx][1,-4,3];
                vl_up,vl_dn=split_vl_or_vr(vl);
                VL_set=vector_update(VL_set,(vl_up,vl_dn,),cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            vr_up,vr_dn=split_vl_or_vr(vr);
            VR_set=vector_update(VR_set,(vr_up,vr_dn,),Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr_up[1,3,8]*vr_dn[8,5,4]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,7,3,2]*mpo_bot[cx][-3,6,5,7]*mps_bot[cx][-4,4,6];
                vr_up,vr_dn=split_vl_or_vr(vr);
                VR_set=vector_update(VR_set,(vr_up,vr_dn,),cx);
            end
        end
        VL_set_set=vector_update(VL_set_set,VL_set,cy);
        VR_set_set=vector_update(VR_set_set,VR_set,cy);
    end

    cy=Ly-1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+1];
    mpo_bot=(psi_double[:,cy]...,);
    mps_bot=mps_bot_set[cy-1];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_bot[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_bot[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_bot[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_bot[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);


    ########################################


    E_total=0;
    E_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*1.0;

    for cx=1:Lx-1
        for cy=1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            rho_plaquatte,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
            rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
            E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives E_set[cx,cy]=E;
            E_total=E_total+E;
        end
    end

    return E_total
end


function energy_disk_test(A0,psi,psi_double,px,py,update_type)
    global chi, multiplet_tol

    #     global chi, multiplet_tol

    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end

    Lx=size(psi,1);
    Ly=size(psi,2);
    J1=parameters["J1"];
    J2=parameters["J2"];
    Jchi=parameters["Jchi"];

    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """

    Lx=size(psi,1);
    Ly=size(psi,2);


    ############################
    if update_type=="local"
        psi,psi_double=update_prepare_local(psi,psi_double,px,py,A0);
    elseif update_type=="bond"
        psi,psi_double=set_bond(psi,psi_double,px,py,A0);
    elseif update_type=="global"
        psi_double=construct_double_layer(psi, nothing);
    end

    ########################################
    #construct top and bot environment

    trun_history=[];
    mps_bot_set=initial_tuple(Ly);
    mps_top_set=initial_tuple(Ly);

    mps_bot=(psi_double[:,1]...,);
    mps_bot_set=vector_update(mps_bot_set,mps_bot,1);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:Ly-2
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs,_=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
        end
        return mps_top
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),Ly);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:3
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs,_=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
        mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
        trun_history=vcat(trun_history,trun_errs);
    end
    #global trun_history
    #println(trun_history)
    ########################################
    #construct left anf right environment
    VL_set_set=initial_tuple(Ly);
    VR_set_set=initial_tuple(Ly);

    cy=1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+2];
    mpo_top=(psi_double[:,cy+1]...,);
    mps_bot=mps_bot_set[cy];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);

    for cy=2:Ly-2
        VL_set=initial_tuple(Lx);
        VR_set=initial_tuple(Lx);
        mps_top=mps_top_set[cy+2];
        mpo_top=(psi_double[:,cy+1]...,);
        mpo_bot=(psi_double[:,cy]...,);
        mps_bot=mps_bot_set[cy-1];
        if left_right_env_method=="exact"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            VL_set=vector_update(VL_set,vl,1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl[1,3,5,7]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mpo_bot[cx][5,6,-3,4]*mps_bot[cx][7,-4,6];
                VL_set=vector_update(VL_set,vl,cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            VR_set=vector_update(VR_set,vr,Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr[1,3,5,7]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mpo_bot[cx][-3,6,5,4]*mps_bot[cx][-4,7,6];
                VR_set=vector_update(VR_set,vr,cx);
            end
        elseif left_right_env_method=="trun"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            vl_up,vl_dn=split_vl_or_vr(vl);
            VL_set=vector_update(VL_set,(vl_up,vl_dn,),1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl_up[4,6,7]*vl_dn[7,2,1]*mps_top[cx][4,-1,5]*mpo_top[cx][6,8,-2,5]*mpo_bot[cx][2,3,-3,8]*mps_bot[cx][1,-4,3];
                vl_up,vl_dn=split_vl_or_vr(vl);
                VL_set=vector_update(VL_set,(vl_up,vl_dn,),cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            vr_up,vr_dn=split_vl_or_vr(vr);
            VR_set=vector_update(VR_set,(vr_up,vr_dn,),Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr_up[1,3,8]*vr_dn[8,5,4]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,7,3,2]*mpo_bot[cx][-3,6,5,7]*mps_bot[cx][-4,4,6];
                vr_up,vr_dn=split_vl_or_vr(vr);
                VR_set=vector_update(VR_set,(vr_up,vr_dn,),cx);
            end
        end
        VL_set_set=vector_update(VL_set_set,VL_set,cy);
        VR_set_set=vector_update(VR_set_set,VR_set,cy);
    end

    cy=Ly-1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+1];
    mpo_bot=(psi_double[:,cy]...,);
    mps_bot=mps_bot_set[cy-1];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_bot[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_bot[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_bot[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_bot[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);


    ########################################


    E_total=0;
    E_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*1.0;

    for cx=2:2#1:Lx-1
        for cy=2:2#1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            rho_plaquatte,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
            E=normfun(rho_plaquatte,U_s_s);
            # E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives E_set[cx,cy]=E;
            E_total=E_total+E;
        end
    end

    return E_total
end

function normfun(rho,U_s_s)
    @tensor rho[:]:=rho[1,2,3,4]*U_s_s[-1,-5,1]*U_s_s[-2,-6,2]*U_s_s[-3,-7,3]*U_s_s[-4,-8,4];
    rho=@tensor rho[1,2,3,4,1,2,3,4];

    return rho
end

function cost_fun_local(x) #variational parameters are vector of TensorMap
    global chi, parameters, psi,psi_double,px,py

    E=energy_disk_new(x,psi,psi_double,px,py,"local");
    E=real(E);
    global E_tem;
    E_tem=E;
    return E
end

function cost_fun_local_test(x) #variational parameters are vector of TensorMap
    global chi, parameters, psi,psi_double,px,py

    E=energy_disk_test(x,psi,psi_double,px,py,"local");
    E=real(E);
    global E_tem;
    E_tem=E;
    return E
end


function cost_fun_global(psi) #variational parameters are vector of TensorMap
    global chi, parameters

    E=energy_disk_new(nothing,psi,nothing,nothing,nothing,"global");
    E=real(E);
    global E_tem;
    E_tem=E;
    return E
end





function cost_fun_double_layer(xx,px,py, psi_double_open,psi_double,U_s_s,funtype) #variational parameters are vector of TensorMap
    global chi, parameters
    if funtype=="energy"
        E,E_set=energy_disk_double_layer(xx,px,py, psi_double_open,psi_double,U_s_s);
    elseif funtype=="norm"
        E,E_set=norm_disk_double_layer(xx,px,py, psi_double_open,psi_double,U_s_s);
    end

    E=real(E);
    global E_tem;
    E_tem=E;
    # return E,E_set
    return E
end




function energy_disk_double_layer(AA0,px,py,psi_double_open,psi_double,U_s_s)
    global chi, multiplet_tol

    mpo_mps_fun=simple_truncate_to_moddle;

    Lx=size(psi,1);
    Ly=size(psi,2);
    J1=parameters["J1"];
    J2=parameters["J2"];
    Jchi=parameters["Jchi"];

    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """

    Lx=size(psi,1);
    Ly=size(psi,2);



    ########################################
    psi_double_open=matrix_update(psi_double_open,px,py,AA0);

    AA_contracted=contract_physical(AA0, U_s_s);
    psi_double=matrix_update(psi_double,px,py,AA_contracted);
    ########################################
    #construct top and bot environment

    trun_history=[];
    mps_bot_set=initial_tuple(Ly);
    mps_top_set=initial_tuple(Ly);
    log_coe_bot_set=initial_tuple(Ly);
    log_coe_top_set=initial_tuple(Ly);

    mps_bot=(psi_double[:,1]...,);
    mps_bot_set=vector_update(mps_bot_set,mps_bot,1);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:Ly-2
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
        log_coe_bot_set=vector_update(log_coe_bot_set,log(coe),cy);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
        end
        return mps_top
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),Ly);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:3
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
        log_coe_top_set=vector_update(log_coe_top_set,log(coe),cy);
        mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
        trun_history=vcat(trun_history,trun_errs);
    end
    #global trun_history
    #println(trun_history)
    ########################################
    #construct left anf right environment
    VL_set_set=initial_tuple(Ly);
    VR_set_set=initial_tuple(Ly);

    cy=1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+2];
    mpo_top=(psi_double[:,cy+1]...,);
    mps_bot=mps_bot_set[cy];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);

    for cy=2:Ly-2
        VL_set=initial_tuple(Lx);
        VR_set=initial_tuple(Lx);
        mps_top=mps_top_set[cy+2];
        mpo_top=(psi_double[:,cy+1]...,);
        mpo_bot=(psi_double[:,cy]...,);
        mps_bot=mps_bot_set[cy-1];
        if left_right_env_method=="exact"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            VL_set=vector_update(VL_set,vl,1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl[1,3,5,7]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mpo_bot[cx][5,6,-3,4]*mps_bot[cx][7,-4,6];
                VL_set=vector_update(VL_set,vl,cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            VR_set=vector_update(VR_set,vr,Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr[1,3,5,7]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mpo_bot[cx][-3,6,5,4]*mps_bot[cx][-4,7,6];
                VR_set=vector_update(VR_set,vr,cx);
            end
        elseif left_right_env_method=="trun"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            vl_up,vl_dn=split_vl_or_vr(vl);
            VL_set=vector_update(VL_set,(vl_up,vl_dn,),1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl_up[4,6,7]*vl_dn[7,2,1]*mps_top[cx][4,-1,5]*mpo_top[cx][6,8,-2,5]*mpo_bot[cx][2,3,-3,8]*mps_bot[cx][1,-4,3];
                vl_up,vl_dn=split_vl_or_vr(vl);
                VL_set=vector_update(VL_set,(vl_up,vl_dn,),cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            vr_up,vr_dn=split_vl_or_vr(vr);
            VR_set=vector_update(VR_set,(vr_up,vr_dn,),Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr_up[1,3,8]*vr_dn[8,5,4]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,7,3,2]*mpo_bot[cx][-3,6,5,7]*mps_bot[cx][-4,4,6];
                vr_up,vr_dn=split_vl_or_vr(vr);
                VR_set=vector_update(VR_set,(vr_up,vr_dn,),cx);
            end
        end
        VL_set_set=vector_update(VL_set_set,VL_set,cy);
        VR_set_set=vector_update(VR_set_set,VR_set,cy);
    end

    cy=Ly-1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+1];
    mpo_bot=(psi_double[:,cy]...,);
    mps_bot=mps_bot_set[cy-1];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_bot[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_bot[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_bot[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_bot[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);


    ########################################
    
    

    E_total=0;
    E_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*1.0;

    for cx=1:Lx-1
        for cy=1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            # iPEPS_2x2=psi[x_range,y_range];
            #rho_plaquatte,U_s_s,trun_history=build_density_matrix_2x2_new(psi_double_open, psi_double,x_range,y_range, chi, multiplet_tol);
            rho_plaquatte,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,nothing, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
            #rho_plaquatte=normalize_rho(rho_plaquatte,U_s_s);
            @tensor rho_plaquatte[:]:=rho_plaquatte[1,2,3,4]*U_s_s[-1,-5,1]*U_s_s[-2,-6,2]*U_s_s[-3,-7,3]*U_s_s[-4,-8,4];
            rho_plaquatte=permute(rho_plaquatte,(1,2,3,4,),(5,6,7,8,));
            E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            log_coe=0;
            if cy>=3
                log_coe=log_coe+sum(log_coe_bot_set[2:cy-1]);
            end
            if cy+2<=Ly-1
                log_coe=log_coe+sum(log_coe_top_set[cy+2:Ly-1]);
            end
            E=E*exp(log_coe);
            @ignore_derivatives E_set[cx,cy]=E;
            E_total=E_total+E;
        end
    end

    return E_total,E_set
end



function norm_disk_double_layer(AA0,px,py,psi_double_open,psi_double,U_s_s)
    global chi, multiplet_tol

    mpo_mps_fun=simple_truncate_to_moddle;

    Lx=size(psi,1);
    Ly=size(psi,2);



    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """

    Lx=size(psi,1);
    Ly=size(psi,2);


    ########################################
    psi_double_open=matrix_update(psi_double_open,px,py,AA0);

    AA_contracted=contract_physical(AA0, U_s_s);
    psi_double=matrix_update(psi_double,px,py,AA_contracted);
    ########################################
    #construct top and bot environment

    trun_history=[];
    mps_bot_set=initial_tuple(Ly);
    mps_top_set=initial_tuple(Ly);
    log_coe_bot_set=initial_tuple(Ly);
    log_coe_top_set=initial_tuple(Ly);

    mps_bot=(psi_double[:,1]...,);
    mps_bot_set=vector_update(mps_bot_set,mps_bot,1);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:Ly-2
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
        log_coe_bot_set=vector_update(log_coe_bot_set,log(coe),cy);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
    end


    function treat_mps_top(mps_top)
        #convert mps_top to normal order
        mps_top=mps_top[end:-1:1];
        for cx=2:Lx-1
            mps_top=mps_update(mps_top,permute(mps_top[cx],(2,1,3,)),cx);
        end
        return mps_top
    end

    mps_top=(psi_double[:,Ly]...,);
    mps_top=pi_rotate_mps(mps_top);
    mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),Ly);
    mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    trun_history=vcat(trun_history,trun_errs);
    for cy=Ly-1:-1:3
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
        log_coe_top_set=vector_update(log_coe_top_set,log(coe),cy);
        mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
        trun_history=vcat(trun_history,trun_errs);
    end
    #global trun_history
    #println(trun_history)
    ########################################
    #construct left anf right environment
    VL_set_set=initial_tuple(Ly);
    VR_set_set=initial_tuple(Ly);

    cy=1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+2];
    mpo_top=(psi_double[:,cy+1]...,);
    mps_bot=mps_bot_set[cy];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);

    for cy=2:Ly-2
        VL_set=initial_tuple(Lx);
        VR_set=initial_tuple(Lx);
        mps_top=mps_top_set[cy+2];
        mpo_top=(psi_double[:,cy+1]...,);
        mpo_bot=(psi_double[:,cy]...,);
        mps_bot=mps_bot_set[cy-1];
        if left_right_env_method=="exact"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            VL_set=vector_update(VL_set,vl,1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl[1,3,5,7]*mps_top[cx][1,-1,2]*mpo_top[cx][3,4,-2,2]*mpo_bot[cx][5,6,-3,4]*mps_bot[cx][7,-4,6];
                VL_set=vector_update(VL_set,vl,cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            VR_set=vector_update(VR_set,vr,Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr[1,3,5,7]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,4,3,2]*mpo_bot[cx][-3,6,5,4]*mps_bot[cx][-4,7,6];
                VR_set=vector_update(VR_set,vr,cx);
            end
        elseif left_right_env_method=="trun"
            @tensor vl[:]:=mps_top[1][-1,1]*mpo_top[1][2,-2,1]*mpo_bot[1][3,-3,2]*mps_bot[1][-4,3];
            vl_up,vl_dn=split_vl_or_vr(vl);
            VL_set=vector_update(VL_set,(vl_up,vl_dn,),1);
            for cx=2:Lx-2
                @tensor vl[:]:=vl_up[4,6,7]*vl_dn[7,2,1]*mps_top[cx][4,-1,5]*mpo_top[cx][6,8,-2,5]*mpo_bot[cx][2,3,-3,8]*mps_bot[cx][1,-4,3];
                vl_up,vl_dn=split_vl_or_vr(vl);
                VL_set=vector_update(VL_set,(vl_up,vl_dn,),cx);
            end
            @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_top[Lx][-2,2,1]*mpo_bot[Lx][-3,3,2]*mps_bot[Lx][-4,3];
            vr_up,vr_dn=split_vl_or_vr(vr);
            VR_set=vector_update(VR_set,(vr_up,vr_dn,),Lx);
            for cx=Lx-1:-1:3
                @tensor vr[:]:=vr_up[1,3,8]*vr_dn[8,5,4]*mps_top[cx][-1,1,2]*mpo_top[cx][-2,7,3,2]*mpo_bot[cx][-3,6,5,7]*mps_bot[cx][-4,4,6];
                vr_up,vr_dn=split_vl_or_vr(vr);
                VR_set=vector_update(VR_set,(vr_up,vr_dn,),cx);
            end
        end
        VL_set_set=vector_update(VL_set_set,VL_set,cy);
        VR_set_set=vector_update(VR_set_set,VR_set,cy);
    end

    cy=Ly-1;
    VL_set=initial_tuple(Lx);
    VR_set=initial_tuple(Lx);
    mps_top=mps_top_set[cy+1];
    mpo_bot=(psi_double[:,cy]...,);
    mps_bot=mps_bot_set[cy-1];
    @tensor vl[:]:=mps_top[1][-1,1]*mpo_bot[1][2,-2,1]*mps_bot[1][-3,2];
    VL_set=vector_update(VL_set,vl,1);
    for cx=2:Lx-2
        @tensor vl[:]:=vl[1,3,5]*mps_top[cx][1,-1,2]*mpo_bot[cx][3,4,-2,2]*mps_bot[cx][5,-3,4];
        VL_set=vector_update(VL_set,vl,cx);
    end
    @tensor vr[:]:=mps_top[Lx][-1,1]*mpo_bot[Lx][-2,2,1]*mps_bot[Lx][-3,2];
    VR_set=vector_update(VR_set,vr,Lx);
    for cx=Lx-1:-1:3
        @tensor vr[:]:=vr[1,3,5]*mps_top[cx][-1,1,2]*mpo_bot[cx][-2,4,3,2]*mps_bot[cx][-3,5,4];
        VR_set=vector_update(VR_set,vr,cx);
    end
    VL_set_set=vector_update(VL_set_set,VL_set,cy);
    VR_set_set=vector_update(VR_set_set,VR_set,cy);


    ########################################


    E_total=0;
    E_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*1.0;

    for cx=1:Lx-1
        for cy=1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            # iPEPS_2x2=psi[x_range,y_range];
            # rho_plaquatte,U_s_s,trun_history=build_density_matrix_2x2_new(psi_double_open, psi_double,x_range,y_range, chi, multiplet_tol);
            rho_plaquatte,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,nothing, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
            E=@tensor rho_plaquatte[1,2,3,4]*U_s_s[5,5,1]*U_s_s[6,6,2]*U_s_s[7,7,3]*U_s_s[8,8,4];
            log_coe=0;
            if cy>=3
                log_coe=log_coe+sum(log_coe_bot_set[2:cy-1]);
            end
            if cy+2<=Ly-1
                log_coe=log_coe+sum(log_coe_top_set[cy+2:Ly-1]);
            end
            E=E*exp(log_coe);
            @ignore_derivatives E_set[cx,cy]=E;
            E_total=E_total+E;
        end
    end

    return E_total,E_set
end

