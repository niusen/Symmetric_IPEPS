
function overlap(psi1,psi2,px,py)
    Lx0,Ly0=size(psi1);#original cluster size without adding trivial boundary
    global chi, multiplet_tol
    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end

    global n_mps_sweep
    #disable sweep, other wise norm_coe is incorrect
    n_mps_sweep=0;

    
    Lx=Lx0+2;
    Ly=Ly0+2;

    psi_PEPS1=iPESS_to_iPEPS_matrix(psi1);
    psi_PEPS2=iPESS_to_iPEPS_matrix(psi2);
    psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi_PEPS1,psi_PEPS2,Lx0,Ly0);




    t1=parameters["t1"];
    t2=parameters["t2"];
    U=parameters["U"];
    μ=parameters["μ"];
    ϕ=parameters["ϕ"];



    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """
    

    ########################################
    #construct top and bot environment
    norm_coe_set=Vector{ComplexF64}(undef,Ly)
    trun_history=[];
    mps_bot_set=initial_tuple(Ly);
    mps_top_set=initial_tuple(Ly);

    mps_bot=(psi_double[:,1]...,);
    mps_bot_set=vector_update(mps_bot_set,mps_bot,1);
    mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    norm_coe_set[1]=1;
    trun_history=vcat(trun_history,trun_errs);
    for cy=2:Ly-2
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs,norm_coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
        mps_bot_set=vector_update(mps_bot_set,mps_bot,cy);
        trun_history=vcat(trun_history,trun_errs);
        norm_coe_set[cy]=norm_coe;
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
    norm_coe_set[Ly]=1;
    for cy=Ly-1:-1:3
        mpo=pi_rotate_mpo((psi_double[:,cy]...,));
        mps_top,trun_errs,norm_coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
        mps_top_set=vector_update(mps_top_set,treat_mps_top(mps_top),cy);
        trun_history=vcat(trun_history,trun_errs);
        norm_coe_set[cy]=norm_coe;
    end
    #global trun_history
    #println(trun_history)
    println(norm_coe_set)
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
    Ex_set=@ignore_derivatives zeros(Lx0-1,Ly0)*im*0;
    Ey_set=@ignore_derivatives zeros(Lx0,Ly0-1)*im*0;
    E_ld_ru_set=@ignore_derivatives zeros(Lx0-1,Ly0-1)*im*0;
    EU_set=@ignore_derivatives zeros(Lx0,Ly0)*im*0;
    occu_set=@ignore_derivatives zeros(Lx0,Ly0)*im*0;

    #(Lx-1)x(Ly-1) triangles

    if (px in (1:Lx0-1)) && (py in 1:Ly0-1)
        cx=px;
        cy=py;

        x_range=[cx,cx+1];
        y_range=[cy,cy+1];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+t1*(ex*im-ex'*im)  +t1*(ey+ey')*((-1)^(cx-1))  +t2*(e_ld_ru+e_ld_ru')*((-1)^(cx-1))  +U*eU;

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives Ex_set[cx,cy]=ex;
        # @ignore_derivatives Ey_set[cx+1,cy]=ey;
        # @ignore_derivatives E_ld_ru_set[cx,cy]=e_ld_ru;
        # @ignore_derivatives EU_set[cx+1,cy]=eU;
        # @ignore_derivatives occu_set[cx+1,cy]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)
        # println("aaa")

    elseif (px in (1:Lx0-1)) && (py==Ly0)
        cx=px;
        cy=py;
        
        x_range=[cx,cx+1];
        y_range=[cy-1,cy];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+t1*(ex*im-ex'*im)  +U*eU;

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives Ex_set[cx,cy+1]=ex;
        # @ignore_derivatives EU_set[cx+1,cy+1]=eU;
        # @ignore_derivatives occu_set[cx+1,cy+1]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)

    elseif (px==0) && (py in 1:Ly0-1)
        cx=px;
        cy=py;

        x_range=[cx+1,cx+2];
        y_range=[cy,cy+1];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+t1*(ey+ey')*((-1)^(cx-1-1))  +U*eU;

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives Ey_set[cx,cy]=ey;
        # @ignore_derivatives EU_set[cx,cy]=eU;
        # @ignore_derivatives occu_set[cx,cy]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)

    elseif (px==0) && (py==Ly0)
        cx=px;
        cy=py;
        x_range=[cx+1,cx+2];
        y_range=[cy-1,cy];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+U*eU;

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives EU_set[cx,cy+1]=eU;
        # @ignore_derivatives occu_set[cx,cy+1]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)

    else
        error("unknown case")
    end

    return norm_coe*Norm
end




function energy_disk_old(psi,psi_double)
    Lx0,Ly0=size(psi);#original cluster size without adding trivial boundary
    global chi, multiplet_tol
    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end

    Lx=Lx0+2;
    Ly=Ly0+2;
    t1=parameters["t1"];
    t2=parameters["t2"];
    U=parameters["U"];
    μ=parameters["μ"];
    ϕ=parameters["ϕ"];



    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """
    

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
    Ex_set=@ignore_derivatives zeros(Lx0-1,Ly0)*im*0;
    Ey_set=@ignore_derivatives zeros(Lx0,Ly0-1)*im*0;
    E_ld_ru_set=@ignore_derivatives zeros(Lx0-1,Ly0-1)*im*0;
    EU_set=@ignore_derivatives zeros(Lx0,Ly0)*im*0;
    occu_set=@ignore_derivatives zeros(Lx0,Ly0)*im*0;

    #(Lx-1)x(Ly-1) triangles

    for cx=1:Lx0-1
        for cy=1:Ly0-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
            ex,ey,e_ld_ru,occu,eU=compute_ob_2x2_triangle(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);
            E_total=E_total+t1*(ex*im-ex'*im)  +t1*(ey+ey')*((-1)^(cx-1))  +t2*(e_ld_ru+e_ld_ru')*((-1)^(cx-1))  +U*eU;

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives Ex_set[cx,cy]=ex;
            @ignore_derivatives Ey_set[cx+1,cy]=ey;
            @ignore_derivatives E_ld_ru_set[cx,cy]=e_ld_ru;
            @ignore_derivatives EU_set[cx+1,cy]=eU;
            @ignore_derivatives occu_set[cx+1,cy]=occu;

            # E_total=E_total+E;
        end
    end

    for cx=1:Lx0-1
        for cy=Ly0:Ly0
            x_range=[cx,cx+1];
            y_range=[cy-1,cy];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
            ex,occu,eU=compute_ob_2x2_top(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

            E_total=E_total+t1*(ex*im-ex'*im)  +U*eU;

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives Ex_set[cx,cy]=ex;
            @ignore_derivatives EU_set[cx+1,cy]=eU;
            @ignore_derivatives occu_set[cx+1,cy]=occu;

            # E_total=E_total+E;
        end
    end

    for cx=0:0
        for cy=1:Ly0-1
            x_range=[cx+1,cx+2];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
            ey,occu,eU=compute_ob_2x2_left(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

            E_total=E_total+t1*(ey+ey')*((-1)^(cx-1))  +U*eU;

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives Ey_set[cx+1,cy]=ey;
            @ignore_derivatives EU_set[cx+1,cy]=eU;
            @ignore_derivatives occu_set[cx+1,cy]=occu;

            # E_total=E_total+E;
        end
    end

    for cx=0:0
        for cy=Ly0:Ly0
            x_range=[cx+1,cx+2];
            y_range=[cy-1,cy];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
            occu,eU=compute_ob_2x2_left_top(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

            E_total=E_total+U*eU;

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives EU_set[cx+1,cy]=eU;
            @ignore_derivatives occu_set[cx+1,cy]=occu;

            # E_total=E_total+E;
        end
    end

    return E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set
end






