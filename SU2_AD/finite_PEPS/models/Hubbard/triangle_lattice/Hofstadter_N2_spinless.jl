
function overlap(psi1::Matrix{Triangle_iPESS},psi2::Matrix{Triangle_iPESS},ppx,ppy)
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
    occu_set=@ignore_derivatives zeros(Lx0,Ly0)*im*0;

    #(Lx-1)x(Ly-1) triangles

    if (ppx in (1:Lx0-1)) && (ppy in 1:Ly0-1)
        cx=ppx;
        cy=ppy;

        x_range=[cx,cx+1];
        y_range=[cy,cy+1];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+t1*(ex*im-ex'*im)  +t1*(ey+ey')*((-1)^(cx-1))  +t2*(e_ld_ru+e_ld_ru')*((-1)^(cx-1));

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives Ex_set[cx,cy]=ex;
        # @ignore_derivatives Ey_set[cx+1,cy]=ey;
        # @ignore_derivatives E_ld_ru_set[cx,cy]=e_ld_ru;
        # @ignore_derivatives occu_set[cx+1,cy]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)
        # println("aaa")

    elseif (ppx in (1:Lx0-1)) && (ppy==Ly0)
        cx=ppx;
        cy=ppy;
        
        x_range=[cx,cx+1];
        y_range=[cy-1,cy];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+t1*(ex*im-ex'*im);

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives Ex_set[cx,cy+1]=ex;
        # @ignore_derivatives occu_set[cx+1,cy+1]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)

    elseif (ppx==0) && (ppy in 1:Ly0-1)
        cx=ppx;
        cy=ppy;

        x_range=[cx+1,cx+2];
        y_range=[cy,cy+1];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+t1*(ey+ey')*((-1)^(cx-1-1));

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives Ey_set[cx,cy]=ey;
        # @ignore_derivatives occu_set[cx,cy]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)

    elseif (ppx==0) && (ppy==Ly0)
        cx=ppx;
        cy=ppy;
        x_range=[cx+1,cx+2];
        y_range=[cy-1,cy];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total;

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
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



function overlap(psi1::Matrix{TensorMap},ppx,ppy)
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


    psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi1,Lx0,Ly0);




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
    occu_set=@ignore_derivatives zeros(Lx0,Ly0)*im*0;

    #(Lx-1)x(Ly-1) triangles

    if (ppx in (1:Lx0-1)) && (ppy in 1:Ly0-1)
        cx=ppx;
        cy=ppy;

        x_range=[cx,cx+1];
        y_range=[cy,cy+1];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+t1*(ex*im-ex'*im)  +t1*(ey+ey')*((-1)^(cx-1))  +t2*(e_ld_ru+e_ld_ru')*((-1)^(cx-1));

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives Ex_set[cx,cy]=ex;
        # @ignore_derivatives Ey_set[cx+1,cy]=ey;
        # @ignore_derivatives E_ld_ru_set[cx,cy]=e_ld_ru;
        # @ignore_derivatives occu_set[cx+1,cy]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)
        # println("aaa")

    elseif (ppx in (1:Lx0-1)) && (ppy==Ly0)
        cx=ppx;
        cy=ppy;
        
        x_range=[cx,cx+1];
        y_range=[cy-1,cy];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+t1*(ex*im-ex'*im);

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives Ex_set[cx,cy+1]=ex;
        # @ignore_derivatives occu_set[cx+1,cy+1]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)

    elseif (ppx==0) && (ppy in 1:Ly0-1)
        cx=ppx;
        cy=ppy;

        x_range=[cx+1,cx+2];
        y_range=[cy,cy+1];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total+t1*(ey+ey')*((-1)^(cx-1-1));

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
        # @ignore_derivatives Ey_set[cx,cy]=ey;
        # @ignore_derivatives occu_set[cx,cy]=occu;

        # E_total=E_total+E;
        norm_coe=prod(norm_coe_set[1:y_range[1]-1+1])*prod(norm_coe_set[y_range[2]+1+1:Ly]);
        # println(norm_coe_set[1:y_range[1]-1+1])
        # println(norm_coe_set[y_range[2]+1+1:Ly])
        # println(norm_coe)

    elseif (ppx==0) && (ppy==Ly0)
        cx=ppx;
        cy=ppy;
        x_range=[cx+1,cx+2];
        y_range=[cy-1,cy];

        iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
        Norm=norm_ob_2x2(mps_bot_set,mps_top_set,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

        # E_total=E_total;

        #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
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



function energy_disk_global(psi::Matrix,psi_double::Matrix)
    #avoid using trivial boundary tensors.
    #remove trivial boundary legs
    Lx,Ly=size(psi);#original cluster size without adding trivial boundary


    psi_double=construct_double_layer_swap_sites_new(psi,psi_double,Lx,Ly);



    global chi, multiplet_tol
    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end


    t1=parameters["t1"];
    t2=parameters["t2"];
    V=parameters["V"];
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
    Ex_set=@ignore_derivatives zeros(Lx-1,Ly)*im*0;
    Ey_set=@ignore_derivatives zeros(Lx,Ly-1)*im*0;
    E_ld_ru_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*0;
    NNx_set=@ignore_derivatives zeros(Lx-1,Ly)*im*0;
    NNy_set=@ignore_derivatives zeros(Lx,Ly-1)*im*0;
    NN_ld_ru_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*0;
    occu_set=@ignore_derivatives zeros(Lx,Ly)*im*0;

    #(Lx-1)x(Ly-1) triangles

    for cx=1:Lx-1
        for cy=1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range,y_range];
            ex,ey,e_ld_ru, nnx,nny,nn_ld_ru, occu=compute_ob_2x2_triangle_new(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
            E_total=E_total+t1*(ex*im-ex'*im)  +t1*(ey+ey')*((-1)^(cx-1))  +t2*(e_ld_ru+e_ld_ru')*((-1)^(cx-1)) +V*(nnx+nny+nn_ld_ru);


            @ignore_derivatives Ex_set[cx,cy]=ex;
            @ignore_derivatives Ey_set[cx+1,cy]=ey;
            @ignore_derivatives E_ld_ru_set[cx,cy]=e_ld_ru;
            @ignore_derivatives NNx_set[cx,cy]=nnx;
            @ignore_derivatives NNy_set[cx+1,cy]=nny;
            @ignore_derivatives NN_ld_ru_set[cx,cy]=nn_ld_ru;
            @ignore_derivatives occu_set[cx+1,cy]=occu;


        end
    end

    for cx=1:Lx-1
        for cy=Ly:Ly
            x_range=[cx,cx+1];
            y_range=[cy-1,cy];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range,y_range];
            ex,nnx,occu=compute_ob_2x2_top_new(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);

            E_total=E_total+t1*(ex*im-ex'*im) +V*nnx;


            @ignore_derivatives Ex_set[cx,cy]=ex;
            @ignore_derivatives NNx_set[cx,cy]=nnx;
            @ignore_derivatives occu_set[cx+1,cy]=occu;


        end
    end

    for cx=0:0
        for cy=1:Ly-1
            x_range=[cx+1,cx+2];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range,y_range];
            ey,nny,occu=compute_ob_2x2_left_new(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);

            E_total=E_total+t1*(ey+ey')*((-1)^(cx-1))+V*nny;


            @ignore_derivatives Ey_set[cx+1,cy]=ey;
            @ignore_derivatives NNy_set[cx+1,cy]=nny;
            @ignore_derivatives occu_set[cx+1,cy]=occu;


        end
    end

    for cx=0:0
        for cy=Ly:Ly
            x_range=[cx+1,cx+2];
            y_range=[cy-1,cy];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range,y_range];
            occu=compute_ob_2x2_left_top_new(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);

            E_total=E_total;

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives occu_set[cx+1,cy]=occu;

            # E_total=E_total+E;
        end
    end

    return E_total,Ex_set,Ey_set,E_ld_ru_set, NNx_set,NNy_set,NN_ld_ru_set, occu_set
end



function energy_disk_local(x,psi,psi_double,ppx,ppy)
    #avoid using trivial boundary tensors.
    #remove trivial boundary legs
    Lx,Ly=size(psi);#original cluster size without adding trivial boundary

    psi=matrix_update(psi,ppx,ppy,x);
    psi_double=construct_double_layer_swap_position(psi,psi_double,ppx,ppy,Lx,Ly);



    global chi, multiplet_tol
    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end


    t1=parameters["t1"];
    t2=parameters["t2"];
    V=parameters["V"];
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
    Ex_set=@ignore_derivatives zeros(Lx-1,Ly)*im*0;
    Ey_set=@ignore_derivatives zeros(Lx,Ly-1)*im*0;
    E_ld_ru_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*0;
    NNx_set=@ignore_derivatives zeros(Lx-1,Ly)*im*0;
    NNy_set=@ignore_derivatives zeros(Lx,Ly-1)*im*0;
    NN_ld_ru_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*0;
    occu_set=@ignore_derivatives zeros(Lx,Ly)*im*0;

    #(Lx-1)x(Ly-1) triangles

    for cx=1:Lx-1
        for cy=1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range,y_range];
            ex,ey,e_ld_ru, nnx,nny,nn_ld_ru, occu=compute_ob_2x2_triangle_new(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);
            E_total=E_total+t1*(ex*im-ex'*im)  +t1*(ey+ey')*((-1)^(cx-1))  +t2*(e_ld_ru+e_ld_ru')*((-1)^(cx-1)) +V*(nnx+nny+nn_ld_ru);


            @ignore_derivatives Ex_set[cx,cy]=ex;
            @ignore_derivatives Ey_set[cx+1,cy]=ey;
            @ignore_derivatives E_ld_ru_set[cx,cy]=e_ld_ru;
            @ignore_derivatives NNx_set[cx,cy]=nnx;
            @ignore_derivatives NNy_set[cx+1,cy]=nny;
            @ignore_derivatives NN_ld_ru_set[cx,cy]=nn_ld_ru;
            @ignore_derivatives occu_set[cx+1,cy]=occu;


        end
    end

    for cx=1:Lx-1
        for cy=Ly:Ly
            x_range=[cx,cx+1];
            y_range=[cy-1,cy];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range,y_range];
            ex,nnx, occu=compute_ob_2x2_top_new(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);

            E_total=E_total+t1*(ex*im-ex'*im) +V*nnx;


            @ignore_derivatives Ex_set[cx,cy]=ex;
            @ignore_derivatives NNx_set[cx,cy]=nnx;
            @ignore_derivatives occu_set[cx+1,cy]=occu;


        end
    end

    for cx=0:0
        for cy=1:Ly-1
            x_range=[cx+1,cx+2];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range,y_range];
            ey,nny, occu=compute_ob_2x2_left_new(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);

            E_total=E_total+t1*(ey+ey')*((-1)^(cx-1)) +V*nny;


            @ignore_derivatives Ey_set[cx+1,cy]=ey;
            @ignore_derivatives NNy_set[cx+1,cy]=nny;
            @ignore_derivatives occu_set[cx+1,cy]=occu;


        end
    end

    for cx=0:0
        for cy=Ly:Ly
            x_range=[cx+1,cx+2];
            y_range=[cy-1,cy];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range,y_range];
            occu=compute_ob_2x2_left_top_new(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range,y_range,Lx,Ly);

            E_total=E_total;

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives occu_set[cx+1,cy]=occu;

            # E_total=E_total+E;
        end
    end

    return E_total,Ex_set,Ey_set,E_ld_ru_set, NNx_set,NNy_set,NN_ld_ru_set, occu_set
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
    V=parameters["V"];
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
    # mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
    # trun_history=vcat(trun_history,trun_errs);
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
    # mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    # trun_history=vcat(trun_history,trun_errs);
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
    NNx_set=@ignore_derivatives zeros(Lx-1,Ly)*im*0;
    NNy_set=@ignore_derivatives zeros(Lx,Ly-1)*im*0;
    NN_ld_ru_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*0;
    occu_set=@ignore_derivatives zeros(Lx0,Ly0)*im*0;

    #(Lx-1)x(Ly-1) triangles

    for cx=1:Lx0-1
        for cy=1:Ly0-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            iPEPS_2x2=psi[x_range,y_range];
            iPEPS_double_2x2=psi_double[x_range.+1,y_range.+1];
            ex,ey,e_ld_ru,nnx,nny,nn_ld_ru,occu=compute_ob_2x2_triangle(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);
            E_total=E_total+t1*(ex*im-ex'*im)  +t1*(ey+ey')*((-1)^(cx-1))  +t2*(e_ld_ru+e_ld_ru')*((-1)^(cx-1)) +V*(nnx+nny+nn_ld_ru);

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives Ex_set[cx,cy]=ex;
            @ignore_derivatives Ey_set[cx+1,cy]=ey;
            @ignore_derivatives E_ld_ru_set[cx,cy]=e_ld_ru;
            @ignore_derivatives NNx_set[cx,cy]=nnx;
            @ignore_derivatives NNy_set[cx+1,cy]=nny;
            @ignore_derivatives NN_ld_ru_set[cx,cy]=nn_ld_ru;
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
            ex,nnx,occu=compute_ob_2x2_top(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

            E_total=E_total+t1*(ex*im-ex'*im) +V*nnx;

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives Ex_set[cx,cy]=ex;
            @ignore_derivatives NNx_set[cx,cy]=nnx;
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
            ey,nny,occu=compute_ob_2x2_left(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

            E_total=E_total+t1*(ey+ey')*((-1)^(cx-1)) +V*nny;

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives Ey_set[cx+1,cy]=ey;
            @ignore_derivatives NNy_set[cx+1,cy]=nny;
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
            occu=compute_ob_2x2_left_top(mps_bot_set,mps_top_set,iPEPS_2x2,iPEPS_double_2x2, VL_set_set,VR_set_set, x_range.+1,y_range.+1);

            E_total=E_total;

            #E=H_plaquatte(J1,J2,Jchi,H_Heisenberg, H123chiral, x_range,y_range,Lx,Ly,rho_plaquatte);
            @ignore_derivatives occu_set[cx+1,cy]=occu;

            # E_total=E_total+E;
        end
    end

    return E_total,Ex_set,Ey_set,E_ld_ru_set, NNx_set,NNy_set,NN_ld_ru_set, occu_set
end




function cost_fun_global(x::Matrix{Triangle_iPESS})
    
    global psi_double_env,PEPS_init
    x_PEPS=PESS_to_PEPS_matrix(x,PEPS_init);
    x_double=construct_double_layer_swap_sites(x_PEPS,psi_double_env,Lx,Ly);


    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set=energy_disk_old(x_PEPS,x_double);
    return real(E_total)
end

function cost_fun_local(x::Triangle_iPESS)
    
    global psi_double,PEPS_init,ppx,ppy

    x_PEPS=matrix_update(PEPS_init,ppx,ppy,iPESS_to_iPEPS_tensor(x.Bm,x.Tm));
    x_double=construct_double_layer_swap_sites(x_PEPS,psi_double,Lx,Ly);


    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set=energy_disk_old(x_PEPS,x_double);
    return real(E_total)
end



function cost_fun_local_Bm(x::TensorMap)
    
    global psi_double_env,PEPS_init,psi_init,ppx,ppy

    
    Tm=psi_init[ppx,ppy].Tm;
    Tm=permute(Tm,(1,2,),(3,));
    Bm=permute(x,(1,),(2,3,4,));
    T=permute(Tm*Bm,(1,5,4,2,3,));#L,D,R,U,d,

    x_PEPS=matrix_update(PEPS_init,ppx,ppy,T);
    x_double=construct_double_layer_swap_sites(x_PEPS,psi_double_env,Lx,Ly);


    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set=energy_disk_old(x_PEPS,x_double);
    # E_total=norm(x_double)
    return real(E_total)
end

function cost_fun_local_Tm(x::TensorMap)
    
    global psi_double_env,PEPS_init,psi_init,ppx,ppy

    Tm=permute(x,(1,2,),(3,));
    Bm=permute(psi_init[ppx,ppy].Bm,(1,),(2,3,4,));
    T=permute(Tm*Bm,(1,5,4,2,3,));#L,D,R,U,d,

    x_PEPS=matrix_update(PEPS_init,ppx,ppy,T);
    x_double=construct_double_layer_swap_sites(x_PEPS,psi_double_env,Lx,Ly);


    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set=energy_disk_old(x_PEPS,x_double);
    # E_total=norm(x_double)
    return real(E_total)
end

function cost_fun_local(x::TensorMap)    
    global psi_double,psi,ppx,ppy

    # E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set=energy_disk_test(x_PEPS,x_double);
    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set=energy_disk_local(x,psi,psi_double,ppx,ppy);
    # println(Ex_set)
    # println(Ey_set)
    # println(E_ld_ru_set)
    # println(occu_set)
    #E_total=norm(x_PEPS)
    global E_tem;
    E_tem=E_total;
    #return real(E_total),Ex_set,Ey_set,E_ld_ru_set,occu_set
    return real(E_total)
end

function cost_fun_global(x::Matrix{TensorMap})    
    global psi_double
    E_total,Ex_set,Ey_set,E_ld_ru_set,occu_set=energy_disk_global(x,psi_double);

    global E_tem;
    E_tem=E_total;

    return real(E_total)
end





function finite_diff2(x::Triangle_iPESS)
    function fd_(state_vec::TensorMap,cfun)

        dt=0.000001

        E0=cfun(state_vec);

        grad=similar(state_vec)*0;

            for n_block in eachindex(state_vec.data.values)
                for elem in eachindex(state_vec.data.values[n_block])
                    state_vec_tem=deepcopy(state_vec);
                    T=state_vec_tem.data.values[n_block];
                    T[elem]=T[elem]+dt;
                    state_vec_tem.data.values[n_block]=T;
                    real_part=(cfun(state_vec_tem)-E0)/dt;

                    state_vec_tem=deepcopy(state_vec);
                    T=state_vec_tem.data.values[n_block];
                    T[elem]=T[elem]+dt*im;
                    state_vec_tem.data.values[n_block]=T;
                    imag_part=(cfun(state_vec_tem)-E0)/dt;

                    grad.data.values[n_block][elem]=real_part+im*imag_part;
                end
            end
        return grad
    end
    
    grad_Bm=fd_(x.Bm,cost_fun_local_Bm);
    grad_Tm=fd_(x.Tm,cost_fun_local_Tm);
    return grad_Bm,grad_Tm
end

function finite_diff3(state_vec::TensorMap,cfun)

    dt=0.000001

    E0=cfun(state_vec);

    grad=similar(state_vec)*0;

        for n_block in eachindex(state_vec.data.values)
            for elem in eachindex(state_vec.data.values[n_block])
                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem.data.values[n_block];
                T[elem]=T[elem]+dt;
                state_vec_tem.data.values[n_block]=T;
                real_part=(cfun(state_vec_tem)-E0)/dt;

                state_vec_tem=deepcopy(state_vec);
                T=state_vec_tem.data.values[n_block];
                T[elem]=T[elem]+dt*im;
                state_vec_tem.data.values[n_block]=T;
                imag_part=(cfun(state_vec_tem)-E0)/dt;

                grad.data.values[n_block][elem]=real_part+im*imag_part;
            end
        end
    return grad
end

function finite_diff_local(x::Triangle_iPESS)
    dt=0.00001;
    E0=cost_fun_local(x::Triangle_iPESS);

    Bm=x.Bm;
    Tm=x.Tm;
    Bm_grad=deepcopy(Bm)*0;
    Tm_grad=deepcopy(Tm)*0;
    for cm =1:length(Bm_grad.data.values)
        mm0=Bm.data.values[cm];
        for cp in eachindex(mm0)
            Bm_tem=deepcopy(Bm);
            mm=deepcopy(mm0);
            mm[cp]=mm[cp]+dt;
            Bm_tem.data.values[cm]=mm;
            E=cost_fun_local(Triangle_iPESS(Bm_tem,Tm));
            Bm_grad.data.values[cm][cp]=Bm_grad.data.values[cm][cp]+(E-E0)/dt;


            Bm_tem=deepcopy(Bm);
            mm=deepcopy(mm0);
            mm[cp]=mm[cp]+dt*im;
            Bm_tem.data.values[cm]=mm;
            E=cost_fun_local(Triangle_iPESS(Bm_tem,Tm));
            Bm_grad.data.values[cm][cp]=Bm_grad.data.values[cm][cp]+(E-E0)/dt*im;
        end
    end

    for cm =1:length(Tm_grad.data.values)
        mm0=Tm.data.values[cm];
        for cp in eachindex(mm0)
            Tm_tem=deepcopy(Tm);
            mm=deepcopy(mm0);
            mm[cp]=mm[cp]+dt;
            Tm_tem.data.values[cm]=mm;
            E=cost_fun_local(Triangle_iPESS(Bm,Tm_tem));
            Tm_grad.data.values[cm][cp]=Tm_grad.data.values[cm][cp]+(E-E0)/dt;


            Tm_tem=deepcopy(Tm);
            mm=deepcopy(mm0);
            mm[cp]=mm[cp]+dt*im;
            Tm_tem.data.values[cm]=mm;
            E=cost_fun_local(Triangle_iPESS(Bm,Tm_tem));
            Tm_grad.data.values[cm][cp]=Tm_grad.data.values[cm][cp]+(E-E0)/dt*im;
        end
    end

    return Triangle_iPESS(Bm_grad,Tm_grad)
end