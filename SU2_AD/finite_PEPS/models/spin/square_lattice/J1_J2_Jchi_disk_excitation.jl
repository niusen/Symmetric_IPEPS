
function projector_to_1(V,spin)
    #projector to a one dimension space
    V1=deepcopy(V);
    for cc=1:length(V1.dims.values)
        if V1.dims.keys[cc].j==spin 
            V1=Rep[SU₂](spin=>1);
            break;
        end
    end
    if V.dual
        V1=V1';
    end
    T=TensorMap(randn,V1,V);
    for cc=1:length(T.data.values)
        mm=T.data.values[cc];
        mm=Matrix(I, size(mm,1), size(mm,2));
        T.data.values[cc]=mm;
    end
    return T
end


function construct_environment(Lx,Ly,psi_double)

    

    global mpo_mps_trun_method, left_right_env_method;
    if mpo_mps_trun_method=="canonical"
        mpo_mps_fun=truncate_mpo_mps;
    elseif mpo_mps_trun_method=="simple_middle"
        mpo_mps_fun=simple_truncate_to_moddle;
    end

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
    return mps_bot_set,mps_top_set,log_coe_bot_set,log_coe_top_set, VL_set_set,VR_set_set, trun_history
end



function cost_fun_bra_ket(N_operator_sites,A_bra,A_ket,pos_bra,pos_ket,psi_bra,psi_ket, psi_double_open,psi_double,U_s_s,funtype,real_imag) #variational parameters are vector of TensorMap
    global chi, parameters
    if funtype=="energy"
        E,E_set=energy_disk_bra_ket(N_operator_sites,A_bra,A_ket,pos_bra,pos_ket,psi_bra,psi_ket, psi_double_open,psi_double,U_s_s);
    elseif funtype=="norm"
        E,E_set=norm_disk_bra_ket(N_operator_sites,A_bra,A_ket,pos_bra,pos_ket,psi_bra,psi_ket, psi_double_open,psi_double,U_s_s);
    end

    if real_imag=="real"
        E=real(E);
    elseif real_imag=="imag"
        E=imag(E);
    elseif real_imag=="full"
        E=E;
    end
    global E_tem;
    E_tem=E;
    # return E,E_set
    return E
end



function update_bra_ket(N_operator_sites,A_bra,A_ket,psi_bra,psi_ket,pos_bra,pos_ket)
    if N_operator_sites==1 #basis are single-site spin operators acting on ground state
        psi_bra=matrix_update(psi_bra,pos_bra[1],pos_bra[2],A_bra);
        psi_ket=matrix_update(psi_ket,pos_ket[1],pos_ket[2],A_ket);
    elseif N_operator_sites==2 #basis are two-site spin operators acting on ground state
        psi_bra=matrix_update(psi_bra,pos_bra[1][1],pos_bra[1][2],A_bra[1]);
        psi_bra=matrix_update(psi_bra,pos_bra[2][1],pos_bra[2][2],A_bra[2]);
        psi_ket=matrix_update(psi_ket,pos_ket[1][1],pos_ket[1][2],A_ket[1]);
        psi_ket=matrix_update(psi_ket,pos_ket[2][1],pos_ket[2][2],A_ket[2]);
    end
    return psi_bra,psi_ket
end

function update_bra_ket_double(N_operator_sites,psi_bra,psi_ket,pos_bra,pos_ket,psi_double_open,psi_double)
    function single_site_update(psibra,psiket,pos,psidoubleopen,psidouble)
        global U_s_s
        Lx,Ly=size(psibra);
        AA_open=build_double_layer_open_position(psibra[pos[1],pos[2]],psiket[pos[1],pos[2]],pos[1],pos[2],Lx,Ly,false);
        psidoubleopen=matrix_update(psidoubleopen,pos[1],pos[2],AA_open);
        AA=contract_physical(AA_open, U_s_s);
        psidouble=matrix_update(psidouble,pos[1],pos[2],AA);
        return psidoubleopen,psidouble
    end

    if N_operator_sites==1 #basis are single-site spin operators acting on ground state
        pos_set=(-1,);

        pos=pos_bra;
        if !(pos in pos_set)
            psi_double_open,psi_double=single_site_update(psi_bra,psi_ket,pos,psi_double_open,psi_double);
        end
        pos_set=(pos_set...,pos,);

        pos=pos_ket;
        if !(pos in pos_set)
            psi_double_open,psi_double=single_site_update(psi_bra,psi_ket,pos,psi_double_open,psi_double);
        end
        pos_set=(pos_set...,pos,);
    elseif N_operator_sites==2 #basis are two-site spin operators acting on ground state
        pos_set=(-1,);

        pos=pos_bra[1];
        if !(pos in pos_set)
            psi_double_open,psi_double=single_site_update(psi_bra,psi_ket,pos,psi_double_open,psi_double);
        end
        pos_set=(pos_set...,pos,);

        pos=pos_bra[2];
        if !(pos in pos_set)
            psi_double_open,psi_double=single_site_update(psi_bra,psi_ket,pos,psi_double_open,psi_double);
        end
        pos_set=(pos_set...,pos,);

        pos=pos_ket[1];
        if !(pos in pos_set)
            psi_double_open,psi_double=single_site_update(psi_bra,psi_ket,pos,psi_double_open,psi_double);
        end
        pos_set=(pos_set...,pos,);

        pos=pos_ket[2];
        if !(pos in pos_set)
            psi_double_open,psi_double=single_site_update(psi_bra,psi_ket,pos,psi_double_open,psi_double);
        end
        pos_set=(pos_set...,pos,);

    end
    return psi_double_open,psi_double
end

function energy_disk_bra_ket(N_operator_sites,A_bra,A_ket,pos_bra,pos_ket,psi_bra,psi_ket,psi_double_open,psi_double,U_s_s)
    global chi, multiplet_tol


    Lx=size(psi_bra,1);
    Ly=size(psi_bra,2);
    global parameters
    J1=parameters["J1"];
    J2=parameters["J2"];
    Jchi=parameters["Jchi"];

    H_Heisenberg, H123chiral, H12, H31, H23=@ignore_derivatives Hamiltonians_spin_half("SU2");


    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """
    Lx=size(psi_bra,1);
    Ly=size(psi_bra,2);
    ########################################
    psi_bra,psi_ket=update_bra_ket(N_operator_sites,A_bra,A_ket,psi_bra,psi_ket,pos_bra,pos_ket);
    ########################################
    #remove boundary trivial legs
    psi_ket=remove_trivial_boundary_leg(psi_ket);
    psi_bra=remove_trivial_boundary_leg(psi_bra);
    ########################################
    psi_double_open,psi_double=update_bra_ket_double(N_operator_sites,psi_bra,psi_ket,pos_bra,pos_ket,psi_double_open,psi_double);
    ########################################
    #construct top and bot environment
    mps_bot_set,mps_top_set,log_coe_bot_set,log_coe_top_set, VL_set_set,VR_set_set, trun_history=construct_environment(Lx,Ly,psi_double);
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



function norm_disk_bra_ket(N_operator_sites,A_bra,A_ket,pos_bra,pos_ket,psi_bra,psi_ket,psi_double_open,psi_double,U_s_s)
    global chi, multiplet_tol

    Lx=size(psi_bra,1);
    Ly=size(psi_bra,2);

    """coordinate
        (1,2),(2,2)
        (1,1),(2,1)
    """


    ########################################
    psi_bra,psi_ket=update_bra_ket(N_operator_sites,A_bra,A_ket,psi_bra,psi_ket,pos_bra,pos_ket);
    ########################################
    #remove boundary trivial legs
    psi_ket=remove_trivial_boundary_leg(psi_ket);
    psi_bra=remove_trivial_boundary_leg(psi_bra);
    ########################################
    psi_double_open,psi_double=update_bra_ket_double(N_operator_sites,psi_bra,psi_ket,pos_bra,pos_ket,psi_double_open,psi_double);
    ########################################
    #construct top and bot environment
    mps_bot_set,mps_top_set,log_coe_bot_set,log_coe_top_set, VL_set_set,VR_set_set, trun_history=construct_environment(Lx,Ly,psi_double);
    ########################################


    E_total=0;
    E_set=@ignore_derivatives zeros(Lx-1,Ly-1)*im*1.0;

    for cx=1:Lx-1
        for cy=1:Ly-1
            x_range=[cx,cx+1];
            y_range=[cy,cy+1];
            #iPEPS_2x2=psi[x_range,y_range];
            rho_plaquatte,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,nothing, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol,psi_double_open);
            # rho_plaquatte,U_s_s=build_density_matrix_2x2_new(mps_bot_set,mps_top_set,iPEPS_2x2, VL_set_set,VR_set_set, x_range,y_range, chi, multiplet_tol);
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


function compute_rotation_matrix_1site(psi,op)
    Lx,Ly=size(psi);
    R=zeros(Lx*Ly,Lx*Ly)*im;

    for ca=1:(Lx*Ly)
        psi_bra=deepcopy(psi);
        pos_bra=coordinate_1d_to_2d(Lx,Ly,ca);
        A_bra=psi[pos_bra[1],pos_bra[2]];
        @tensor A_bra[:]:=A_bra[-1,-2,-3,-4,1]*op[-5,1];
        psi_bra=matrix_update(psi_bra,pos_bra[1],pos_bra[2],A_bra);
        psi_bra=remove_trivial_boundary_leg(psi_bra);
        psi_bra=rotate_psi(psi_bra);
        
        for cb=1:(Lx*Ly)
            psi_ket=deepcopy(psi);
            pos_ket=coordinate_1d_to_2d(Lx,Ly,cb);
            A_ket=psi[pos_ket[1],pos_ket[2]];
            @tensor A_ket[:]:=A_ket[-1,-2,-3,-4,1]*op[-5,1];
            psi_ket=matrix_update(psi_ket,pos_ket[1],pos_ket[2],A_ket);
            psi_ket=remove_trivial_boundary_leg(psi_ket);
            
            
            R[ca,cb]=compute_ov(psi_bra,psi_ket);
        end
    end
    return R
end


function compute_rotation_matrix_2site(psi,op1,op2)
    Lx,Ly=size(psi);
    R=zeros((Lx*Ly),(Lx*Ly),(Lx*Ly),(Lx*Ly))*im;

    for ca1=1:(Lx*Ly)
        for ca2=1:(Lx*Ly)
            psi_bra=deepcopy(psi);
            pos_bra=(coordinate_1d_to_2d(Lx,Ly,ca1),coordinate_1d_to_2d(Lx,Ly,ca2),);
            A_bra1=psi[pos_bra[1][1],pos_bra[1][2]];
            A_bra2=psi[pos_bra[2][1],pos_bra[2][2]];
            @tensor A_bra1[:]:=A_bra1[-1,-2,-3,-4,1]*op1[-5,1];
            @tensor A_bra2[:]:=A_bra2[-1,-2,-3,-4,1]*op2[-5,1];

            psi_bra=matrix_update(psi_bra,pos_bra[1][1],pos_bra[1][2],A_bra1);
            psi_bra=matrix_update(psi_bra,pos_bra[2][1],pos_bra[2][2],A_bra2);
            psi_bra=remove_trivial_boundary_leg(psi_bra);
            psi_bra=rotate_psi(psi_bra);
            
            for cb1=1:(Lx*Ly)
                for cb2=1:(Lx*Ly)
                    psi_ket=deepcopy(psi);
                    pos_ket=(coordinate_1d_to_2d(Lx,Ly,cb1),coordinate_1d_to_2d(Lx,Ly,cb2),);

                    A_ket1=psi[pos_ket[1][1],pos_ket[1][2]];
                    A_ket2=psi[pos_ket[2][1],pos_ket[2][2]];
                    @tensor A_ket1[:]:=A_ket1[-1,-2,-3,-4,1]*op1[-5,1];
                    @tensor A_ket2[:]:=A_ket2[-1,-2,-3,-4,1]*op2[-5,1];
                    psi_ket=matrix_update(psi_ket,pos_ket[1][1],pos_ket[1][2],A_ket1);
                    psi_ket=matrix_update(psi_ket,pos_ket[2][1],pos_ket[2][2],A_ket2);
                    psi_ket=remove_trivial_boundary_leg(psi_ket);
                    R[ca1,ca2,cb1,cb2]=compute_ov(psi_bra,psi_ket);
                end                
            end
        end
    end
    return R
end