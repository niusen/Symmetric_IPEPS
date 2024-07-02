
function rotate_doublelayer(psi)
    psi=deepcopy(psi);
    Lx,Ly=size(psi);
    psi_rotated=Matrix{TensorMap}(undef,Ly,Lx);
    for cca=1:Lx
        for ccb=1:Ly
            T=psi[cca,ccb];
            if Rank(T)==2
                if ccb==Ly
                    T=permute(T,(1,2,));
                elseif ccb==1
                    T=permute(T,(2,1,));
                end
            elseif Rank(T)==3
                if ccb==1
                    T=permute(T,(3,1,2,));
                elseif ccb==Ly
                    T=permute(T,(1,2,3,));
                elseif cca==1
                    T=permute(T,(3,1,2,));
                elseif cca==Lx
                    T=permute(T,(3,1,2,));
                end
            elseif Rank(T)==4
                T=permute(T,(4,1,2,3,));
            end
            coord_new=coord_rotate([cca,ccb],Lx,Ly);
            psi_rotated[coord_new[1],coord_new[2]]=T;
        end
    end
    return psi_rotated
end

function rotate_psi(psi)
    psi=deepcopy(psi);
    Lx,Ly=size(psi);
    psi_rotated=Matrix{TensorMap}(undef,Ly,Lx);
    for cca=1:Lx
        for ccb=1:Ly
            T=psi[cca,ccb];
            if Rank(T)==2+1
                if ccb==Ly
                    T=permute(T,(1,2,3,));
                elseif ccb==1
                    T=permute(T,(2,1,3,));
                end
            elseif Rank(T)==3+1
                if ccb==1
                    T=permute(T,(3,1,2,4,));
                elseif ccb==Ly
                    T=permute(T,(1,2,3,4,));
                elseif cca==1
                    T=permute(T,(3,1,2,4,));
                elseif cca==Lx
                    T=permute(T,(3,1,2,4,));
                end
            elseif Rank(T)==4+1
                T=permute(T,(4,1,2,3,5,));
            end
            coord_new=coord_rotate([cca,ccb],Lx,Ly);
            psi_rotated[coord_new[1],coord_new[2]]=T;
        end
    end
    return psi_rotated
end

function coord_rotate(coord,Lx,Ly)
    return [Ly-coord[2]+1,coord[1]]
end

function env_2site_bra_ket(psi_left_bra,psi_left_ket,psi_left_double, mps_top_set,mps_bot_set, log_coe_top_set,log_coe_bot_set, px,py, envL_set,envR_set)
    Lx,Ly=size(psi_left_bra);
    psi_bra=deepcopy(psi_left_bra);
    psi_ket=deepcopy(psi_left_ket);

    @assert 1<=px<=Lx;
    @assert 1<=py<=Ly;
    if mod(px,1)==0.5
        bond_type="x";
    elseif mod(py,1)==0.5
        bond_type="y";
    else
        error("unknown bond");
    end
    if bond_type=="x"
        pos_T1=[px-0.5,py];
        pos_T2=[px+0.5,py];
        ind_T1=3;
        ind_T2=1;
    elseif bond_type=="y"
        pos_T1=[px,py+0.5];
        pos_T2=[px,py-0.5];
        ind_T1=2;
        ind_T2=4;
    end
    pos_T1=Int.(pos_T1);
    pos_T2=Int.(pos_T2);



    function update_psi(psi,pos_T1,pos_T2,bond_type)
        #add virtual phsyical leg
        T1=psi[pos_T1[1],pos_T1[2]];
        T2=psi[pos_T2[1],pos_T2[2]];

        T1=add_trivial_leg(T1);
        T2=add_trivial_leg(T2);
        psi[pos_T1[1],pos_T1[2]]=T1;
        psi[pos_T2[1],pos_T2[2]]=T2;



        if bond_type=="x"
            @assert pos_T1[2]==pos_T2[2];
            @assert (pos_T1[1]+1)==pos_T2[1];
            return psi
        # elseif bond_type=="y"
        #     @assert pos_T1[1]==pos_T2[1];
        #     @assert (pos_T1[2]-1)==pos_T2[2];
        #     psi_rotated=rotate_psi(psi);
        #     return psi_rotated
        end
    end

    if bond_type=="x"
        @assert pos_T1[2]==pos_T2[2];
        @assert (pos_T1[1]+1)==pos_T2[1];

        psi_bra=update_psi(psi_bra,pos_T1,pos_T2,bond_type);
        psi_ket=update_psi(psi_ket,pos_T1,pos_T2,bond_type);

        # psi_double=construct_double_layer(psi_bra,psi_ket);
        AA1,U_L1,U_D1,U_R1,U_U1=construct_double_layer_pos(psi_bra,psi_ket,pos_T1[1],pos_T1[2]);
        AA2,U_L2,U_D2,U_R2,U_U2=construct_double_layer_pos(psi_bra,psi_ket,pos_T2[1],pos_T2[2]);
        psi_left_double[pos_T1[1],pos_T1[2]]=AA1;
        psi_left_double[pos_T2[1],pos_T2[2]]=AA2;
        Norm,log_coe, envL_set,envR_set=get_2site_env_x_new(psi_left_double,pos_T1[1],pos_T2[1],pos_T1[2], mps_top_set,mps_bot_set, log_coe_top_set,log_coe_bot_set, envL_set,envR_set);
        # _,U_L1,U_D1,U_R1,U_U1=build_double_layer_open_position(psi[pos_T1[1],pos_T1[2]],pos_T1[1],pos_T1[2],Lx,Ly,true);
        # _,U_L2,U_D2,U_R2,U_U2=build_double_layer_open_position(psi[pos_T2[1],pos_T2[2]],pos_T2[1],pos_T2[2],Lx,Ly,true);
        

        U1=U_R1;
        U2=U_L2;

        @tensor Norm[:]:=Norm[1,2]*U1'[1,-1,-2]*U2'[-3,-4,2];#D1',D1, D2',D2
    # elseif bond_type=="y"
        # psi_rotated_bra=update_psi(psi_bra,pos_T1,pos_T2,bond_type);
        # psi_rotated_ket=update_psi(psi_ket,pos_T1,pos_T2,bond_type);

        # pos_T1_new=coord_rotate(pos_T1,Lx,Ly);
        # pos_T2_new=coord_rotate(pos_T2,Lx,Ly);
        # psi_double=construct_double_layer(psi_rotated_bra,psi_rotated_ket);
        # Norm,log_coe=get_2site_env_x_new(psi_double,pos_T1_new[1],pos_T2_new[1],pos_T1_new[2]);
        # # _,U_L1,U_D1,U_R1,U_U1=build_double_layer_open_position(psi_rotated[pos_T1_new[1],pos_T1_new[2]],pos_T1_new[1],pos_T1_new[2],Ly,Lx,true);
        # # _,U_L2,U_D2,U_R2,U_U2=build_double_layer_open_position(psi_rotated[pos_T2_new[1],pos_T2_new[2]],pos_T2_new[1],pos_T2_new[2],Ly,Lx,true);
        # _,U_L1,U_D1,U_R1,U_U1=construct_double_layer_pos(psi_rotated_bra,psi_rotated_ket,pos_T1_new[1],pos_T1_new[2]);
        # _,U_L2,U_D2,U_R2,U_U2=construct_double_layer_pos(psi_rotated_bra,psi_rotated_ket,pos_T2_new[1],pos_T2_new[2]);
        # U1=U_R1;
        # U2=U_L2;
        # @tensor Norm[:]:=Norm[1,2]*U1'[1,-1,-2]*U2'[-3,-4,2];#D1',D1, D2',D2

    end

    return Norm,log_coe,psi_left_double, envL_set,envR_set
end


function treat_mps_top(mps)
    #convert mps_top to normal order
    mps=mps[end:-1:1];
    for cx=2:Lx-1
        mps=mps_update(mps,permute(mps[cx],(2,1,3,)),cx);
    end
    return mps
end

function get_2site_env_x_new(psi_double,posxa,posxb,posy, mps_top_set,mps_bot_set, log_coe_top_set,log_coe_bot_set, envL_set,envR_set)
    Lx,Ly=size(psi_double);
    ########################################
    py=posy;
    pxa=posxa;
    pxb=posxb;

    if py==1
        log_coe=sum(log_coe_top_set[3:Ly-1]);
        # mps_up=mps_top;
        mps_up=mps_top_set[3]
        mpo=psi_double[:,py+1];
        mps_down=psi_double[:,py];
    elseif 1<py<Ly
        log_coe=sum(log_coe_top_set[py+1:Ly-1])+sum(log_coe_bot_set[2:py-1]);
        # mps_up=mps_top;
        mpo=psi_double[:,py];
        # mps_down=mps_bot;
        mps_up=mps_top_set[py+1];
        mps_down=mps_bot_set[py-1];
    elseif py==Ly
        log_coe=sum(log_coe_bot_set[2:Ly-2]);
        mps_up=psi_double[:,py];
        mps_up=treat_mps_top(pi_rotate_mps(mps_up));
        mpo=psi_double[:,py-1];
        # mps_down=mps_bot;
        mps_down=mps_bot_set[Ly-2];
    end

    if pxa>1
        envL_set,envR_set=LR_env_right_move(mps_up,mpo,mps_down, envL_set,envR_set, pxa,pxb);
    end

    if py==1
        if pxa==1
            @tensor Norm[:]:=mps_up[1][8,6]*mpo[1][7,9,6]*mps_down[1][-1,7]*mps_up[2][8,1,2]*mpo[2][9,4,3,2]*mps_down[2][-2,5,4]*envR_set[3][1,3,5];
        elseif 1<pxa<Lx-1
            @tensor Norm[:]:=envL_set[pxa-1][1,3,5]*mps_up[pxa][1,11,2]*mpo[pxa][3,4,12,2]*mps_down[pxa][5,-1,4]*mps_up[pxb][11,6,7]*mpo[pxb][12,9,8,7]*mps_down[pxb][-2,10,9]*envR_set[pxb+1][6,8,10];
        elseif pxa==Lx-1
            @tensor Norm[:]:=envL_set[Lx-2][1,3,5]*mps_up[Lx-1][1,8,2]*mpo[Lx-1][3,4,9,2]*mps_down[Lx-1][5,-1,4]*mps_up[Lx][8,6]*mpo[Lx][9,7,6]*mps_down[Lx][-2,7];
        end
    elseif 1<py<Ly
        if pxa==1
            @tensor Norm[:]:=mps_up[1][8,6]*mpo[1][7,-1,6]*mps_down[1][9,7]*mps_up[2][8,1,2]*mpo[2][-2,4,3,2]*mps_down[2][9,5,4]*envR_set[3][1,3,5];
        elseif 1<pxa<Lx-1
            @tensor Norm[:]:=envL_set[pxa-1][1,3,5]*mps_up[pxa][1,11,2]*mpo[pxa][3,4,-1,2]*mps_down[pxa][5,12,4]*mps_up[pxb][11,6,7]*mpo[pxb][-2,9,8,7]*mps_down[pxb][12,10,9]*envR_set[pxb+1][6,8,10];
        elseif pxa==Lx-1
            @tensor Norm[:]:=envL_set[Lx-2][1,3,5]*mps_up[Lx-1][1,8,2]*mpo[Lx-1][3,4,-1,2]*mps_down[Lx-1][5,9,4]*mps_up[Lx][8,6]*mpo[Lx][-2,7,6]*mps_down[Lx][9,7];
        end
    elseif py==Ly
        if pxa==1
            @tensor Norm[:]:=mps_up[1][-1,6]*mpo[1][7,8,6]*mps_down[1][9,7]*mps_up[2][-2,1,2]*mpo[2][8,4,3,2]*mps_down[2][9,5,4]*envR_set[3][1,3,5];
        elseif 1<pxa<Lx-1
            @tensor Norm[:]:=envL_set[pxa-1][1,3,5]*mps_up[pxa][1,-1,2]*mpo[pxa][3,4,11,2]*mps_down[pxa][5,12,4]*mps_up[pxb][-2,6,7]*mpo[pxb][11,9,8,7]*mps_down[pxb][12,10,9]*envR_set[pxb+1][6,8,10];
        elseif pxa==Lx-1
            @tensor Norm[:]:=envL_set[Lx-2][1,3,5]*mps_up[Lx-1][1,-1,2]*mpo[Lx-1][3,4,8,2]*mps_down[Lx-1][5,9,4]*mps_up[Lx][-2,6]*mpo[Lx][8,7,6]*mps_down[Lx][9,7];
        end
    end

    return Norm, log_coe, envL_set,envR_set
end


function cost_tbond(env,tbond_bra,tbond_ket)
    ov=@tensor env[3,4,5,6]*tbond_bra'[3,1,5,2]*tbond_ket[4,1,6,2];
    return ov
end


function initial_boundary_mps(psi_double)
    Lx,Ly=size(psi_double);
    posy=1;
    #truncation method
    mpo_mps_fun=simple_truncate_to_moddle;
    #construct top and bot environment

    # log_coe=0;
    log_coe_bot_set=Vector{Number}(undef,Ly);
    log_coe_top_set=Vector{Number}(undef,Ly);
    mps_bot_set=Vector{Any}(undef,Ly);
    mps_top_set=Vector{Any}(undef,Ly);

    trun_history=[];

    if posy>1
        mps_bot=(psi_double[:,1]...,);
        mps_bot_set[1]=mps_bot;
        mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
        trun_history=vcat(trun_history,trun_errs);
        for cy=2:min(posy-1,Ly-2)
            mpo=(psi_double[:,cy]...,);
            mps_bot,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
            mps_bot_set[cy]=mps_bot;
            log_coe_bot_set[cy]=log(coe);
            # log_coe=log_coe+log(coe);
            trun_history=vcat(trun_history,trun_errs);
        end
    end




    if posy<Ly
        mps_top=(psi_double[:,Ly]...,);
        mps_top=pi_rotate_mps(mps_top);
        mps_top_set[Ly]=treat_mps_top(mps_top);
        mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
        trun_history=vcat(trun_history,trun_errs);
        for cy=Ly-1:-1:max(posy+1,3)
            mpo=pi_rotate_mpo((psi_double[:,cy]...,));
            mps_top,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
            mps_top_set[cy]=treat_mps_top(mps_top);
            # log_coe=log_coe+log(coe);
            log_coe_top_set[cy]=log(coe);
            trun_history=vcat(trun_history,trun_errs);
        end
        # mps_top=treat_mps_top(mps_top);
    end

    # println(trun_history)
    return mps_top_set,mps_bot_set, log_coe_top_set,log_coe_bot_set
end

function initial_LR_env(mps_up,mpo,mps_down)
    Lx=length(mpo);
    envR_set=Vector{Any}(undef,Lx);
    envL_set=Vector{Any}(undef,Lx);

    # if pxa==1
        @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
        envR_set[Lx]=envR;
        for cc=Lx-1:-1:2+1
            @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR_set[cc+1][1,3,5];
            envR_set[cc]=envR;
        end
        # @tensor Norm[:]:=mps_up[1][8,6]*mpo[1][7,9,6]*mps_down[1][-1,7]*mps_up[2][8,1,2]*mpo[2][9,4,3,2]*mps_down[2][-2,5,4]*envR[1,3,5];

    # elseif 1<pxa<Lx-1
    #     @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
    #     @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
    #     for cc=2:pxa-1
    #         @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
    #     end
    #     for cc=Lx-1:-1:pxb+1
    #         @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
    #     end
    #     @tensor Norm[:]:=envL[1,3,5]*mps_up[pxa][1,11,2]*mpo[pxa][3,4,12,2]*mps_down[pxa][5,-1,4]*mps_up[pxb][11,6,7]*mpo[pxb][12,9,8,7]*mps_down[pxb][-2,10,9]*envR[6,8,10];
    # elseif pxa==Lx-1
    #     @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
    #     for cc=2:Lx-2
    #         @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
    #     end
    #     @tensor Norm[:]:=envL[1,3,5]*mps_up[Lx-1][1,8,2]*mpo[Lx-1][3,4,9,2]*mps_down[Lx-1][5,-1,4]*mps_up[Lx][8,6]*mpo[Lx][9,7,6]*mps_down[Lx][-2,7];
    # end
    
    return envL_set,envR_set
end
function LR_env_right_move(mps_up,mpo,mps_down, envL_set,envR_set, pxa,pxb)
    Lx=length(mpo);
    #(pxa,pxb) is the coordinate of tensors of the new bond
    @assert pxa>1;

    if pxa==2
        @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
        envL_set[1]=envL;
    elseif 2<pxa<=Lx-1
        @tensor envL[:]:=mps_up[pxa-1][1,-1,2]*mpo[pxa-1][3,4,-2,2]*mps_down[pxa-1][5,-3,4]*envL_set[pxa-2][1,3,5];
        envL_set[pxa-1]=envL;
    end
    
    return envL_set,envR_set
end

function upmove_boundary_mps(psi_double, posy, mps_top_set,mps_bot_set, log_coe_top_set,log_coe_bot_set)
    Lx,Ly=size(psi_double);
    #up move by one site to posy
    #contract up to posy-1

    #truncation method
    mpo_mps_fun=simple_truncate_to_moddle;
    #construct top and bot environment

    # log_coe=0;


    trun_history=[];

    if posy==2
        mps_bot=(psi_double[:,1]...,);
        mps_bot_set[1]=mps_bot;
        mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
        trun_history=vcat(trun_history,trun_errs);
    elseif 2<posy<=Ly-1
        cy=posy-1;
        mpo=(psi_double[:,cy]...,);
        mps_bot,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot_set[cy-1]);
        mps_bot_set[cy]=mps_bot;
        log_coe_bot_set[cy]=log(coe);
        # log_coe=log_coe+log(coe);
        trun_history=vcat(trun_history,trun_errs);
    end




    # if posy<Ly
    #     mps_top=(psi_double[:,Ly]...,);
    #     mps_top=pi_rotate_mps(mps_top);
    #     mps_top_set[Ly]=treat_mps_top(mps_top);
    #     mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
    #     trun_history=vcat(trun_history,trun_errs);
    #     for cy=Ly-1:-1:max(posy+1,3)
    #         mpo=pi_rotate_mpo((psi_double[:,cy]...,));
    #         mps_top,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
    #         mps_top_set[cy]=treat_mps_top(mps_top);
    #         # log_coe=log_coe+log(coe);
    #         log_coe_top_set[cy]=log(coe);
    #         trun_history=vcat(trun_history,trun_errs);
    #     end
    #     # mps_top=treat_mps_top(mps_top);
    # end

    # println(trun_history)
    return mps_top_set,mps_bot_set, log_coe_top_set,log_coe_bot_set
end




function optimize_overlap_sweep_x(psi_small,psi_big,Dmax,is_rotated)
    #x direction bond
    Lx,Ly=size(psi_small);
    # bond_coord_set=get_bond_pos(Lx,Ly);
    psi_double_small=construct_double_layer(psi_small,psi_small);
    psi_double_big=construct_double_layer(psi_small,psi_big);

    mps_top_set_small,mps_bot_set_small, log_coe_top_set_small,log_coe_bot_set_small=initial_boundary_mps(psi_double_small);
    mps_top_set_big,mps_bot_set_big, log_coe_top_set_big,log_coe_bot_set_big=initial_boundary_mps(psi_double_big);

    for py=1:Ly
        if 1<py<Ly
            mps_top_set_small,mps_bot_set_small, log_coe_top_set_small,log_coe_bot_set_small=upmove_boundary_mps(psi_double_small, py, mps_top_set_small,mps_bot_set_small, log_coe_top_set_small,log_coe_bot_set_small);
            mps_top_set_big,mps_bot_set_big, log_coe_top_set_big,log_coe_bot_set_big=upmove_boundary_mps(psi_double_big, py, mps_top_set_big,mps_bot_set_big, log_coe_top_set_big,log_coe_bot_set_big);
        else
            #no need to update boundary mps
        end
        if py==1
            envL_set_small,envR_set_small=initial_LR_env(mps_top_set_small[3],psi_double_small[:,2],psi_double_small[:,1]);
            envL_set_big,envR_set_big=initial_LR_env(mps_top_set_big[3],psi_double_big[:,2],psi_double_big[:,1]);
        elseif 1<py<Ly
            envL_set_small,envR_set_small=initial_LR_env(mps_top_set_small[py+1],psi_double_small[:,py],mps_bot_set_small[py-1]);
            envL_set_big,envR_set_big=initial_LR_env(mps_top_set_big[py+1],psi_double_big[:,py],mps_bot_set_big[py-1]);
        elseif py==Ly
            mps_up=treat_mps_top(pi_rotate_mps(psi_double_small[:,Ly]));
            envL_set_small,envR_set_small=initial_LR_env(mps_up,psi_double_small[:,Ly-1],mps_bot_set_small[Ly-2]);
            mps_up=treat_mps_top(pi_rotate_mps(psi_double_big[:,Ly]));
            envL_set_big,envR_set_big=initial_LR_env(mps_up,psi_double_big[:,Ly-1],mps_bot_set_big[Ly-2]);
        end

        for px=1.5:1:Lx-0.5
            #jldsave("test.jld2";psi_small,psi_big,cp,bond_coord_set);
            
            # px,py=bond_coord_set[cp,:];
            if is_rotated
                println("y bond: "*string([px,py])*"->"*string(coord_rotate([px,py],Lx,Ly)));
            else
                println("x bond: "*string([px,py]));
            end


            t_bond_big,psi_left_big=get_bond(psi_big,px,py,"dD",0);
            t_bond_origin,psi_left_small=get_bond(psi_small,px,py,"dD",0);

            @time N_env_big,log_coe_big,psi_double_big, envL_set_big,envR_set_big=env_2site_bra_ket(psi_left_small,psi_left_big,psi_double_big, mps_top_set_big,mps_bot_set_big, log_coe_top_set_big,log_coe_bot_set_big, px,py, envL_set_big,envR_set_big);
            @time N_env_small,log_coe_small,psi_double_small, envL_set_small,envR_set_small=env_2site_bra_ket(psi_left_small,psi_left_small,psi_double_small, mps_top_set_small,mps_bot_set_small, log_coe_top_set_small,log_coe_bot_set_small, px,py, envL_set_small,envR_set_small);
            # @time N_env_big_big,log_coe_big_big=env_2site_bra_ket(psi_left_big,psi_left_big,px,py);
            
            log_coe_small=log_coe_small+log(norm(N_env_small));
            log_coe_big=log_coe_big+log(norm(N_env_big));
            # log_coe_big_big=log_coe_big_big+log(norm(N_env_big_big));

            N_env_small=N_env_small/norm(N_env_small);
            N_env_big=N_env_big/norm(N_env_big);
            # N_env_big_big=N_env_big_big/norm(N_env_big_big);

            #solve new bond tensor locally:
            Id=unitary(space(t_bond_big,2),space(t_bond_big,2));
            @tensor M[:]:=N_env_small[-1,-5,-3,-7]*Id[-2,-6]*Id[-4,-8];
            @tensor B[:]:=N_env_big[-1,1,-3,2]*t_bond_big[1,-2,2,-4];
            M=permute(M,(1,2,3,4,),(5,6,7,8,));
            @assert norm(M-M')/norm(M)<1e-6;
            M=(M+M')/2;
            eu,ev=eigh(M);
            @assert norm(ev*eu*ev'-M)/norm(M)<1e-10
            eu=check_positive(eu);
            M_inv=ev*my_pinv(eu)*ev';
            @tensor t_bond_new[:]:=M_inv[-1,-2,-3,-4,1,2,3,4]*B[1,2,3,4];


            ov01=cost_tbond(N_env_big,t_bond_origin,t_bond_big);
            ov11=cost_tbond(N_env_small,t_bond_origin,t_bond_origin);
            # ov00=cost_tbond(N_env_big_big,t_bond_big,t_bond_big);
            # ov_origin=exp(log_coe_big-log_coe_small/2-log_coe_big_big/2)*ov01/sqrt(ov11*ov00);
            ov_origin=exp(log_coe_big-log_coe_small/2)*ov01/sqrt(ov11);

            ov01=cost_tbond(N_env_big,t_bond_new,t_bond_big);
            ov11=cost_tbond(N_env_small,t_bond_new,t_bond_new);
            # ov_new=exp(log_coe_big-log_coe_small/2-log_coe_big_big/2)*ov01/sqrt(ov11*ov00);
            ov_new=exp(log_coe_big-log_coe_small/2)*ov01/sqrt(ov11);

            println("optimized overlap from bond tensor: "*string(ov_origin)*" -> "*string(ov_new));
            @assert (abs(ov_new)>abs(ov_origin))|((1-abs(ov_new/ov_origin))<1e-7);
            ####################
            T1a,T2a=bond_simple_trun(t_bond_new);
            println("virtual space from direct svd: "*string(space(T1a,3)))
            T1aa,T2aa=optimize_truncation(T1a,T2a,t_bond_new,N_env_small)
            psi_newaa,_=set_bond_cut(psi_left_small,nothing,px,py,T1aa,T2aa);
            ovaa=cost_LR([T1aa,T2aa],t_bond_new,N_env_small);
            #####################
            
            T1b,T2b=bond_gauge_fix_trun(t_bond_new,N_env_small);
            println("virtual space from gauge fix: "*string(space(T1b,3)))
            T1bb,T2bb=optimize_truncation(T1b,T2b,t_bond_new,N_env_small)
            psi_newbb,_=set_bond_cut(psi_left_small,nothing,px,py,T1bb,T2bb);
            ovbb=cost_LR([T1bb,T2bb],t_bond_new,N_env_small);
            #######################

            if abs(ovbb)>abs(ovaa)
                psi_small=psi_newbb;
                println("choose gauge fix trun")
            else
                psi_small=psi_newaa;
                println("choose direct trun")
            end

            AA1,_=construct_double_layer_pos(psi_small,psi_small,Int.(px-0.5),py);
            AA2,_=construct_double_layer_pos(psi_small,psi_small,Int.(px+0.5),py);
            psi_double_small[Int.(px-0.5),py]=AA1;
            psi_double_small[Int.(px+0.5),py]=AA2;
            AA1,_=construct_double_layer_pos(psi_small,psi_big,Int.(px-0.5),py);
            AA2,_=construct_double_layer_pos(psi_small,psi_big,Int.(px+0.5),py);
            psi_double_big[Int.(px-0.5),py]=AA1;
            psi_double_big[Int.(px+0.5),py]=AA2;

        end
    end
    return psi_small
end



function optimize_overlap_sweep(psi,psi_big,Dmax)
    #sweep along x bond
    psi=optimize_overlap_sweep_x(psi,psi_big,Dmax,false);

    #sweep along y bond
    psi=rotate_psi(psi);#rotate 90 degree
    psi_big=rotate_psi(psi_big);#rotate 90 degree
    psi=optimize_overlap_sweep_x(psi,psi_big,Dmax,true);
    psi=rotate_psi(rotate_psi(rotate_psi(psi)));#rotate 270 degree
    psi_big=rotate_psi(rotate_psi(rotate_psi(psi_big)));#rotate 270 degree
    return psi
end


function full_update(parameters,dt,psi,Dmax,n_lattice_sweep,direction)
    #second order trotter gate
    psi=deepcopy(psi);
    psi=disk_to_torus(psi);#add trivial links at boundary 
    Lx,Ly=size(psi);
    triangle_set_set=get_triangles(parameters,Lx,Ly,dt);

    if direction=="forward"
        #from beginning to end
    elseif direction=="backward"
        #from end to beginning
        for c1=1:length(triangle_set_set)
            triangle_set_set[c1]=triangle_set_set[c1][end:-1:1];
        end
        triangle_set_set=triangle_set_set[end:-1:1];
    end

    
    psi_big=deepcopy(psi);
    for c1=1:length(triangle_set_set)
        triangle_set=triangle_set_set[c1];
        
        for c2=1:length(triangle_set)
            triangle=triangle_set[c2];
            psi_big=apply_triangle_op(psi_big,triangle["plaquatte"],triangle["sites"],triangle["gate"]);

            println(triangle["plaquatte"])
            println(triangle["sites"])
        end

        psi_big=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi_big));#remove boundary links
        psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));#remove boundary links
        psi_old=deepcopy(psi);
        for cop=1:n_lattice_sweep
            psi=optimize_overlap_sweep(psi,psi_big,Dmax);
        end

        # ov11=compute_ov(psi_big,psi_big);
        ov00=compute_ov(psi,psi);
        ov10=compute_ov(psi_big,psi);
        # ov_opt=ov10/sqrt(ov00*ov11);
        ov_opt=ov10/sqrt(ov00);
        # ov_init=compute_ov(psi_big,psi_old)/sqrt(compute_ov(psi_big,psi_big)*compute_ov(psi_old,psi_old))
        ov_init=compute_ov(psi_big,psi_old)/sqrt(compute_ov(psi_old,psi_old))
        println("overlap after sweep: "*string(ov_init)*"->"*string(ov_opt));

        psi=disk_to_torus(psi);#add trivial links at boundary 
        psi_big=deepcopy(psi);#for next evolution
    end
    #normalization of tensors
    for cc1=1:Lx
        for cc2=1:Ly
            psi[cc1,cc2]=psi[cc1,cc2]/norm(psi[cc1,cc2]);
        end
    end
    

    #####################
    psi=cylinder_xpbc_to_disk(torus_to_cylinder_xpbc(psi));#remove boundary links
    psi=gauge_fix_global(psi,1,false);
    E_new=real(cost_fun_global(psi));
    println("Energy of updated state: "*string(E_new));flush(stdout);



    return psi,E_new
end