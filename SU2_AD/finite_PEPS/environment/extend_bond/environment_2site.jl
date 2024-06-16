function add_trivial_leg(T::TensorMap)
    if sectortype(space(T,1)) == Trivial
        Vtrivial=(ℂ^1);
    else
        Vtrivial=Rep[SU₂](0=>1);
    end
    if Rank(T)==2
        V=space(T,2);
        Un=unitary(V*Vtrivial,V);
        @tensor T[:]:=T[-1,1]*Un[-2,-3,1];
    elseif Rank(T)==3
        V=space(T,3);
        Un=unitary(V*Vtrivial,V);
        @tensor T[:]:=T[-1,-2,1]*Un[-3,-4,1];
    elseif Rank(T)==4
        V=space(T,4);
        Un=unitary(V*Vtrivial,V);
        @tensor T[:]:=T[-1,-2,-3,1]*Un[-4,-5,1];
    end
    return T
end

function get_2site_env_x(psi_double,posxa,posxb,posy)
    Lx,Ly=size(psi_double);

    #truncation method
    mpo_mps_fun=simple_truncate_to_moddle;
    #construct top and bot environment

    log_coe=0;
    trun_history=[];

    if posy>1
        mps_bot=(psi_double[:,1]...,);
        mps_bot,trun_errs,_=left_truncate_simple(mps_bot, chi, multiplet_tol);
        trun_history=vcat(trun_history,trun_errs);
        for cy=2:min(posy-1,Ly-2)
            mpo=(psi_double[:,cy]...,);
            mps_bot,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_bot);
            log_coe=log_coe+log(coe);
            trun_history=vcat(trun_history,trun_errs);
        end
    end


    function treat_mps_top(mps)
        #convert mps_top to normal order
        mps=mps[end:-1:1];
        for cx=2:Lx-1
            mps=mps_update(mps,permute(mps[cx],(2,1,3,)),cx);
        end
        return mps
    end

    if posy<Ly
        mps_top=(psi_double[:,Ly]...,);
        mps_top=pi_rotate_mps(mps_top);
        mps_top,trun_errs,_=left_truncate_simple(mps_top, chi, multiplet_tol);
        trun_history=vcat(trun_history,trun_errs);
        for cy=Ly-1:-1:max(posy+1,3)
            mpo=pi_rotate_mpo((psi_double[:,cy]...,));
            mps_top,trun_errs,coe=Zygote.checkpointed(mpo_mps_fun, mpo, mps_top);
            log_coe=log_coe+log(coe);
            trun_history=vcat(trun_history,trun_errs);
        end
        mps_top=treat_mps_top(mps_top);
    end

    # println(trun_history)

    ########################################


    py=posy;
    pxa=posxa;
    pxb=posxb;


    if py==1
        mps_up=mps_top;
        mpo=psi_double[:,py+1];
        mps_down=psi_double[:,py];
        if pxa==1
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=Lx-1:-1:2+1
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=mps_up[1][8,6]*mpo[1][7,9,6]*mps_down[1][-1,7]*mps_up[2][8,1,2]*mpo[2][9,4,3,2]*mps_down[2][-2,5,4]*envR[1,3,5];

        elseif 1<pxa<Lx-1
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=2:pxa-1
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            for cc=Lx-1:-1:pxb+1
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=envL[1,3,5]*mps_up[pxa][1,11,2]*mpo[pxa][3,4,12,2]*mps_down[pxa][5,-1,4]*mps_up[pxb][11,6,7]*mpo[pxb][12,9,8,7]*mps_down[pxb][-2,10,9]*envR[6,8,10];
        elseif pxa==Lx-1
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            for cc=2:Lx-2
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            @tensor Norm[:]:=envL[1,3,5]*mps_up[Lx-1][1,8,2]*mpo[Lx-1][3,4,9,2]*mps_down[Lx-1][5,-1,4]*mps_up[Lx][8,6]*mpo[Lx][9,7,6]*mps_down[Lx][-2,7];
        end


    elseif 1<py<Ly
        mps_up=mps_top;
        mpo=psi_double[:,py];
        mps_down=mps_bot;

        if pxa==1
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=Lx-1:-1:2+1
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=mps_up[1][8,6]*mpo[1][7,-1,6]*mps_down[1][9,7]*mps_up[2][8,1,2]*mpo[2][-2,4,3,2]*mps_down[2][9,5,4]*envR[1,3,5];

        elseif 1<pxa<Lx-1
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=2:pxa-1
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            for cc=Lx-1:-1:pxb+1
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=envL[1,3,5]*mps_up[pxa][1,11,2]*mpo[pxa][3,4,-1,2]*mps_down[pxa][5,12,4]*mps_up[pxb][11,6,7]*mpo[pxb][-2,9,8,7]*mps_down[pxb][12,10,9]*envR[6,8,10];
        elseif pxa==Lx-1
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            for cc=2:Lx-2
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            @tensor Norm[:]:=envL[1,3,5]*mps_up[Lx-1][1,8,2]*mpo[Lx-1][3,4,-1,2]*mps_down[Lx-1][5,9,4]*mps_up[Lx][8,6]*mpo[Lx][-2,7,6]*mps_down[Lx][9,7];
        end


    elseif py==Ly
        mps_up=psi_double[:,py];
        mps_up=treat_mps_top(pi_rotate_mps(mps_up));
        mpo=psi_double[:,py-1];
        mps_down=mps_bot;

        if pxa==1
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=Lx-1:-1:2+1
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=mps_up[1][-1,6]*mpo[1][7,8,6]*mps_down[1][9,7]*mps_up[2][-2,1,2]*mpo[2][8,4,3,2]*mps_down[2][9,5,4]*envR[1,3,5];

        elseif 1<pxa<Lx-1
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            @tensor envR[:]:=mps_up[Lx][-1,1]*mpo[Lx][-2,2,1]*mps_down[Lx][-3,2];
            for cc=2:pxa-1
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            for cc=Lx-1:-1:pxb+1
                @tensor envR[:]:=mps_up[cc][-1,1,2]*mpo[cc][-2,4,3,2]*mps_down[cc][-3,5,4]*envR[1,3,5];
            end
            @tensor Norm[:]:=envL[1,3,5]*mps_up[pxa][1,-1,2]*mpo[pxa][3,4,11,2]*mps_down[pxa][5,12,4]*mps_up[pxb][-2,6,7]*mpo[pxb][11,9,8,7]*mps_down[pxb][12,10,9]*envR[6,8,10];

        elseif pxa==Lx-1
            @tensor envL[:]:=mps_up[1][-1,1]*mpo[1][2,-2,1]*mps_down[1][-3,2];
            for cc=2:Lx-2
                @tensor envL[:]:=mps_up[cc][1,-1,2]*mpo[cc][3,4,-2,2]*mps_down[cc][5,-3,4]*envL[1,3,5];
            end
            @tensor Norm[:]:=envL[1,3,5]*mps_up[Lx-1][1,-1,2]*mpo[Lx-1][3,4,8,2]*mps_down[Lx-1][5,9,4]*mps_up[Lx][-2,6]*mpo[Lx][8,7,6]*mps_down[Lx][9,7];
        end


    end
    return Norm, log_coe
end

function env_2site(psi_left,px,py)
    Lx,Ly=size(psi_left);
    psi=deepcopy(psi_left);

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

        psi_double=construct_double_layer(psi,psi);
        Norm,log_coe=get_2site_env_x(psi_double,pos_T1[1],pos_T2[1],pos_T1[2]);
        _,U_L1,U_D1,U_R1,U_U1=build_double_layer_open_position(psi[pos_T1[1],pos_T1[2]],pos_T1[1],pos_T1[2],Lx,Ly,true);
        _,U_L2,U_D2,U_R2,U_U2=build_double_layer_open_position(psi[pos_T2[1],pos_T2[2]],pos_T2[1],pos_T2[2],Lx,Ly,true);
        U1=U_R1;
        U2=U_L2;
        @tensor Norm[:]:=Norm[1,2]*U1'[1,-1,-2]*U2'[-3,-4,2];#D1',D1, D2',D2
    elseif bond_type=="y"
        @assert pos_T1[1]==pos_T2[1];
        @assert (pos_T1[2]-1)==pos_T2[2];
        psi_rotated=Matrix{TensorMap}(undef,Ly,Lx);
        function coord_rotate(coord,Lx,Ly)
            return [Ly-coord[2]+1,coord[1]]
        end
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

        pos_T1_new=coord_rotate(pos_T1,Lx,Ly);
        pos_T2_new=coord_rotate(pos_T2,Lx,Ly);
        psi_double=construct_double_layer(psi_rotated,psi_rotated);
        Norm,log_coe=get_2site_env_x(psi_double,pos_T1_new[1],pos_T2_new[1],pos_T1_new[2]);
        _,U_L1,U_D1,U_R1,U_U1=build_double_layer_open_position(psi_rotated[pos_T1_new[1],pos_T1_new[2]],pos_T1_new[1],pos_T1_new[2],Ly,Lx,true);
        _,U_L2,U_D2,U_R2,U_U2=build_double_layer_open_position(psi_rotated[pos_T2_new[1],pos_T2_new[2]],pos_T2_new[1],pos_T2_new[2],Ly,Lx,true);
        U1=U_R1;
        U2=U_L2;
        @tensor Norm[:]:=Norm[1,2]*U1'[1,-1,-2]*U2'[-3,-4,2];#D1',D1, D2',D2

    end

    return Norm,log_coe
end


function bond_simple_trun(t_bond,D0=nothing)
    global Dmax
    if D0==nothing
        D0=Dmax
    else
        D0=D0
    end
    u,s,v=tsvd(permute(t_bond,(1,2,),(3,4,)); trunc=truncdim(D0));

    T1=permute(u*sqrt(s),(1,2,3,));
    T2=permute(sqrt(s)*v,(1,2,3,));

    T1=T1/norm(T1);
    T2=T2/norm(T2);
    return T1,T2
end

function check_positive(T)
    T_dense=convert(Array,T);
    T_new=deepcopy(T);
    @assert (norm(diag(T_dense))-norm(T_dense))/norm(T_dense)<1e-14;#verify diagonal


    #change negative eigenvalue to zero
    if sectortype(space(T,1)) == Trivial
        mm=T.data;
        for cc=1:size(mm,1)
            if real(mm[cc,cc])<0
                mm[cc,cc]=0;
            end
        end
        T_new=TensorMap(mm,codomain(eu),domain(eu));
    else
        for cc=1:length(T.data.values)
            mm=T.data.values[cc];
            for dd=1:size(mm,1)
                if real(mm[dd,dd])<0
                    mm[dd,dd]=0;
                end
            end
            T_new.data.values[cc]=mm;
        end
    end

    @assert norm(T-T_new)/norm(T_new)<0.01;
    return T_new
end

function bond_gauge_fix_trun(t_bond,N_env,D0=nothing)
    global Dmax
    if D0==nothing
        D0=Dmax
    else
        D0=D0
    end

    N_env=deepcopy(N_env);
    t_bond=deepcopy(t_bond);

    N_env=permute(N_env,(1,3,),(2,4,));
    @assert norm(N_env-N_env')/norm(N_env)<1e-6;
    N_env=(N_env+N_env')/2;
    N_env=N_env/norm(N_env);
    eu,ev=eigh(N_env);

    eu=check_positive(eu);



    X=sqrt(eu)*ev';#D0,Dl,Dr
    # println(norm(X'*X-N_env))
    u1,s1,v1=tsvd(permute(X,(2,),(1,3)));#Dl,   (D0,Dr)
    u2,s2,v2=tsvd(permute(X,(2,1,),(3,)));#(Dl,D0),  Dr

    gate1=my_pinv(s1)*u1';
    gate1_inv=u1*s1;
    gate2=v2'*my_pinv(s2);
    gate2_inv=s2*v2;

    @tensor X_new[:]:=gate1[-2,2]*X[-1,2,3]*gate2[3,-3];
    @tensor t_bond_new[:]:=gate1_inv[1,-1]*t_bond[1,-2,3,-4]*gate2_inv[-3,3];

    # @tensor tt0[:]:=X[-1,1,2]*t_bond[1,-2,2,-3];
    # @tensor tt1[:]:=X_new[-1,1,2]*t_bond_new[1,-2,2,-3];
    # println(norm(tt0-tt1)/norm(tt0))

    
    u,s,v=tsvd(permute(t_bond_new,(1,2,),(3,4,)); trunc=truncdim(D0));

    u=u*sqrt(s);
    v=sqrt(s)*v;
    @tensor T1[:]:=gate1[1,-1]*u[1,-2,-3];
    @tensor T2[:]:=v[-1,2,-3]*gate2[-2,2];

    T1=T1/norm(T1);
    T2=T2/norm(T2);
    return T1,T2

end

function cost_LR(x,tbond,Nenv)
    t1=x[1];
    t2=x[2];
    @tensor tbond_short[:]:=t1[-1,-2,1]*t2[1,-3,-4];
    ov11=@tensor Nenv[3,4,5,6]* tbond_short'[3,1,5,2]*tbond_short[4,1,6,2];
    ov22=@tensor Nenv[3,4,5,6]* tbond'[3,1,5,2]*tbond[4,1,6,2];
    ov12=@tensor Nenv[3,4,5,6]* tbond'[3,1,5,2]*tbond_short[4,1,6,2];
    ov=abs(ov12)/sqrt(ov11*ov22);
    @assert abs(imag(ov11)/real(ov11))<1e-10;
    @assert abs(imag(ov22)/real(ov22))<1e-10;
    return -abs(ov)
end

function solve_T1(T1,T2,tbond,Nenv)
    I=unitary(space(T1,2),space(T1,2));
    @tensor M[:]:=Nenv[-1,-4,1,3]*T2'[-3,1,2]*T2[-6,3,2]*I[-2,-5];
    @tensor B[:]:=Nenv[-1,4,1,2]*T2'[-3,1,3]*tbond[4,-2,2,3];
    M=permute(M,(1,2,3,),(4,5,6,));
    # u,s,v=tsvd(M);
    # M_inv=v'*my_pinv(s)*u';
    @assert norm(M-M')/norm(M)<1e-6;
    M=(M+M')/2;
    eu,ev=eigh(M);
    @assert norm(ev*eu*ev'-M)/norm(M)<1e-10
    eu=check_positive(eu);
    M_inv=ev*my_pinv(eu)*ev';


    @tensor T1_new[:]:=M_inv[-1,-2,-3,1,2,3]*B[1,2,3];
    return T1_new
end
function solve_T2(T1,T2,tbond,Nenv)
    I=unitary(space(T2,3),space(T2,3));
    @tensor M[:]:=Nenv[1,3,-2,-5]*T1'[1,2,-1]*T1[3,2,-4]*I[-3,-6];
    @tensor B[:]:=Nenv[1,2,-2,3]*T1'[1,4,-1]*tbond[2,4,3,-3];
    M=permute(M,(1,2,3,),(4,5,6,));
    # u,s,v=tsvd(M);
    # M_inv=v'*my_pinv(s)*u';
    @assert norm(M-M')/norm(M)<1e-6;
    M=(M+M')/2;
    eu,ev=eigh(M);
    @assert norm(ev*eu*ev'-M)/norm(M)<1e-10
    eu=check_positive(eu);
    M_inv=ev*my_pinv(eu)*ev';
    @tensor T2_new[:]:=M_inv[-1,-2,-3,1,2,3]*B[1,2,3];
    return T2_new
end

function optimize_truncation(T1,T2,tbond,Nenv,D0=nothing)
    Nenv=Nenv/norm(Nenv);
    tbond=tbond/norm(tbond);
    ov0=cost_LR([T1,T2],tbond,Nenv);
    # println(ov0);
    ##################
    for co=1:10
        T1=solve_T1(T1,T2,tbond,Nenv);
        T2=solve_T2(T1,T2,tbond,Nenv);
    end

    ###################
    x_new=[T1,T2];
    ov1=cost_LR(x_new,tbond,Nenv);
    # println(ov1);
    println("optimized overlap from site tensor: "*string(ov0)*" -> "*string(ov1));
    @assert (ov1<ov0)|((1-abs(ov1/ov0))<1e-7);
    ##################
    T1new=x_new[1];
    T2new=x_new[2];

    #gauge fix
    global Dmax
    if D0==nothing
        D0=Dmax
    else
        D0=D0
    end
    @tensor t_bond_new[:]:=T1new[-1,-2,1]*T2new[1,-3,-4];
    u,s,v=tsvd(permute(t_bond_new,(1,2,),(3,4,)); trunc=truncdim(D0));
    T1=u*sqrt(s);
    T2=sqrt(s)*v;

    T1=T1/norm(T1);
    T2=T2/norm(T2);

    return T1,T2
end

# function optimize_truncation(T1,T2,tbond,Nenv,D0=nothing)
#     ov0=cost_LR([T1,T2],tbond,Nenv);
#     # println(ov0);
#     ##################
#     ls = BackTracking(c_1=0.0001,ρ_hi=0.5,ρ_lo=0.1,iterations=10,order=3,maxstep=Inf);
#     LS_maxiter=10;#number of gradient optimization 
#     grad_tol=1e-8;
#     ov_new, x_new, iter_bt3 = gdoptimize_trun(tbond, Nenv, f_LR, g!_LR, fg!_LR, [T1,T2], ls,LS_maxiter, 1e-8, grad_tol);
#     ov1=cost_LR(x_new,tbond,Nenv);
#     # println(ov1);
#     println("optimized overlap: "*string(ov0)*" -> "*string(ov1));
#     @assert (ov1<ov0)|((1-abs(ov1/ov0))<1e-7);
#     ##################
#     T1new=x_new[1];
#     T2new=x_new[2];

#     #gauge fix
#     global Dmax
#     if D0==nothing
#         D0=Dmax
#     else
#         D0=D0
#     end
#     @tensor t_bond_new[:]:=T1new[-1,-2,1]*T2new[1,-3,-4];
#     u,s,v=tsvd(permute(t_bond_new,(1,2,),(3,4,)); trunc=truncdim(D0));
#     T1=u*sqrt(s);
#     T2=sqrt(s)*v;

#     T1=T1/norm(T1);
#     T2=T2/norm(T2);

#     return T1,T2
# end


# function gdoptimize_trun(tbond, Nenv, f, g!, fg!, x0, linesearch, maxiter::Int = 20, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 


#     x = deepcopy(x0)
#     gvec = similar(x)
#     g!_LR(gvec, x,tbond, Nenv)
#     fx = f_LR(x,tbond, Nenv)
#     gnorm = norm(gvec)
#     gtol = max(g_rtol*gnorm, g_atol)

#     # Univariate line search functions
#     ϕ(α) = f_LR(x + α*s,tbond, Nenv)
#     function dϕ(α)
#         g!_LR(gvec, x + α*s,tbond, Nenv)
#         return real(dot(gvec, s)) #I am not sure if taking real part is reasonable. If the output is complex the algorithm fails.
#     end
#     function ϕdϕ(α)
#         phi = fg!_LR(gvec, x + α*s,tbond, Nenv)
#         dphi = real(dot(gvec, s)) #I am not sure if taking real part is reasonable. If the output is complex the algorithm fails.
#         return (phi, dphi)
#     end

#     s = similar(gvec) # Step direction

#     iter = 0
#     while iter < maxiter && gnorm > gtol
#         # println("optim iteration "*string(iter))
#         x=x/norm(x);

#         iter += 1
#         s = (-1)*gvec

#         dϕ_0 = real(dot(s, gvec))
#         #α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
#         α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1/3, fx, dϕ_0)

#         x = x + α*s
#         g!(gvec, x,tbond, Nenv)
#         gnorm = norm(gvec)
#     end

#     return (fx, x, iter)
# end
# function f_LR(x,tbond,Nenv)
#     E=cost_LR(x,tbond,Nenv); 
#     return E;
# end
# function g!_LR(gvec, x0,tbond,Nenv)# this function changes the value of gvec  
#     ∂E=gradient(x ->cost_LR(x,tbond,Nenv), x0)[1];
#     for cc in eachindex(gvec)
#         setindex!(gvec,∂E[cc],cc);#this will change the input variable
#     end
#     return gvec
# end
# function fg!_LR(gvec, x,tbond,Nenv)
#     #println("one fg!")
#     g!_LR(gvec, x,tbond,Nenv)
#     f_LR(x,tbond,Nenv)
# end




function gauge_fix_global(psi,iterations,compute_energy=false)
    psi=deepcopy(psi);
    println("global gauge fix")
    
    global n_mps_sweep
    n_mps_sweep=5;

    E_opt=real(cost_fun_global(psi));

    bond_coord_set=[];
    for cx=1.5:1:Lx-0.5
        for cy=1:Ly
            px=cx;
            py=cy;
            bond_coord_set=vcat(bond_coord_set,(px,py,))
        end
    end

    for cx=1:Lx
        for cy=1.5:1:Ly-0.5
            px=cx;
            py=cy;
            bond_coord_set=vcat(bond_coord_set,(px,py,))
        end
    end

    #########################################

    bond_noise=0;
    trun_bond_type="dD"
    psi_double=construct_double_layer(psi,psi);

    for ci=1:iterations
        for cp =1:length(bond_coord_set)
            global E_opt
            px,py=bond_coord_set[cp];
            if compute_energy
                println("bond: "*string([px,py]));
            end

            n_mps_sweep=5;
            global D_connect
            t_bond,psi_left=get_bond(psi,px,py,trun_bond_type,bond_noise);


            N_env,log_coe=env_2site(psi_left,px,py);
            T1b,T2b=bond_gauge_fix_trun(t_bond,N_env,D_connect);
            psi,psi_double=set_bond_cut(psi_left,psi_double,px,py,T1b,T2b);
            if compute_energy
                n_mps_sweep=5;
                E_cut=real(cost_fun_global(psi));
                println("space: "*string(space(T1b,3)))
                println("Energy gauge fix cut= "*string(E_cut))
            end
    
            #normalization of tensors
            for cc1=1:Lx
                for cc2=1:Ly
                    psi[cc1,cc2]=psi[cc1,cc2]/norm(psi[cc1,cc2]);
                end
            end

        end
    end
    

    return psi
end