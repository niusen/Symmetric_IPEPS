using Statistics

function env_pinv(T)
    cut_off = 1e-6;
    epsilon = 1e-6;
    cut_off=cut_off*maximum(diag(convert(Array,T)));
    epsilon=epsilon*maximum(diag(convert(Array,T)));
    T_new=deepcopy(T);

    for (k,dst) in blocks(T_new)
        src = blocks(T_new)[k]
        @inbounds for i in 1:size(dst,1)
            if abs(dst[i,i])>cut_off
                dst[i,i] = dst[i,i]/(dst[i,i]^2+epsilon);
            else
                dst[i,i]=0;
            end
        end
    end
    return T_new
end

function get_Heff(H_env,Norm_env)
    global euu
    # euu,ev=eigen(Norm_env);
    # u1,s1,v1=tsvd(Norm_env; trunc=truncerr(1e-2));
    u1,s1,v1=tsvd(Norm_env);
    # H_eff=pinv(Norm_env)*H_env;
    H_eff=v1'*env_pinv(s1)*u1'*H_env;
    return H_eff
end



function variational_opt_site(psi,chi,only_nearest_plaquatte,px,py)
    global Lx,Ly


        psi_double=construct_double_layer(psi,psi);
        mps_set_down_move, trun_errs, norm_coe_down_move, UR_set_down_move, UL_set_down_move, unitarys_R_set_down_move, unitarys_L_set_down_move, projectors_R_set_down_move, projectors_L_set_down_move=get_projector_down_move(psi_double,py+1);
        data_down_move=Dict("mps_set"=>mps_set_down_move, " trun_errs"=> trun_errs, "norm_coe"=>norm_coe_down_move,"UR_set"=>UR_set_down_move, "UL_set"=>UL_set_down_move,  "unitarys_R_set"=>unitarys_R_set_down_move, "unitarys_L_set"=>unitarys_L_set_down_move, "projectors_R_set"=>projectors_R_set_down_move, "projectors_L_set"=>projectors_L_set_down_move);    
    
        mps_set_up_move, trun_errs, norm_coe_up_move, UR_set_up_move, UL_set_up_move, unitarys_R_set_up_move, unitarys_L_set_up_move, projectors_R_set_up_move, projectors_L_set_up_move=get_projector_up_move(psi_double,py-1);
        data_up_move=Dict("mps_set"=>mps_set_up_move, " trun_errs"=> trun_errs, "norm_coe"=>norm_coe_up_move,"UR_set"=>UR_set_up_move, "UL_set"=>UL_set_up_move,  "unitarys_R_set"=>unitarys_R_set_up_move, "unitarys_L_set"=>unitarys_L_set_up_move, "projectors_R_set"=>projectors_R_set_up_move, "projectors_L_set"=>projectors_L_set_up_move);    


            println("site: "*string([px,py]))
            
            ####################################
            Norm_env,norm_log_coe=norm_env(psi,psi_double,px,py, mps_set_down_move, norm_coe_down_move, mps_set_up_move, norm_coe_up_move);
            ###################################
            A_double_open,U_L,U_D,U_R,U_U=build_double_layer_open_position(psi[px,py],px,py,Lx,Ly,true);
            if py==Ly
                if px==1
                    A_double_open=permute(A_double_open,(2,1,3,));
                elseif 1<px<Lx
                    A_double_open=permute(A_double_open,(1,3,2,4,));
                end
            end
            ######################################
            H_env_total=Norm_env*0;
            for cx=1:Lx-1
                for cy=1:Ly-1
                    x_range=[cx,cx+1];  y_range=[cy,cy+1];
                    if (abs(mean(x_range)-px)+abs(mean(y_range)-py)>1) & only_nearest_plaquatte
                        continue
                    end
                    println("plaquatte: "*string(x_range)*", "*string(y_range))
                    H_env,H_log_coe=Ham_env_term(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move);
                    H_env_total=H_env_total+H_env;
                end
            end
            
            eu,ev=H_eig_solve(H_env_total,Norm_env,A_double_open,px,py,U_L,U_D,U_R,U_U);


            # if (px==1)&(py==3)
            #     jldsave("problem.jld2"; psi);
            #     error("break");
            # end
            
            psi[px,py]=ev;
            A_double_new,_=construct_double_layer_pos(psi,psi,px,py);
            if py==Ly
                if px==1
                    A_double_new=permute(A_double_new,(2,1,));
                elseif 1<px<Lx
                    A_double_new=permute(A_double_new,(1,3,2,));
                end
            end
            
            psi_double[px,py]=A_double_new;
            mps_set_down_move[px,py]=psi_double[px,py];
            data_down_move["mps_set"]=mps_set_down_move;
            mps_set_up_move[px,py]=psi_double[px,py];
            data_up_move["mps_set"]=mps_set_up_move;
    return psi
end



function variational_opt(psi,chi,only_nearest_plaquatte)
    global Lx,Ly

    for py=1:Ly#3:3#Ly
        psi_double=construct_double_layer(psi,psi);
        mps_set_down_move, trun_errs, norm_coe_down_move, UR_set_down_move, UL_set_down_move, unitarys_R_set_down_move, unitarys_L_set_down_move, projectors_R_set_down_move, projectors_L_set_down_move=get_projector_down_move(psi_double,py+1);
        data_down_move=Dict("mps_set"=>mps_set_down_move, " trun_errs"=> trun_errs, "norm_coe"=>norm_coe_down_move,"UR_set"=>UR_set_down_move, "UL_set"=>UL_set_down_move,  "unitarys_R_set"=>unitarys_R_set_down_move, "unitarys_L_set"=>unitarys_L_set_down_move, "projectors_R_set"=>projectors_R_set_down_move, "projectors_L_set"=>projectors_L_set_down_move);    
    
        mps_set_up_move, trun_errs, norm_coe_up_move, UR_set_up_move, UL_set_up_move, unitarys_R_set_up_move, unitarys_L_set_up_move, projectors_R_set_up_move, projectors_L_set_up_move=get_projector_up_move(psi_double,py-1);
        data_up_move=Dict("mps_set"=>mps_set_up_move, " trun_errs"=> trun_errs, "norm_coe"=>norm_coe_up_move,"UR_set"=>UR_set_up_move, "UL_set"=>UL_set_up_move,  "unitarys_R_set"=>unitarys_R_set_up_move, "unitarys_L_set"=>unitarys_L_set_up_move, "projectors_R_set"=>projectors_R_set_up_move, "projectors_L_set"=>projectors_L_set_up_move);    

        for px=1:Lx#1:1#Lx
            println("site: "*string([px,py]))
            
            ####################################
            Norm_env,norm_log_coe=norm_env(psi,psi_double,px,py, mps_set_down_move, norm_coe_down_move, mps_set_up_move, norm_coe_up_move);
            ###################################
            A_double_open,U_L,U_D,U_R,U_U=build_double_layer_open_position(psi[px,py],px,py,Lx,Ly,true);
            if py==Ly
                if px==1
                    A_double_open=permute(A_double_open,(2,1,3,));
                elseif 1<px<Lx
                    A_double_open=permute(A_double_open,(1,3,2,4,));
                end
            end
            ######################################
            H_env_total=Norm_env*0;
            for cx=1:Lx-1
                for cy=1:Ly-1
                    x_range=[cx,cx+1];  y_range=[cy,cy+1];
                    if (abs(mean(x_range)-px)+abs(mean(y_range)-py)>1) & only_nearest_plaquatte
                        continue
                    end
                    println("plaquatte: "*string(x_range)*", "*string(y_range))
                    H_env,H_log_coe=Ham_env_term(psi,psi_double, px,py, x_range,y_range,  data_down_move,data_up_move);
                    H_env_total=H_env_total+H_env;
                end
            end
            
            eu,ev=H_eig_solve(H_env_total,Norm_env,A_double_open,px,py,U_L,U_D,U_R,U_U);


            # if (px==1)&(py==3)
            #     jldsave("problem.jld2"; psi);
            #     error("break");
            # end
            
            psi[px,py]=ev;
            A_double_new,_=construct_double_layer_pos(psi,psi,px,py);
            if py==Ly
                if px==1
                    A_double_new=permute(A_double_new,(2,1,));
                elseif 1<px<Lx
                    A_double_new=permute(A_double_new,(1,3,2,));
                end
            end
            
            psi_double[px,py]=A_double_new;
            mps_set_down_move[px,py]=psi_double[px,py];
            data_down_move["mps_set"]=mps_set_down_move;
            mps_set_up_move[px,py]=psi_double[px,py];
            data_up_move["mps_set"]=mps_set_up_move;
        end
    end

    return psi
end

function H_eig_solve(H_env_total,Norm_env,A_double_open,px,py,U_L,U_D,U_R,U_U)

    #test total energy
    if Rank(Norm_env)==3
        e=@tensor A_double_open[1,2,3]*H_env_total[1,2,3];
        Norm=@tensor A_double_open[1,2,3]*Norm_env[1,2,3];
    elseif Rank(Norm_env)==4
        e=@tensor A_double_open[1,2,3,4]*H_env_total[1,2,3,4];
        Norm=@tensor A_double_open[1,2,3,4]*Norm_env[1,2,3,4];
    elseif Rank(Norm_env)==5
        e=@tensor A_double_open[1,2,3,4,5]*H_env_total[1,2,3,4,5];
        Norm=@tensor A_double_open[1,2,3,4,5]*Norm_env[1,2,3,4,5];
    end
    println("original energy: "*string(e/Norm));

    eu,ev=solve_tensor(H_env_total,Norm_env,px,py,U_L,U_D,U_R,U_U);

    return eu,ev
end




function solve_tensor(H_env,Norm_env,px,py,U_L,U_D,U_R,U_U)
    global Lx,Ly
    global U_phy
    U_s_s=@ignore_derivatives unitary(fuse(U_phy'*U_phy), U_phy' ⊗ U_phy);


    if (px==1)
        if (py==1) #left_bot
            U_U_d=unitary(fuse(space(U_U,2)*space(U_s_s,3)), space(U_U,2)*space(U_s_s,3));
            @tensor H_env[:]:=H_env[-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor H_env[:]:=H_env[1,-3,-4]*U_R[-1,-2,1];
            H_env=permute(H_env,(1,3,),(2,4,));

            @tensor Norm_env[:]:=Norm_env[-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor Norm_env[:]:=Norm_env[1,-3,-4]*U_R[-1,-2,1];
            Norm_env=permute(Norm_env,(1,3,),(2,4,));

            H_env=H_env/norm(Norm_env);
            Norm_env=Norm_env/norm(Norm_env);

            H_eff=get_Heff(H_env,Norm_env);

            eu,ev=eigen(H_eff);
            #println(sort(real.(diag(eu.data.values[1]))));
            v0=TensorMap(randn,space(H_eff,1),space(H_eff,2)');
            v0=permute(v0,(1,2,));
            function fun_LD(H,v)
                @tensor v_new[:]:=H_eff[-1,-2,1,2]*v[1,2];
                return v_new
            end
            fun_ld(x)=fun_LD(H_eff,x);
            eu,ev=eigsolve(fun_ld, v0, 3,:LM,Arnoldi(krylovdim=20));
            eu=eu[1];
            ev=ev[1];
            @tensor A_new[:]:=ev[-1,1]*U_U_d[1,-2,-3];
            println("new energy: "*string(eu));
        elseif (1<py<Ly) #left
            U_U_d=unitary(fuse(space(U_U,2)*space(U_s_s,3)), space(U_U,2)*space(U_s_s,3));
            @tensor H_env[:]:=H_env[-2,-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor H_env[:]:=H_env[1,-3,-4,-5]*U_D[1,-1,-2];
            @tensor H_env[:]:=H_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            H_env=permute(H_env,(1,3,5,),(2,4,6,));

            @tensor Norm_env[:]:=Norm_env[-2,-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor Norm_env[:]:=Norm_env[1,-3,-4,-5]*U_D[1,-1,-2];
            @tensor Norm_env[:]:=Norm_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            Norm_env=permute(Norm_env,(1,3,5,),(2,4,6,));

            H_env=H_env/norm(Norm_env);
            Norm_env=Norm_env/norm(Norm_env);

            H_eff=get_Heff(H_env,Norm_env);

            eu,ev=eigen(H_eff);
            #println(sort(real.(diag(eu.data.values[1]))))
            v0=TensorMap(randn,space(H_eff,1)*space(H_eff,2),space(H_eff,3)');
            v0=permute(v0,(1,2,3,));
            function fun_Left(H,v)
                @tensor v_new[:]:=H_eff[-1,-2,-3,1,2,3]*v[1,2,3];
                return v_new
            end
            fun_left(x)=fun_Left(H_eff,x);
            eu,ev=eigsolve(fun_left, v0, 3,:LM,Arnoldi(krylovdim=20));
            eu=eu[1];
            ev=ev[1];
            @tensor A_new[:]:=ev[-1,-2,1]*U_U_d[1,-3,-4];
            println("new energy: "*string(eu));
        elseif (py==Ly) #left_top
            ###############
            H_env=permute(H_env,(2,1,3,));
            Norm_env=permute(Norm_env,(2,1,3,));
            ###############
            @tensor H_env[:]:=H_env[-1,-2,1]*U_s_s[1,-3,-4];
            @tensor H_env[:]:=H_env[1,-3,-4,-5]*U_D[1,-1,-2];
            @tensor H_env[:]:=H_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            H_env=permute(H_env,(1,3,5,),(2,4,6,));

            @tensor Norm_env[:]:=Norm_env[-1,-2,1]*U_s_s[1,-3,-4];
            @tensor Norm_env[:]:=Norm_env[1,-3,-4,-5]*U_D[1,-1,-2];
            @tensor Norm_env[:]:=Norm_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            Norm_env=permute(Norm_env,(1,3,5,),(2,4,6,));

            H_env=H_env/norm(Norm_env);
            Norm_env=Norm_env/norm(Norm_env);

            H_eff=get_Heff(H_env,Norm_env);

            eu,ev=eigen(H_eff);
            #println(sort(real.(diag(eu.data.values[1]))))
            v0=TensorMap(randn,space(H_eff,1)*space(H_eff,2),space(H_eff,3)');
            v0=permute(v0,(1,2,3,));
            function fun_LU(H,v)
                @tensor v_new[:]:=H_eff[-1,-2,-3,1,2,3]*v[1,2,3];
                return v_new
            end
            fun_lu(x)=fun_LU(H_eff,x);
            eu,ev=eigsolve(fun_lu, v0, 3,:LM,Arnoldi(krylovdim=20));
            eu=eu[1];
            ev=ev[1];
            A_new=ev;
            println("new energy: "*string(eu));
        end
    elseif (1<px<Lx)
        if (py==1) #bot
            U_U_d=unitary(fuse(space(U_U,2)*space(U_s_s,3)), space(U_U,2)*space(U_s_s,3));
            @tensor H_env[:]:=H_env[-1,-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor H_env[:]:=H_env[1,-3,-4,-5]*U_L[1,-1,-2];
            @tensor H_env[:]:=H_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            H_env=permute(H_env,(1,3,5,),(2,4,6,));

            @tensor Norm_env[:]:=Norm_env[-1,-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor Norm_env[:]:=Norm_env[1,-3,-4,-5]*U_L[1,-1,-2];
            @tensor Norm_env[:]:=Norm_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            Norm_env=permute(Norm_env,(1,3,5,),(2,4,6,));

            H_env=H_env/norm(Norm_env);
            Norm_env=Norm_env/norm(Norm_env);

            H_eff=get_Heff(H_env,Norm_env);

            eu,ev=eigen(H_eff);
            #println(sort(real.(diag(eu.data.values[1]))))
            v0=TensorMap(randn,space(H_eff,1)*space(H_eff,2),space(H_eff,3)');
            v0=permute(v0,(1,2,3,));
            function fun_Bot(H,v)
                @tensor v_new[:]:=H_eff[-1,-2,-3,1,2,3]*v[1,2,3];
                return v_new
            end
            fun_bot(x)=fun_Bot(H_eff,x);
            eu,ev=eigsolve(fun_bot, v0, 3,:LM,Arnoldi(krylovdim=20));
            eu=eu[1];
            ev=ev[1];
            @tensor A_new[:]:=ev[-1,-2,1]*U_U_d[1,-3,-4];
            println("new energy: "*string(eu));
        elseif (1<py<Ly) #bulk
            U_U_d=unitary(fuse(space(U_U,2)*space(U_s_s,3)), space(U_U,2)*space(U_s_s,3));
            @tensor H_env[:]:=H_env[-1,-2,-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            U_L_D=unitary(fuse(space(U_L,3)*space(U_D,3)), space(U_L,3)*space(U_D,3));
            @tensor H_env[:]:=H_env[5,6,-3,-4,-5]*U_L[5,1,2]*U_D[6,3,4]*U_L_D'[1,3,-1]*U_L_D[-2,2,4];
            @tensor H_env[:]:=H_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            H_env=permute(H_env,(1,3,5,),(2,4,6,));

            @tensor Norm_env[:]:=Norm_env[-1,-2,-3,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor Norm_env[:]:=Norm_env[5,6,-3,-4,-5]*U_L[5,1,2]*U_D[6,3,4]*U_L_D'[1,3,-1]*U_L_D[-2,2,4];
            @tensor Norm_env[:]:=Norm_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            Norm_env=permute(Norm_env,(1,3,5,),(2,4,6,));

            H_env=H_env/norm(Norm_env);
            Norm_env=Norm_env/norm(Norm_env);

            H_eff=get_Heff(H_env,Norm_env);

            eu,ev=eigen(H_eff);
            #println(sort(real.(diag(eu.data.values[1]))))
            v0=TensorMap(randn,space(H_eff,1)*space(H_eff,2),space(H_eff,3)');
            v0=permute(v0,(1,2,3,));
            function fun_Bulk(H,v)
                @tensor v_new[:]:=H_eff[-1,-2,-3,1,2,3]*v[1,2,3];
                return v_new
            end
            fun_bulk(x)=fun_Bulk(H_eff,x);
            eu,ev=eigsolve(fun_bulk, v0, 3,:LM,Arnoldi(krylovdim=20));
            eu=eu[1];
            ev=ev[1];
            @tensor A_new[:]:=ev[1,-3,2]*U_L_D[1,-1,-2]*U_U_d[2,-4,-5];
            println("new energy: "*string(eu));
        elseif (py==Ly) #top
            #################
            H_env=permute(H_env,(1,3,2,4,));
            Norm_env=permute(Norm_env,(1,3,2,4,));
            ##################
            @tensor H_env[:]:=H_env[-1,-2,-3,5]*U_s_s[5,-4,-5];
            U_L_D=unitary(fuse(space(U_L,3)*space(U_D,3)), space(U_L,3)*space(U_D,3));
            @tensor H_env[:]:=H_env[5,6,-3,-4,-5]*U_L[5,1,2]*U_D[6,3,4]*U_L_D'[1,3,-1]*U_L_D[-2,2,4];
            @tensor H_env[:]:=H_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            H_env=permute(H_env,(1,3,5,),(2,4,6,));

            @tensor Norm_env[:]:=Norm_env[-1,-2,-3,5]U_s_s[5,-4,-5];
            @tensor Norm_env[:]:=Norm_env[5,6,-3,-4,-5]*U_L[5,1,2]*U_D[6,3,4]*U_L_D'[1,3,-1]*U_L_D[-2,2,4];
            @tensor Norm_env[:]:=Norm_env[-1,-2,1,-5,-6]*U_R[-3,-4,1];
            Norm_env=permute(Norm_env,(1,3,5,),(2,4,6,));

            H_env=H_env/norm(Norm_env);
            Norm_env=Norm_env/norm(Norm_env);

            H_eff=get_Heff(H_env,Norm_env);

            eu,ev=eigen(H_eff);
            #println(sort(real.(diag(eu.data.values[1]))))
            v0=TensorMap(randn,space(H_eff,1)*space(H_eff,2),space(H_eff,3)');
            v0=permute(v0,(1,2,3,));
            function fun_Top(H,v)
                @tensor v_new[:]:=H_eff[-1,-2,-3,1,2,3]*v[1,2,3];
                return v_new
            end
            fun_top(x)=fun_Top(H_eff,x);
            eu,ev=eigsolve(fun_top, v0, 3,:LM,Arnoldi(krylovdim=20));
            eu=eu[1];
            ev=ev[1];
            @tensor A_new[:]:=ev[1,-3,-4]*U_L_D[1,-1,-2];
            println("new energy: "*string(eu));
        end
    elseif (px==Lx)
        if (py==1) #right_bot
            U_U_d=unitary(fuse(space(U_U,2)*space(U_s_s,3)), space(U_U,2)*space(U_s_s,3));
            @tensor H_env[:]:=H_env[-1,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor H_env[:]:=H_env[1,-4,-5]*U_L[1,-1,-2];
            H_env=permute(H_env,(1,3,),(2,4,));

            @tensor Norm_env[:]:=Norm_env[-1,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor Norm_env[:]:=Norm_env[1,-4,-5]*U_L[1,-1,-2];
            Norm_env=permute(Norm_env,(1,3,),(2,4,));

            H_env=H_env/norm(Norm_env);
            Norm_env=Norm_env/norm(Norm_env);

            H_eff=get_Heff(H_env,Norm_env);

            eu,ev=eigen(H_eff);
            #println(sort(real.(diag(eu.data.values[1]))))
            v0=TensorMap(randn,space(H_eff,1),space(H_eff,2)');
            v0=permute(v0,(1,2,));
            function fun_RD(H,v)
                @tensor v_new[:]:=H_eff[-1,-2,1,2]*v[1,2];
                return v_new
            end
            fun_rd(x)=fun_RD(H_eff,x);
            eu,ev=eigsolve(fun_rd, v0, 3,:LM,Arnoldi(krylovdim=20));
            eu=eu[1];
            ev=ev[1];
            @tensor A_new[:]:=ev[-1,1]*U_U_d[1,-2,-3];
            println("new energy: "*string(eu));
        elseif (1<py<Ly) #right
            U_U_d=unitary(fuse(space(U_U,2)*space(U_s_s,3)), space(U_U,2)*space(U_s_s,3));
            @tensor H_env[:]:=H_env[-1,-2,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor H_env[:]:=H_env[1,2,-5,-6]*U_L[1,-1,-2]*U_D[2,-3,-4];
            H_env=permute(H_env,(1,3,5,),(2,4,6,));

            @tensor Norm_env[:]:=Norm_env[-1,-2,5,6]*U_U[1,3,5]*U_s_s[6,2,4]*U_U_d'[1,2,-4]*U_U_d[-5,3,4];
            @tensor Norm_env[:]:=Norm_env[1,2,-5,-6]*U_L[1,-1,-2]*U_D[2,-3,-4];
            Norm_env=permute(Norm_env,(1,3,5,),(2,4,6,));

            H_env=H_env/norm(Norm_env);
            Norm_env=Norm_env/norm(Norm_env);

            H_eff=get_Heff(H_env,Norm_env);

            eu,ev=eigen(H_eff);
            #println(sort(real.(diag(eu.data.values[1]))))
            v0=TensorMap(randn,space(H_eff,1)*space(H_eff,2),space(H_eff,3)');
            v0=permute(v0,(1,2,3,));
            function fun_Right(H,v)
                @tensor v_new[:]:=H_eff[-1,-2,-3,1,2,3]*v[1,2,3];
                return v_new
            end
            fun_right(x)=fun_Right(H_eff,x);
            eu,ev=eigsolve(fun_right, v0, 3,:LM,Arnoldi(krylovdim=20));
            eu=eu[1];
            ev=ev[1];
            @tensor A_new[:]:=ev[-1,-2,1]*U_U_d[1,-3,-4];
            println("new energy: "*string(eu));
        elseif (py==Ly) #right_top
            @tensor H_env[:]:=H_env[-1,-2,5]*U_s_s[5,-4,-5];
            U_L_D=unitary(fuse(space(U_L,3)*space(U_D,3)), space(U_L,3)*space(U_D,3));
            @tensor H_env[:]:=H_env[5,6,-4,-5]*U_L[5,1,2]*U_D[6,3,4]*U_L_D'[1,3,-1]*U_L_D[-2,2,4];
            H_env=permute(H_env,(1,3,),(2,4,));

            @tensor Norm_env[:]:=Norm_env[-1,-2,5]U_s_s[5,-4,-5];
            @tensor Norm_env[:]:=Norm_env[5,6,-4,-5]*U_L[5,1,2]*U_D[6,3,4]*U_L_D'[1,3,-1]*U_L_D[-2,2,4];
            Norm_env=permute(Norm_env,(1,3,),(2,4,));

            H_env=H_env/norm(Norm_env);
            Norm_env=Norm_env/norm(Norm_env);

            H_eff=get_Heff(H_env,Norm_env);

            eu,ev=eigen(H_eff);
            #println(sort(real.(diag(eu.data.values[1]))))
            v0=TensorMap(randn,space(H_eff,1),space(H_eff,2)');
            v0=permute(v0,(1,2,));
            function fun_TR(H,v)
                @tensor v_new[:]:=H_eff[-1,-2,1,2]*v[1,2];
                return v_new
            end
            fun_tr(x)=fun_TR(H_eff,x);
            eu,ev=eigsolve(fun_tr, v0, 3,:LM,Arnoldi(krylovdim=20));
            eu=eu[1];
            ev=ev[1];
            @tensor A_new[:]:=ev[1,-3]*U_L_D[1,-1,-2];
            println("new energy: "*string(eu));
        end
    end
    return eu,A_new
end