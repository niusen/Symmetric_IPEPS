function build_A_kagome(x::Kagome_iPESS_immutable)
    B1=x.B1;
    B2=x.B2;
    B3=x.B3;
    Tup=x.Tup;
    Tdn=x.Tdn;
    @tensor PEPS_tensor[:] := B1[-1,1,-5]*B2[4,3,-6]*B3[-4,2,-7]*Tup[1,3,2]*Tdn[-3,4,-2];
    A_unfused=PEPS_tensor;

    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];
    return A_unfused,A_fused,U_phy
end

function build_A_kagome(x::Kagome_iPESS)
    B1=x.B1;
    B2=x.B2;
    B3=x.B3;
    Tup=x.Tup;
    Tdn=x.Tdn;
    @tensor PEPS_tensor[:] := B1[-1,1,-5]*B2[4,3,-6]*B3[-4,2,-7]*Tup[1,3,2]*Tdn[-3,4,-2];
    A_unfused=PEPS_tensor;

    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];
    return A_unfused,A_fused,U_phy
end

function initial_SU2_state(Vspace,init_statenm="nothing",init_noise=0)
    if init_statenm=="nothing" 
        global Lx,Ly
        println("Random initial state");flush(stdout);
        Vp=SU2Space(1/2=>1);
        state=Matrix{Kagome_iPESS}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                b1=TensorMap(randn,Vv*Vv,Vp);
                b2=TensorMap(randn,Vv*Vv,Vp);
                b3=TensorMap(randn,Vv*Vv,Vp);
                tup=TensorMap(randn,Vv',Vv*Vv);
                tdn=TensorMap(randn,Vv',Vv*Vv);
                b1=permute(b1,(1,2,3,));
                b2=permute(b2,(1,2,3,));
                b3=permute(b3,(1,2,3,));
                tup=permute(tup,(1,2,3,));
                tdn=permute(tdn,(1,2,3,));

                #state=define_tensor_group(b1,b2,b3,tup,tdn)
                state[cx,cy]=Kagome_iPESS(Ba,Bb,Bc,Tu,Td);
            end
        end
        return state
    else
        
        println("load state: "*init_statenm);flush(stdout);
        x=load(init_statenm)["x"];
        state=similar(x);
        for cc in eachindex(x)
            ansatz=x[cc];
            B_a=ansatz.B1;
            B_b=ansatz.B2;
            B_c=ansatz.B3;
            T_u=ansatz.Tup;
            T_d=ansatz.Tdn;

            @assert space(B_a,1)==Vspace
            Ba_noise=TensorMap(randn,codomain(B_a),domain(B_a));
            Bb_noise=TensorMap(randn,codomain(B_b),domain(B_b));
            Bc_noise=TensorMap(randn,codomain(B_c),domain(B_c));
            Tu_noise=TensorMap(randn,codomain(T_u),domain(T_u));
            Td_noise=TensorMap(randn,codomain(T_d),domain(T_d));
            
            Ba=B_a+Ba_noise*init_noise*norm(B_a)/norm(Ba_noise);
            Bb=B_b+Bb_noise*init_noise*norm(B_b)/norm(Bb_noise);
            Bc=B_c+Bc_noise*init_noise*norm(B_c)/norm(Bc_noise);
            Tu=T_u+Tu_noise*init_noise*norm(T_u)/norm(Tu_noise);
            Td=T_d+Td_noise*init_noise*norm(T_d)/norm(Td_noise);
            ansatz_new=Kagome_iPESS(Ba,Bb,Bc,Tu,Td);

            state[cc]=ansatz_new;
        end

        


        return state
    end
end

function cost_fun(x::Matrix{T}) where T<:iPEPS_ansatz_immutable #variational parameters are vector of TensorMap
    global Lx,Ly,U_phy
    A_unfused_cell=initial_tuple_cell(Lx,Ly);
    A_fused_cell=initial_tuple_cell(Lx,Ly);

    for cx=1:Lx
        for cy=1:Ly
            global U_phy
            A_unfused,A_fused,U_phy=build_A_kagome(x[cx, cy]);
            A_unfused_cell=fill_tuple(A_unfused_cell, A_unfused, cx,cy);
            A_fused_cell=fill_tuple(A_fused_cell, A_fused, cx,cy);
        end
    end

    global chi, parameters, energy_setting, grad_ctm_setting
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

    CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_fused_cell,chi,init,[],grad_ctm_setting)
    E_total, E_up_cell, E_down_cell=evaluate_ob(parameters, U_phy, x, A_fused_cell, AA_fused_cell, CTM_cell, grad_ctm_setting, energy_setting.kagome_method, energy_setting.E_up_method, energy_setting.E_dn_method);
    E=E_total/3/(Lx*Ly);
    #println(E)
    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM_cell);
    E_tem=deepcopy(E)
    return E
end

function cost_fun_test(x::Matrix{T}) where T<:iPEPS_ansatz #variational parameters are vector of TensorMap
    global Lx,Ly,U_phy
    A_unfused_cell=initial_tuple_cell(Lx,Ly);
    A_fused_cell=initial_tuple_cell(Lx,Ly);

    for cx=1:Lx
        for cy=1:Ly
            global U_phy
            A_unfused,A_fused,U_phy=build_A_kagome(x[cx, cy]);
            A_unfused_cell=fill_tuple(A_unfused_cell, A_unfused, cx,cy);
            A_fused_cell=fill_tuple(A_fused_cell, A_fused, cx,cy);
        end
    end

    global chi, parameters, energy_setting, grad_ctm_setting
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

    CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_fused_cell,chi,init,[],grad_ctm_setting)
    E_total, E_up_cell, E_down_cell=evaluate_ob(parameters, U_phy, x, A_fused_cell, AA_fused_cell, CTM_cell, grad_ctm_setting, energy_setting.kagome_method, energy_setting.E_up_method, energy_setting.E_dn_method);
    E=E_total/3/(Lx*Ly);
    #println(E)
    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM_cell);
    E_tem=deepcopy(E)
    return E
end



function energy_CTM(x, chi, parameters, ctm_setting, energy_setting, init, init_CTM)
    global Lx,Ly,U_phy

    A_unfused_cell=initial_tuple_cell(Lx,Ly);
    A_fused_cell=initial_tuple_cell(Lx,Ly);

    for cx=1:Lx
        for cy=1:Ly
            global U_phy
        A_unfused,A_fused,U_phy=build_A_kagome(x[cx, cy]);
        A_unfused_cell=fill_tuple(A_unfused_cell, A_unfused, cx,cy);
        A_fused_cell=fill_tuple(A_fused_cell, A_fused, cx,cy);
        end
    end

    CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_fused_cell,chi,init, init_CTM, ctm_setting);

    #@assert ite_err<3*(1e-5)


    if (parameters["J2"]==0) & (parameters["J3"]==0)
        #kagome_method="E_single_triangle"
        E_total, E_up_cell, E_down_cell=evaluate_ob(parameters, U_phy, x, A_fused_cell, AA_fused_cell, CTM_cell, ctm_setting, energy_setting.kagome_method, energy_setting.E_up_method, energy_setting.E_dn_method);
        energy=E_total/3/(Lx*Ly);
    elseif parameters["Jtrip"]==0
        #kagome_method="E_bond"
        E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23,   E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b,  E_NNNN_11,E_NNNN_22,E_NNNN_33=evaluate_ob(parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        E_NN=parameters["J1"]*(E_up_12+E_up_31+E_up_23+E_down_12+E_down_31+E_down_23);
        E_NNN=parameters["J2"]*(E_NNN_23a+E_NNN_12a+E_NNN_31a+E_NNN_23b+E_NNN_12b+E_NNN_31b);
        E_NNNN=parameters["J3"]*(E_NNNN_11+E_NNNN_22+E_NNNN_33);
        energy=(E_NN+E_NNN+E_NNNN)/3;
        println(real([E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23]))
        println(real([E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b]))
        println(real([E_NNNN_11,E_NNNN_22,E_NNNN_33]))
    else
        #kagome_method="E_triangle";
        E_up, E_down=evaluate_ob(parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        #kagome_method="E_bond";
        E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23,   E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b,  E_NNNN_11,E_NNNN_22,E_NNNN_33=evaluate_ob(parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        E_NN=parameters["J1"]*(E_up_12+E_up_31+E_up_23+E_down_12+E_down_31+E_down_23);
        E_NNN=parameters["J2"]*(E_NNN_23a+E_NNN_12a+E_NNN_31a+E_NNN_23b+E_NNN_12b+E_NNN_31b);
        E_NNNN=parameters["J3"]*(E_NNNN_11+E_NNNN_22+E_NNNN_33);
        energy=(E_up+E_down)/3+(E_NNN+E_NNNN)/3;
    end


    #return energy,CTM,U_L,U_D,U_R,U_U
    if energy_setting.cal_chiral_order
        chiral_order_parameters=Dict([("J1", 0), ("J2", 0), ("J3", 0), ("Jchi", 0), ("Jtrip", 1)]);
        chiral_order_up, chiral_order_down=evaluate_ob(chiral_order_parameters, U_phy, x, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM,  ctm_setting, "E_triangle");
        return real(energy),chiral_order_up, chiral_order_down,ite_num,ite_err,CTM_cell
    else
        chiral_order_up=[];
        chiral_order_down=[];
        return real(energy),real(E_up_cell), real(E_down_cell),ite_num,ite_err,CTM_cell
    end
    
end




