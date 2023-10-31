function get_grad(x)
    ∂E = cost_fun'(x)
    #E=fun(state_vec)
    global E_tem, CTM_tem
    
    if isa(∂E, Tuple)
        for elem in ∂E
            @assert !isnan(norm(elem))
        end
    elseif isa(∂E, Vector)
        @assert !isnan(norm(∂E))
    end
    
    return E_tem,∂E,CTM_tem
end

function FD(state_vec)

    dt=0.0001

    E0=cost_fun(state_vec);

    grad=Vector{Float64}(undef,0);

    for cc=1:length(state_vec)
        state_vec_tem=deepcopy(state_vec);
        state_vec_tem[cc]=state_vec_tem[cc]+dt;
        grad=vcat(grad,(cost_fun(state_vec_tem)-E0)/dt);

    end

    return E0, grad

end



function cost_fun(x :: Vector) #variational parameters are coefficients of elementary tensors
    global chi, parameters, ipess_irrep, elementary_tensors, energy_setting, grad_ctm_setting
    #Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=vector_to_coes(elementary_tensors, ipess_irrep, x);

    bond_tensor,triangle_tensor=construct_su2_PG_IPESS_vec(x,elementary_tensors, ipess_irrep);
    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;

    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

    # norm_A=norm(A_fused)
    # A_fused= A_fused/norm_A;
    #CTM, AA_fused, U_L,U_D,U_R,U_U=init_CTM(chi,A_fused,"PBC",true,true);
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,[],grad_ctm_setting)
    E_up, E_down=evaluate_ob(parameters, U_phy, [], A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, grad_ctm_setting, energy_setting.kagome_method, energy_setting.E_up_method);
    E=real(E_up+E_down)/3;
    #println(E)
    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM);
    E_tem=deepcopy(E)
    return E
end




function energy_CTM(ipess_irrep, elementary_tensors, state, chi, parameters, ctm_setting, energy_setting, init, init_CTM)

    bond_tensor,triangle_tensor=construct_su2_PG_IPESS_vec(state, elementary_tensors, ipess_irrep);
    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;

    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];




    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init, init_CTM, ctm_setting);

    #@assert ite_err<3*(1e-5)

    if (parameters["J2"]==0) & (parameters["J3"]==0)
        #kagome_method="E_single_triangle"
        E_up, E_down=evaluate_ob(parameters, U_phy, [], A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method, energy_setting.E_up_method);
        energy=(E_up+E_down)/3;
    elseif parameters["Jtrip"]==0
        #kagome_method="E_bond"
        E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23,   E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b,  E_NNNN_11,E_NNNN_22,E_NNNN_33=evaluate_ob(parameters, U_phy, [], A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        E_NN=parameters["J1"]*(E_up_12+E_up_31+E_up_23+E_down_12+E_down_31+E_down_23);
        E_NNN=parameters["J2"]*(E_NNN_23a+E_NNN_12a+E_NNN_31a+E_NNN_23b+E_NNN_12b+E_NNN_31b);
        E_NNNN=parameters["J3"]*(E_NNNN_11+E_NNNN_22+E_NNNN_33);
        energy=(E_NN+E_NNN+E_NNNN)/3;
        println(real([E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23]))
        println(real([E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b]))
        println(real([E_NNNN_11,E_NNNN_22,E_NNNN_33]))
    else
        #kagome_method="E_triangle";
        E_up, E_down=evaluate_ob(parameters, U_phy, [], A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        #kagome_method="E_bond";
        E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23,   E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b,  E_NNNN_11,E_NNNN_22,E_NNNN_33=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        E_NN=parameters["J1"]*(E_up_12+E_up_31+E_up_23+E_down_12+E_down_31+E_down_23);
        E_NNN=parameters["J2"]*(E_NNN_23a+E_NNN_12a+E_NNN_31a+E_NNN_23b+E_NNN_12b+E_NNN_31b);
        E_NNNN=parameters["J3"]*(E_NNNN_11+E_NNNN_22+E_NNNN_33);
        energy=(E_up+E_down)/3+(E_NNN+E_NNNN)/3;
    end


    #return energy,CTM,U_L,U_D,U_R,U_U
    if energy_setting.cal_chiral_order
        chiral_order_parameters=Dict([("J1", 0), ("J2", 0), ("J3", 0), ("Jchi", 0), ("Jtrip", 1)]);
        chiral_order_up, chiral_order_down=evaluate_ob(chiral_order_parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM,  ctm_setting, "E_triangle");
        return energy,chiral_order_up, chiral_order_down,ite_num,ite_err,CTM
    else
        chiral_order_up=[];
        chiral_order_down=[];
        return energy,E_up, E_down,ite_num,ite_err,CTM
    end
    
end






function grad_line_search(E_old, state_vec, D, chi, parameters, optim_setting, LS_ctm_setting, energy_setting, grad0=None, direction0=None, alpha0=1, ls_ratio=1/3, ls_max=10)
    
    if nonchiral=="No"
        filenm="LS_D_"*string(D)*"_chi_"*string(chi)*".json"
    elseif nonchiral=="A1_even"
        filenm="LS_A1even_D_"*string(D)*"_chi_"*string(chi)*".json"
    elseif nonchiral=="A1_odd"
        filenm="LS_A1odd_D_"*string(D)*"_chi_"*string(chi)*".json"
    end
    global elementary_tensors, ipess_irrep;
    state_vec=normalize_IPESS_SU2_PG_vec(elementary_tensors, ipess_irrep, state_vec);

    global E_tem, CTM_tem
    @time E0_, grad,CTM_tem=get_grad(state_vec);

    if E_old==[]
        E0=E0_;
    else
        E0=E_old;#if there is only a few CTM steps are used for obtaining grad, then the energy from the last line search is more reliable
    end

    global grad_norm
    println("Norm of dA: "*string(grad_norm))

    
    println("state: "*string(state_vec));flush(stdout);
    println("grad: "*string(grad));flush(stdout);

    if optim_setting.linesearch_CTM_method=="from_converged_CTM"
        init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
        CTM0=deepcopy(CTM_tem);
    elseif optim_setting.linesearch_CTM_method=="restart"
        init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
        CTM0=[];
    end


    global LS_ctm_setting

    
    

    direction=-grad
    #print(grad0);flush(stdout);
    #print(grad);flush(stdout);
    if grad0==nothing
        direction=-grad;
    else
        norm_grad=norm(grad)
        norm_grad0=norm(grad0)
        beta=(norm_grad^2)/(norm_grad0^2)
        direction=-grad+beta*direction0;
    end
    vec0=deepcopy(state_vec);
    vec_tem=[];

    #line search
    improved=false
    alpha=alpha0
    println("line search");flush(stdout);
    println("E,chiral_order_up, chiral_order_down,ite_num,ite_err")
    println("conjugate gradient opt");flush(stdout);
    for ls_step=0:ls_max-1
        vec_tem=vec0+direction*alpha*(ls_ratio^ls_step);
        E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(ipess_irrep, elementary_tensors, vec_tem, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0);
        #println("Number of iterations for linesearch: "*string(ite_num));
        E=real(E);
        println(string(E)*", "*string(real(chiral_order_up))*", "*string(real(chiral_order_down))*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
        if E<E0
            improved=true
            break
        end
    end
    if improved
        vec=deepcopy(vec_tem);
        global json_dict
        state=set_vector(deepcopy(json_dict), vec);
        open(filenm,"w") do f
            JSON.print(f, state)
        end
    else
        println("gradient opt");flush(stdout);
        for ls_step = 0:ls_max-1
            vec_tem=vec0-grad*alpha*(ls_ratio^ls_step)

            
            E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(ipess_irrep, elementary_tensors, vec_tem, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0);
            #println("Number of iterations for linesearch: "*string(ite_num));
            E=real(E);
            println(string(E)*", "*string(real(chiral_order_up))*", "*string(real(chiral_order_down))*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
            if E<E0
                improved=true
                break
            end
        end
        if improved
            vec=deepcopy(vec_tem);
            global json_dict
            state=set_vector(deepcopy(json_dict), vec);
            open(filenm,"w") do f
                JSON.print(f, state)
            end
        else
            vec=deepcopy(vec0)
            E=deepcopy(E0)
        end
    end
    improvement=E-E0
    
    return E,vec,grad,direction,improvement
end



function run_FiniteDiff(parameters, D, chi, ipess_irrep, LS_ctm_setting, optim_setting, energy_setting)
    
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    if energy_setting.kagome_method =="E_single_triangle"
        println("Only compute energy in a single triangle");
    end

    A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
    global json_dict
    json_dict, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd=initial_state(ipess_irrep.Bond_irrep, ipess_irrep.Triangle_irrep, ipess_irrep.nonchiral, D, optim_setting.init_statenm,optim_setting.init_noise)
    
    global A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb 
    global ipess_irrep, elementary_tensors 
    elementary_tensors=Elementary_tensors(A_set,B_set,A1_set,A2_set,A1_has_odd,A2_has_odd);


    state_vec=coes_to_vector(Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, ipess_irrep)
    state_vec=normalize_IPESS_SU2_PG_vec(elementary_tensors, ipess_irrep, state_vec);



    println("optimization start");flush(stdout);
    grad=nothing;
    direction=nothing;
    alpha0=3;
    ls_ratio=1/3;
    ls_max=5;

    E0=[];
    for ite=1:100
        E,state_vec,grad,direction,improvement=grad_line_search(E0, state_vec, D, chi, parameters, optim_setting, LS_ctm_setting, energy_setting, grad, direction, alpha0, ls_ratio, ls_max)
        E0=deepcopy(E);
        println("grad norm: "*string(norm(grad)));flush(stdout)
        if -improvement<1e-7
            break
        end
    end

end

