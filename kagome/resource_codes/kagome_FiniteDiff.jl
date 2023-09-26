
function energy_CTM(D,chi,parameters,state_dict, ctm_setting, energy_setting, init)
    global A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb
    
    bond_tensor,triangle_tensor=construct_su2_PG_IPESS(state_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);

    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;
    
    U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,ctm_setting);

    @assert ite_err<3*(1e-5)

    if (parameters["J2"]==0) & (parameters["J3"]==0)
        #kagome_method="E_single_triangle"
        E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        energy=(E_up+E_down)/3;
    elseif parameters["Jtrip"]==0
        #kagome_method="E_bond"
        E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23,   E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b,  E_NNNN_11,E_NNNN_22,E_NNNN_33=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
        E_NN=parameters["J1"]*(E_up_12+E_up_31+E_up_23+E_down_12+E_down_31+E_down_23);
        E_NNN=parameters["J2"]*(E_NNN_23a+E_NNN_12a+E_NNN_31a+E_NNN_23b+E_NNN_12b+E_NNN_31b);
        E_NNNN=parameters["J3"]*(E_NNNN_11+E_NNNN_22+E_NNNN_33);
        energy=(E_NN+E_NNN+E_NNNN)/3;
        println(real([E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23]))
        println(real([E_NNN_23a, E_NNN_12a, E_NNN_31a,E_NNN_23b, E_NNN_12b, E_NNN_31b]))
        println(real([E_NNNN_11,E_NNNN_22,E_NNNN_33]))
    else
        #kagome_method="E_triangle";
        E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, ctm_setting, energy_setting.kagome_method);
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
        
    else
        chiral_order_up=[];
        chiral_order_down=[];
        
    end
    return energy,chiral_order_up, chiral_order_down,ite_num,ite_err,CTM
end



function Grad_FiniteDiff(state, nonchiral, A1_has_odd, A2_has_odd, D, chi, parameters, ctm_setting, grad_CTM_method, energy_setting, dt=0.001, E0=nothing)

    state=normalize_IPESS_SU2_PG(state);
    #print(E0);flush(stdout);

    init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);    
    E0,chiral_order_up, chiral_order_down,ite_num,ite_err,CTM=energy_CTM(D,chi,parameters, state, ctm_setting, energy_setting, init_CTM);
    E0=real(E0);
    
    grad_ctm_setting=deepcopy(ctm_setting);
    grad_ctm_setting.CTM_ite_info=false;

    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(state);

    println("energy E0 is "*string(E0));flush(stdout);

    Grad_FD=Dict([("Bond_A_coe", zeros(Float64, length(Bond_A_coe))), ("Bond_B_coe", zeros(Float64, length(Bond_B_coe))), ("Triangle_A1_coe", zeros(Float64, length(Triangle_A1_coe))),("Triangle_A2_coe", zeros(Float64, length(Triangle_A2_coe)))]);
    dE_data=[]
    Grad_FD_data=[]

    #println(state["coes"])

    if grad_CTM_method=="restart"
        init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
    elseif grad_CTM_method=="from_converged_CTM"
        init_CTM=Dict([("CTM", deepcopy(CTM)), ("init_type", "PBC")]);
    end

    #Bond A tensor diff
    if Bond_irrep in ["A","A+iB"]
        Bond_A_grad=zeros(Float64, length(Bond_A_coe))
        for ct =1:length(Bond_A_coe)
            Bond_A_coe_tem=deepcopy(Bond_A_coe);
            Bond_A_coe_tem[ct]=Bond_A_coe_tem[ct]+dt;
            state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe_tem, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe);
            #println(state_tem["coes"])

            E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(D,chi,parameters, state_tem, grad_ctm_setting, energy_setting, deepcopy(init_CTM));
            println("Number of iterations for grad: "*string(ite_num));flush(stdout);
            E=real(E);
            Bond_A_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        #print(Bond_A_grad);flush(stdout);
        Grad_FD["Bond_A_grad"]=Bond_A_grad;
        Grad_FD_data=vcat(Grad_FD_data, Bond_A_grad);
    end

    #Bond B tensor diff
    if Bond_irrep in ["B","A+iB"]
        Bond_B_grad=zeros(Float64, length(Bond_B_coe))
        for ct=1:length(Bond_B_coe)
            Bond_B_coe_tem=deepcopy(Bond_B_coe);
            Bond_B_coe_tem[ct]=Bond_B_coe_tem[ct]+dt
            state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe_tem, Triangle_A1_coe, Triangle_A2_coe);
            #println(state_tem["coes"])

            E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(D,chi,parameters, state_tem, grad_ctm_setting, energy_setting, deepcopy(init_CTM));
            println("Number of iterations for grad: "*string(ite_num));flush(stdout);
            E=real(E);
            Bond_B_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        #print(Bond_B_grad);flush(stdout);
        Grad_FD["Bond_B_grad"]=Bond_B_grad;
        Grad_FD_data=vcat(Grad_FD_data, Bond_B_grad);
    end

    #triangle A1 tensor diff
    if Triangle_irrep in ["A1","A1+iA2"]
        Triangle_A1_grad=zeros(Float64, length(Triangle_A1_coe))
        for ct=1:length(Triangle_A1_coe)
            if (nonchiral=="No") | ((nonchiral=="A1_even")&(A1_has_odd[ct]==0)) | ((nonchiral=="A1_odd")&(A1_has_odd[ct]==1))
                Triangle_A1_coe_tem=deepcopy(Triangle_A1_coe);
                Triangle_A1_coe_tem[ct]=Triangle_A1_coe_tem[ct]+dt
                state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe_tem, Triangle_A2_coe);
                #println(state_tem["coes"])
                
                E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(D,chi,parameters, state_tem, grad_ctm_setting, energy_setting, deepcopy(init_CTM));
                println("Number of iterations for grad: "*string(ite_num));flush(stdout);
                E=real(E);
                Triangle_A1_grad[ct]=(E-E0)/dt;
                dE_data=vcat(dE_data, E-E0);
                #println("energy is "*string(E));flush(stdout);
            elseif ((nonchiral=="A1_even")&(A1_has_odd[ct]==1)) | ((nonchiral=="A1_odd")&(A1_has_odd[ct]==0))
                Triangle_A1_grad[ct]=0;
                dE_data=vcat(dE_data, 0);
            else
                error("incorrect type 'nonchiral'")
            end
        end
        #print(Triangle_A1_grad);flush(stdout);
        Grad_FD["Triangle_A1_grad"]=Triangle_A1_grad;
        Grad_FD_data=vcat(Grad_FD_data, Triangle_A1_grad);
    end

    #triangle A2 tensor diff
    if Triangle_irrep in ["A2","A1+iA2"]
        Triangle_A2_grad=zeros(Float64, length(Triangle_A2_coe))
        for ct=1:length(Triangle_A2_coe)
            if (nonchiral=="No") | ((nonchiral=="A1_even")&(A2_has_odd[ct]==1)) | ((nonchiral=="A1_odd")&(A2_has_odd[ct]==0))
                Triangle_A2_coe_tem=deepcopy(Triangle_A2_coe);
                Triangle_A2_coe_tem[ct]=Triangle_A2_coe_tem[ct]+dt
                state_tem=wrap_json_state(Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe_tem);
                #println(state_tem["coes"])
                
                E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(D,chi,parameters, state_tem, grad_ctm_setting, energy_setting, deepcopy(init_CTM));
                println("Number of iterations for grad: "*string(ite_num));flush(stdout);
                E=real(E);
                Triangle_A2_grad[ct]=(E-E0)/dt;
                dE_data=vcat(dE_data, E-E0);
                #println("energy is "*string(E));flush(stdout);
            elseif ((nonchiral=="A1_even")&(A2_has_odd[ct]==0)) | ((nonchiral=="A1_odd")&(A2_has_odd[ct]==1))
                Triangle_A2_grad[ct]=0;
                dE_data=vcat(dE_data, 0);
            else
                error("incorrect type 'nonchiral'")
            end
        end
        #print(Triangle_A2_grad)
        Grad_FD["Triangle_A2_grad"]=Triangle_A2_grad;
        Grad_FD_data=vcat(Grad_FD_data, Triangle_A2_grad);
    end


    return E0,Grad_FD,Grad_FD_data,CTM
end



function grad_line_search(state, nonchiral, A1_has_odd, A2_has_odd, D, chi, parameters, ctm_setting, grad_CTM_method,linesearch_CTM_method, energy_setting, dt, E0, grad0=None, direction0=None, alpha0=1, ls_ratio=1/3, ls_max=10)
    
    if nonchiral=="No"
        filenm="LS_D_"*string(D)*"_chi_"*string(chi)*".json"
    elseif nonchiral=="A1_even"
        filenm="LS_A1even_D_"*string(D)*"_chi_"*string(chi)*".json"
    elseif nonchiral=="A1_odd"
        filenm="LS_A1odd_D_"*string(D)*"_chi_"*string(chi)*".json"
    end
    
    state=normalize_IPESS_SU2_PG(state)
    
    E0,_,grad,CTM=Grad_FiniteDiff(state, nonchiral, A1_has_odd, A2_has_odd, D, chi, parameters, ctm_setting, grad_CTM_method, energy_setting, dt, E0)
    
    println("state: "*string(get_vector(state)));flush(stdout);
    println("grad: "*string(grad));flush(stdout);

    if linesearch_CTM_method=="restart"
        init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
    elseif linesearch_CTM_method=="from_converged_CTM"
        init_CTM=Dict([("CTM", deepcopy(CTM)), ("init_type", "PBC")]);
    end

    LS_ctm_setting=deepcopy(ctm_setting);
    LS_ctm_setting.CTM_ite_info=false;


    E=E0;
    Bond_irrep, Triangle_irrep, nonchiral, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe=get_tensor_coes(state)

    println("E0= "*string(E0));flush(stdout);

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
    vec0=deepcopy(get_vector(state));
    vec_tem=[];

    #line search
    improved=false
    alpha=alpha0
    println("line search");flush(stdout);
    println("E,chiral_order_up, chiral_order_down,ite_num,ite_err")
    println("conjugate gradient opt");flush(stdout);
    for ls_step=0:ls_max-1
        vec_tem=vec0+direction*alpha*(ls_ratio^ls_step);
        state_tem=set_vector(state, vec_tem)

        E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(D,chi,parameters,state_tem, LS_ctm_setting, energy_setting, init_CTM);

        #println("Number of iterations for linesearch: "*string(ite_num));
        
        E=real(E);
        println(string(E)*", "*string(real(chiral_order_up))*", "*string(real(chiral_order_down))*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
        if E<E0
            improved=true
            break
        end
    end
    if improved
        state=set_vector(state, vec_tem)

        open(filenm,"w") do f
            JSON.print(f, state)
        end
    else
        println("gradient opt");flush(stdout);
        for ls_step = 0:ls_max-1
            vec_tem=vec0-grad*alpha*(ls_ratio^ls_step)
            state_tem=set_vector(state, vec_tem)
            
            E,chiral_order_up, chiral_order_down,ite_num,ite_err,_=energy_CTM(D,chi,parameters,state_tem, LS_ctm_setting, energy_setting, init_CTM);

            #println("Number of iterations for linesearch: "*string(ite_num));
            E=real(E);
            println(string(E)*", "*string(real(chiral_order_up))*", "*string(real(chiral_order_down))*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
            if E<E0
                improved=true
                break
            end
        end
    
            
        if improved
            state=set_vector(state, vec_tem)
            open(filenm,"w") do f
                JSON.print(f, state)
            end
        else
            state=set_vector(state, vec0)
            E=E0
        end
    end
    improvement=E-E0
    
    open(filenm,"w") do f
        JSON.print(f, state)
    end
    return E,state,grad,direction,improvement
end



function run_FiniteDiff(parameters,D,chi,Bond_irrep,Triangle_irrep,nonchiral,ctm_setting,optim_setting,energy_setting)
    
    # multi_threads=true;if Threads.nthreads()==1; multi_threads=false; end
    # println("number of threads: "*string(Threads.nthreads()));flush(stdout);
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    println("Bond_irrep: "*Bond_irrep);flush(stdout);
    println("nonchiral: "*nonchiral);flush(stdout);
    if energy_setting.kagome_method =="E_single_triangle"
        println("Only compute energy in a single triangle");
    end
    init_statenm=optim_setting.init_statenm;
    init_noise=optim_setting.init_noise;
    grad_CTM_method=optim_setting.grad_CTM_method;
    linesearch_CTM_method=optim_setting.linesearch_CTM_method;
    state, Bond_A_coe, Bond_B_coe, Triangle_A1_coe, Triangle_A2_coe, A1_has_odd, A2_has_odd=initial_state(Bond_irrep, Triangle_irrep, nonchiral, D,init_statenm,init_noise)


    println("optimization start");flush(stdout);
    #E0,_,_=Grad_FiniteDiff(state, cfg.ctm_args, args.chi)
    dt=0.001;
    grad=nothing;
    direction=nothing;
    alpha0=3;
    ls_ratio=1/3;
    ls_max=5;
    E0=nothing;
    nonchiral=nonchiral;
    for ite=1:100
        
        @time E0,state,grad,direction,improvement=grad_line_search(state, nonchiral,A1_has_odd, A2_has_odd, D, chi, parameters, ctm_setting, grad_CTM_method,linesearch_CTM_method, energy_setting, dt, E0, grad, direction, alpha0, ls_ratio, ls_max)
        println("grad norm: "*string(norm(grad)));flush(stdout)
        if -improvement<1e-7
            break
        end
    end

end

