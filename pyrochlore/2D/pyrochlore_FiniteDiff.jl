
function energy_CTM(H_plaquatte,D,chi,parameters,state_dict,ctm_setting,init)

    A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
    
    bond_tensor,square_tensor=construct_su2_PG_IPESS(state_dict,A_set,A1_set,A2_set,B1_set,B2_set, S_label, Sz_label, virtual_particle, Va, Vb);

    PEPS_tensor,A_fused,U_phy=build_PEPS(bond_tensor,square_tensor);

    conv_check="singular_value";
    CTM_ite_info=false;
    CTM_conv_info=true;
    
    CTM, AA_fused, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A_fused,chi,init,ctm_setting);

    #rho=build_density_op(U_phy, PEPS_tensor, AA_fused, U_L,U_D,U_R,U_U, CTM);#L',U',R',D',  L,U,R,D
    #energy=plaquatte_ob(rho,Sigma)

    energy=ob_efficient(H_plaquatte, U_phy, AA_fused, CTM,bond_tensor,square_tensor);#L',U',R',D',  L,U,R,D

    


    #return energy,CTM,U_L,U_D,U_R,U_U
    return energy,ite_num,ite_err,CTM

end








function Grad_FiniteDiff(H_plaquatte,state, D, chi, parameters,ctm_setting,grad_CTM_method, dt=0.001, E0=nothing)

    state=normalize_IPESS_SU2_PG(state);
    #print(E0);flush(stdout);

    init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
    E0,ite_num,ite_err,CTM=energy_CTM(H_plaquatte,D,chi,parameters,state,ctm_setting,init_CTM);
    E0=real(E0);
    
    grad_ctm_setting=deepcopy(ctm_setting);
    grad_ctm_setting.CTM_ite_info=false;


    Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=get_tensor_coes(state);

    println("energy E0 is "*string(E0));flush(stdout);

    Grad_FD=Dict([("Bond_A_coe", zeros(Float64, length(Bond_A_coe))), ("Square_A1_coe", zeros(Float64, length(Square_A1_coe))),("Square_A2_coe", zeros(Float64, length(Square_A2_coe))),("Square_B1_coe", zeros(Float64, length(Square_B1_coe))),("Square_B2_coe", zeros(Float64, length(Square_B2_coe)))]);
    dE_data=[]
    Grad_FD_data=[]

    #println(state["coes"])

    #Bond A tensor diff
    if Bond_irrep=="A"
        Bond_A_grad=zeros(Float64, length(Bond_A_coe))
        for ct =1:length(Bond_A_coe)
            Bond_A_coe_tem=deepcopy(Bond_A_coe);
            Bond_A_coe_tem[ct]=Bond_A_coe_tem[ct]+dt;
            state_tem=wrap_json_state(Bond_irrep, Square_irrep, Bond_A_coe_tem, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe);
            #println(state_tem["coes"])
            if grad_CTM_method=="restart"
                init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
            elseif grad_CTM_method=="from_converged_CTM"
                init_CTM=Dict([("CTM", deepcopy(CTM)), ("init_type", "PBC")]);
            end
            E,ite_num,ite_err,_=energy_CTM(H_plaquatte,D,chi,parameters,state_tem,grad_ctm_setting,init_CTM);
            println("Number of iterations for grad: "*string(ite_num));
            E=real(E);
            Bond_A_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        #print(Bond_A_grad);flush(stdout);
        Grad_FD["Bond_A_grad"]=Bond_A_grad;
        Grad_FD_data=vcat(Grad_FD_data, Bond_A_grad);
    end


    #Square A1 tensor diff
    if Square_irrep=="A1"
        Square_A1_grad=zeros(Float64, length(Square_A1_coe))
        for ct=1:length(Square_A1_coe)
            Square_A1_coe_tem=deepcopy(Square_A1_coe);
            Square_A1_coe_tem[ct]=Square_A1_coe_tem[ct]+dt
            state_tem=wrap_json_state(Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe_tem, Square_A2_coe, Square_B1_coe, Square_B2_coe);
            #println(state_tem["coes"])
            if grad_CTM_method=="restart"
                init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
            elseif grad_CTM_method=="from_converged_CTM"
                init_CTM=Dict([("CTM", deepcopy(CTM)), ("init_type", "PBC")]);
            end
            E,ite_num,ite_err,_=energy_CTM(H_plaquatte,D,chi,parameters,state_tem,grad_ctm_setting,init_CTM);
            println("Number of iterations for grad: "*string(ite_num));
            E=real(E);
            Square_A1_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        #print(Square_A1_grad);flush(stdout);
        Grad_FD["Square_A1_grad"]=Square_A1_grad;
        Grad_FD_data=vcat(Grad_FD_data, Square_A1_grad);
    elseif Square_irrep=="A2"
        Square_A2_grad=zeros(Float64, length(Square_A2_coe))
        for ct=1:length(Square_A2_coe)
            Square_A2_coe_tem=deepcopy(Square_A2_coe);
            Square_A2_coe_tem[ct]=Square_A2_coe_tem[ct]+dt
            state_tem=wrap_json_state(Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe_tem, Square_B1_coe, Square_B2_coe);
            #println(state_tem["coes"])
            if grad_CTM_method=="restart"
                init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
            elseif grad_CTM_method=="from_converged_CTM"
                init_CTM=Dict([("CTM", deepcopy(CTM)), ("init_type", "PBC")]);
            end
            E,ite_num,ite_err,_=energy_CTM(H_plaquatte,D,chi,parameters,state_tem,grad_ctm_setting,init_CTM);
            println("Number of iterations for grad: "*string(ite_num));
            E=real(E);
            Square_A2_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        Grad_FD["Square_A2_grad"]=Square_A2_grad;
        Grad_FD_data=vcat(Grad_FD_data, Square_A2_grad);

    elseif Square_irrep=="B1"
        Square_B1_grad=zeros(Float64, length(Square_B1_coe))
        for ct=1:length(Square_B1_coe)
            Square_B1_coe_tem=deepcopy(Square_B1_coe);
            Square_B1_coe_tem[ct]=Square_B1_coe_tem[ct]+dt
            state_tem=wrap_json_state(Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe_tem, Square_B2_coe);
            #println(state_tem["coes"])
            if grad_CTM_method=="restart"
                init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
            elseif grad_CTM_method=="from_converged_CTM"
                init_CTM=Dict([("CTM", deepcopy(CTM)), ("init_type", "PBC")]);
            end
            E,ite_num,ite_err,_=energy_CTM(H_plaquatte,D,chi,parameters,state_tem,grad_ctm_setting,init_CTM);
            println("Number of iterations for grad: "*string(ite_num));
            E=real(E);
            Square_B1_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        Grad_FD["Square_B1_grad"]=Square_B1_grad;
        Grad_FD_data=vcat(Grad_FD_data, Square_B1_grad);
    elseif Square_irrep=="B2"
        Square_B2_grad=zeros(Float64, length(Square_B2_coe))
        for ct=1:length(Square_B2_coe)
            Square_B2_coe_tem=deepcopy(Square_B2_coe);
            Square_B2_coe_tem[ct]=Square_B2_coe_tem[ct]+dt
            state_tem=wrap_json_state(Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe_tem);
            #println(state_tem["coes"])
            if grad_CTM_method=="restart"
                init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
            elseif grad_CTM_method=="from_converged_CTM"
                init_CTM=Dict([("CTM", deepcopy(CTM)), ("init_type", "PBC")]);
            end
            E,ite_num,ite_err,_=energy_CTM(H_plaquatte,D,chi,parameters,state_tem,grad_ctm_setting,init_CTM);
            println("Number of iterations for grad: "*string(ite_num));
            E=real(E);
            Square_B2_grad[ct]=(E-E0)/dt;
            dE_data=vcat(dE_data, E-E0);
            #println("energy is "*string(E));flush(stdout);
        end
        Grad_FD["Square_B2_grad"]=Square_B2_grad;
        Grad_FD_data=vcat(Grad_FD_data, Square_B2_grad);

    end

    return E0,Grad_FD,Grad_FD_data,CTM
end



function grad_line_search(H_plaquatte,state, D, chi, parameters, ctm_setting, grad_CTM_method, linesearch_CTM_method, dt, E0, grad0=None, direction0=None, alpha0=1, ls_ratio=1/3, ls_max=10)
    

    filenm="LS_D_"*string(D)*"_chi_"*string(chi)*".json"

   
    
    state=normalize_IPESS_SU2_PG(state)
    
    E0,_,grad,CTM=Grad_FiniteDiff(H_plaquatte,state, D, chi, parameters, ctm_setting, grad_CTM_method, dt, E0)
    
    println("state: "*string(get_vector(state)));flush(stdout);
    println("grad: "*string(grad));flush(stdout);

    LS_ctm_setting=deepcopy(ctm_setting);
    LS_ctm_setting.CTM_ite_info=false;

    E=E0;
    Bond_irrep, Square_irrep, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=get_tensor_coes(state)

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
    println("E, ite_num, ite_err")
    println("conjugate gradient opt");flush(stdout);
    for ls_step=0:ls_max-1
        vec_tem=vec0+direction*alpha*(ls_ratio^ls_step);
        state_tem=set_vector(state, vec_tem)
        if linesearch_CTM_method=="restart"
            init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
        elseif linesearch_CTM_method=="from_converged_CTM"
            init_CTM=Dict([("CTM", deepcopy(CTM)), ("init_type", "PBC")]);
        end
        E,ite_num,ite_err,_=energy_CTM(H_plaquatte,D,chi,parameters,state_tem,LS_ctm_setting,init_CTM);
        println("Number of iterations for linesearch: "*string(ite_num));
        
        E=real(E);
        println(string(E)*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
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
            if linesearch_CTM_method=="restart"
                init_CTM=Dict([("CTM", []), ("init_type", "PBC")]);
            elseif linesearch_CTM_method=="from_converged_CTM"
                init_CTM=Dict([("CTM", deepcopy(CTM)), ("init_type", "PBC")]);
            end
            E,ite_num,ite_err,_=energy_CTM(H_plaquatte,D,chi,parameters,state_tem,LS_ctm_setting,init_CTM);
            println("Number of iterations for linesearch: "*string(ite_num));
            E=real(E);
            println(string(E)*", "*string(ite_num)*", "*string(ite_err));flush(stdout);
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



function run_FiniteDiff(parameters,D,chi,Bond_irrep,Square_irrep,ctm_setting,optim_setting)
    
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    println("Bond_irrep: "*Bond_irrep);flush(stdout);
    println("Square_irrep: "*Bond_irrep);flush(stdout);

    U_phy=unitary(Rep[SU₂](0=>1, 1=>1, 2=>1) ← (Rep[SU₂](1=>1)' ⊗ Rep[SU₂](1=>1)'));
    H_plaquatte=plaquatte_Heisenberg(parameters["J1"],parameters["J2"]);
    H_plaquatte=fuse_H(H_plaquatte,U_phy);

    init_statenm=optim_setting.init_statenm;
    init_noise=optim_setting.init_noise;
    grad_CTM_method=optim_setting.grad_CTM_method;
    linesearch_CTM_method=optim_setting.linesearch_CTM_method;
    state, Bond_A_coe, Square_A1_coe, Square_A2_coe, Square_B1_coe, Square_B2_coe=initial_state(Bond_irrep, Square_irrep, D, init_statenm,init_noise);

    println("optimization start");flush(stdout);
    #E0,_,_=Grad_FiniteDiff(state, cfg.ctm_args, args.chi)
    dt=0.001;
    grad=nothing;
    direction=nothing;
    alpha0=3;
    ls_ratio=1/3;
    ls_max=5;
    E0=nothing;
    for ite=1:100
        
        @time E0,state,grad,direction,improvement=grad_line_search(H_plaquatte,state, D, chi, parameters, ctm_setting, grad_CTM_method,linesearch_CTM_method, dt, E0, grad, direction, alpha0, ls_ratio, ls_max)
        println("grad norm: "*string(norm(grad)));flush(stdout)
        if -improvement<1e-7
            break
        end
    end

end

