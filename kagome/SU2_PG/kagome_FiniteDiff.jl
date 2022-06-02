function initial_state(init_statenm,Bond_irrep)
    if init_statenm==None #random initial state

    else
        #filenm="LS_D_"*string(D)*"_chi_40.json"
        json_dict=read_json_state(init_statenm);

        A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
        
        bond_tensor,triangle_tensor=construct_su2_PG_IPESS(state_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);
end


function cal_(parameters,D,chi)
    
    multi_threads=true;if Threads.nthreads()==1; multi_threads=false; end
    println("number of threads: "*string(Threads.nthreads()));flush(stdout);
    
    CTM_conv_tol=1e-6;
    
    CTM_ite_nums=50;
    CTM_trun_tol=1e-12;
    
end


function energy_CTM(D,chi,parameters, CTM_conv_tol,CTM_ite_nums,CTM_trun_tol,state_dict)

    A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb=construct_tensor(D);
    
    bond_tensor,triangle_tensor=construct_su2_PG_IPESS(state_dict,A_set,B_set,A1_set,A2_set, A_set_occu,B_set_occu,A1_set_occu,A2_set_occu, S_label, Sz_label, virtual_particle, Va, Vb);
    
    PEPS_tensor=bond_tensor;
    @tensor PEPS_tensor[:] := bond_tensor[-1,1,-5]*bond_tensor[4,3,-6]*bond_tensor[-4,2,-7]*triangle_tensor[1,3,2]*triangle_tensor[4,-2,-3];
    A_unfused=PEPS_tensor;
    
    U_phy=unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6) ⊗ space(PEPS_tensor, 7));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2,3]*U_phy[-5,1,2,3];

    CTM=[];
    U_L=[];
    U_D=[];
    U_R=[];
    U_U=[];

    init=Dict([("CTM", []), ("init_type", "PBC")]);
    conv_check="singular_value";

    CTM, AA_fused, U_L,U_D,U_R,U_U=CTMRG(A_fused,chi,conv_check,CTM_conv_tol,init,CTM_ite_nums,CTM_trun_tol);
    
    E_up, E_down=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_triangle");
    E_up_12, E_up_31, E_up_23, E_down_12, E_down_31, E_down_23=evaluate_ob(parameters, U_phy, A_unfused, A_fused, AA_fused, U_L,U_D,U_R,U_U, CTM, "E_bond");
    energy=(E_up+E_down)/3);

    return energy,CTM,U_L,U_D,U_R,U_U
end




print('optimization start')
#E0,_,_=Grad_FiniteDiff(state, cfg.ctm_args, args.chi)
grad=None
direction=None
for ite in range(0,100):
    time_start=time.time()
    E0,state,grad,direction,improvement=grad_line_search(state, cfg.ctm_args, args.chi, E0=E0, grad0=grad, direction0=direction, alpha0=3, ls_max=5)
    print('grad norm: '+format(torch.norm(grad)))
    time_end=time.time()
    print('totally cost',time_end-time_start)
    if -improvement<1e-7:
        break







        def normalize_IPESS_SU2_PG(state):
        if state.Bond_irrep=='A':
            Bond_norm=torch.norm(state.coes['Bond_A_coe'])
            state.coes["Bond_A_coe"]=state.coes["Bond_A_coe"]/Bond_norm
        elif state.Bond_irrep=='B':
            Bond_norm=torch.norm(state.coes['Bond_B_coe'])
            state.coes["Bond_B_coe"]=state.coes["Bond_B_coe"]/Bond_norm
        elif state.Bond_irrep=='A+iB':
            Bond_norm=torch.sqrt((torch.norm(state.coes['Bond_A_coe']))**2+(torch.norm(state.coes['Bond_B_coe']))**2)
            state.coes["Bond_A_coe"]=state.coes["Bond_A_coe"]/Bond_norm
            state.coes["Bond_B_coe"]=state.coes["Bond_B_coe"]/Bond_norm
        Triangle_norm=torch.sqrt((torch.norm(state.coes["Triangle_A1_coe"]))**2+(torch.norm(state.coes["Triangle_A2_coe"]))**2)
        state.coes["Triangle_A1_coe"]=state.coes["Triangle_A1_coe"]/Triangle_norm
        state.coes["Triangle_A2_coe"]=state.coes["Triangle_A2_coe"]/Triangle_norm
        return state

    @torch.no_grad()
    def Grad_FiniteDiff(state, ctm_args, chi, E0=None):
        dt=args.dt

        state=normalize_IPESS_SU2_PG(state)
        #print(E0)
        if args.ansatz in ["IPESS_SU2_PG"]:
            state_sym= state
            state_sym.sites= state_sym.build_onsite_tensors()
        ctm_env_in = ENV(chi, state_sym)

        init_env(state, ctm_env_in)
        if E0==None:
            ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state_sym, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
            E0 = energy_f_NoCheck(state_sym, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
        coes=copy.deepcopy(state_sym.coes)
        #print('energy E0 is '+format(E0))
        dt=0.001

        coes_tem=copy.deepcopy(coes)
        Grad_FD={}
        dE_data=torch.empty((0),device=cfg.global_args.device)
        Grad_FD_data=torch.empty((0),device=cfg.global_args.device)

        #Bond A tensor diff
        if state.Bond_irrep in ['A','A+iB']:
            Bond_A_grad=torch.zeros(coes['Bond_A_coe'].size(dim=0),dtype=torch.float64, device=cfg.global_args.device)
            for ct in range(0,state_sym.coes['Bond_A_coe'].size(dim=0)):
                coes_tem=copy.deepcopy(coes)
                coes_tem['Bond_A_coe'][ct]=coes_tem['Bond_A_coe'][ct]+dt
                state_sym.coes=copy.deepcopy(coes_tem)
                state_sym.sites= state_sym.build_onsite_tensors()
                init_env(state, ctm_env)
                ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state_sym, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
                E = energy_f_NoCheck(state_sym, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
                Bond_A_grad[ct]=(E-E0)/dt
                dE_data=torch.cat((dE_data, torch.tensor([E-E0],device=cfg.global_args.device)), axis=0)
                #print('energy is '+format(E))
            #print(Bond_A_grad)
            Grad_FD['Bond_A_grad']=Bond_A_grad
            Grad_FD_data=torch.cat((Grad_FD_data, Bond_A_grad), axis=0)

        #Bond B tensor diff
        if state.Bond_irrep in ['B','A+iB']:
            Bond_B_grad=torch.zeros(coes['Bond_B_coe'].size(dim=0),dtype=torch.float64, device=cfg.global_args.device)
            for ct in range(0,state_sym.coes['Bond_B_coe'].size(dim=0)):
                coes_tem=copy.deepcopy(coes)
                coes_tem['Bond_B_coe'][ct]=coes_tem['Bond_B_coe'][ct]+dt
                state_sym.coes=copy.deepcopy(coes_tem)
                state_sym.sites= state_sym.build_onsite_tensors()
                init_env(state, ctm_env)
                ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state_sym, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
                E = energy_f_NoCheck(state_sym, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
                Bond_B_grad[ct]=(E-E0)/dt
                dE_data=torch.cat((dE_data, torch.tensor([E-E0],device=cfg.global_args.device)), axis=0)
                #print('energy is '+format(E))
            #print(Bond_B_grad)
            Grad_FD['Bond_B_grad']=Bond_B_grad
            Grad_FD_data=torch.cat((Grad_FD_data, Bond_B_grad), axis=0)

        #triangle A1 tensor diff
        Triangle_A1_grad=torch.zeros(coes['Triangle_A1_coe'].size(dim=0),dtype=torch.float64, device=cfg.global_args.device)
        for ct in range(0,state_sym.coes['Triangle_A1_coe'].size(dim=0)):
            coes_tem=copy.deepcopy(coes)
            coes_tem['Triangle_A1_coe'][ct]=coes_tem['Triangle_A1_coe'][ct]+dt
            state_sym.coes=copy.deepcopy(coes_tem)
            state_sym.sites= state_sym.build_onsite_tensors()
            init_env(state, ctm_env)
            ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state_sym, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
            E = energy_f_NoCheck(state_sym, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
            Triangle_A1_grad[ct]=(E-E0)/dt
            dE_data=torch.cat((dE_data, torch.tensor([E-E0],device=cfg.global_args.device)), axis=0)
            #print('energy is '+format(E))
        #print(Triangle_A1_grad)
        Grad_FD['Triangle_A1_grad']=Triangle_A1_grad
        Grad_FD_data=torch.cat((Grad_FD_data, Triangle_A1_grad), axis=0)

        #triangle A2 tensor diff
        Triangle_A2_grad=torch.zeros(coes['Triangle_A2_coe'].size(dim=0),dtype=torch.float64, device=cfg.global_args.device)
        for ct in range(0,state_sym.coes['Triangle_A2_coe'].size(dim=0)):
            coes_tem=copy.deepcopy(coes)
            coes_tem['Triangle_A2_coe'][ct]=coes_tem['Triangle_A2_coe'][ct]+dt
            state_sym.coes=copy.deepcopy(coes_tem)
            state_sym.sites= state_sym.build_onsite_tensors()
            init_env(state, ctm_env)
            ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state_sym, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
            E = energy_f_NoCheck(state_sym, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
            Triangle_A2_grad[ct]=(E-E0)/dt
            dE_data=torch.cat((dE_data, torch.tensor([E-E0],device=cfg.global_args.device)), axis=0)
            #print('energy is '+format(E))
        #print(Triangle_A2_grad)
        Grad_FD['Triangle_A2_grad']=Triangle_A2_grad
        Grad_FD_data=torch.cat((Grad_FD_data, Triangle_A2_grad), axis=0)

        # print('Energy difference is:')
        # print(dE_data)
        # print('Grad is:')
        # print(Grad_FD_data)
        # print('Normalized grad is:')
        # print(Grad_FD_data/max(abs(Grad_FD_data)))

        state_sym.coes=coes
        return E0,Grad_FD,Grad_FD_data

    @torch.no_grad()
    def grad_line_search(state, ctm_args, chi, E0=E0, grad0=None, direction0=None, alpha0=1, ls_ratio=1/3, ls_max=10):
        if args.nonchiral=='no':
            filenm='LS_D_'+str(args.bond_dim)+'_chi_'+str(args.chi)+'.json'
        elif args.nonchiral=='A1_even':
            filenm='LS_A1even_D_'+str(args.bond_dim)+'_chi_'+str(args.chi)+'.json'
        elif args.nonchiral=='A1_odd':
            filenm='LS_A1odd_D_'+str(args.bond_dim)+'_chi_'+str(args.chi)+'.json'

        print('line search')
        state=normalize_IPESS_SU2_PG(state)
        state.sites= state.build_onsite_tensors()
        ctm_env_in = ENV(chi, state)
        init_env(state, ctm_env_in)

        #E0,_,grad=Grad_FiniteDiff(state, ctm_args, chi)
        _,_,grad=Grad_FiniteDiff(state, ctm_args, chi)
        print('state:'+format(state.get_vector()))
        print('grad:'+format(grad))

        init_env(state, ctm_env_in)
        ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
        E0 = energy_f_NoCheck(state, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
        print(format(E0))
        #grad=torch.from_numpy(grad).to(cfg.global_args.device)


        direction=-grad
        #print(grad0)
        #print(grad)
        if grad0==None:
            direction=-grad
        else:
            norm_grad=torch.norm(grad)
            norm_grad0=torch.norm(grad0)
            beta=(norm_grad**2)/(norm_grad0**2)
            direction=-grad+beta*direction0

        vec0=copy.deepcopy(state.get_vector())
        
        #line search
        improved=False
        alpha=alpha0
        print('conjugate gradient opt')
        for ls_step in range(0,ls_max):
            vec_tem=vec0+direction*alpha*(ls_ratio**ls_step)
            state.set_vector(vec_tem)
            state.sites= state.build_onsite_tensors()
            init_env(state, ctm_env_in)
            ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
            E = energy_f_NoCheck(state, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
            print(format(E))
            if E<E0:
                improved=True
                break
        if improved:
            state.set_vector(vec_tem)
            state.sites= state.build_onsite_tensors()
            init_env(state, ctm_env_in)
            ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
            E = energy_f_NoCheck(state, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
            state.write_to_file(filenm)
        else:
            print('gradient opt')
            for ls_step in range(0,ls_max):
                vec_tem=vec0-grad*alpha*(ls_ratio**ls_step)
                state.set_vector(vec_tem)
                state.sites= state.build_onsite_tensors()
                init_env(state, ctm_env_in)
                ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
                E = energy_f_NoCheck(state, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
                print(format(E))
                if E<E0:
                    improved=True
                    break
                
            if improved:
                state.set_vector(vec_tem)
                state.sites= state.build_onsite_tensors()
                init_env(state, ctm_env_in)
                ctm_env, history, t_ctm, t_conv_check = ctmrg.run(state, ctm_env_in, conv_check=ctmrg_conv_energy, ctm_args=cfg.ctm_args)
                E = energy_f_NoCheck(state, ctm_env, force_cpu=ctm_args.conv_check_cpu).real
                state.write_to_file(filenm)
            else:
                state.set_vector(vec0)
                state.sites= state.build_onsite_tensors()
                E=E0
        improvement=E-E0
        
        state.write_to_file(filenm)
        return E,state,grad,direction,improvement





