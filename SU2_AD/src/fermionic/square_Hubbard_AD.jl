function initial_fPEPS_state_spinless_Z2(Vspace,init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    Vp=Rep[ℤ₂](0=>1,1=>1)';
    if init_statenm=="nothing" 
        println("Random initial state");flush(stdout);
        
        if init_complex_tensor
            A=TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp)+TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp)*im;
        else
            A=TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp);
        end

        A=permute(A,(1,2,3,4,5,));
        A=A/norm(A);
        
        state=Square_iPEPS(A);
        return state
    else
        
        println("load state: "*init_statenm);flush(stdout);
        data=load(init_statenm);

        A0=data["A"];
        if Vspace==space(A0,1)
            A=A0;
        else
            println("Extend bond dimension of initial state")
            if space(A0,1)==Rep[ℤ₂](0=>1, 1=>1)
                if Vspace==Rep[ℤ₂](0=>2, 1=>2)
                    M=zeros(4,4,4,4,2)*im;
                    M[[1,3],[1,3],[1,3],[1,3],1:2]=convert(Array,A0);
                    A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
                elseif Vspace==SU2Space(0=>1,1/2=>2)
                    M=zeros(5,5,5,5,2)*im;
                    M[1:3,1:3,1:3,1:3,1:2]=convert(Array,A0);
                    A=TensorMap(M,Vspace*Vspace,Vspace*Vspace*SU2Space(1/2=>1));
                end
            elseif space(A0,1)==SU2Space(0=>2,1/2=>1)
            elseif space(A0,1)==SU2Space(0=>1,1/2=>2)
            end
            A=permute(A,(1,2,3,4,5,));
        end

        if init_complex_tensor
            A_noise=TensorMap(randn,codomain(A),domain(A))+im*TensorMap(randn,codomain(A),domain(A));
        else
            A_noise=TensorMap(randn,codomain(A),domain(A));
        end

        A=A+A_noise*init_noise*norm(A)/norm(A_noise);

        state=Square_iPEPS(A);

        return state
    end
end

function cost_fun(x) #variational parameters are vector of TensorMap
    global chi, parameters, energy_setting, grad_ctm_setting
    A=x.T;
    norm_A=norm(A)
    A= A/norm_A;
    
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

    CTM, AA, U_L,U_D,U_R,U_U,ite_num,ite_err=fermi_CTMRG(A,chi,init,[],grad_ctm_setting);

    if energy_setting.model=="spinless_Hubbard"
        Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonians_spinless_Z2();
        t1 =parameters["t1"];
        μ=parameters["μ"];
        ex=hopping_x(CTM,Cdag,C,A,AA,grad_ctm_setting);
        ey=hopping_y(CTM,Cdag,C,A,AA,grad_ctm_setting);
        e0=ob_onsite(CTM,occu,A,AA,grad_ctm_setting);

        E=real(t1*ex+t1'*ex' +t1*ey+t1'*ey'-2*μ*e0);

    elseif energy_setting.model=="spinless_Hubbard_pairing"
        Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonians_spinless_Z2();
        t1 =parameters["t1"];
        γ=parameters["γ"];
        μ=parameters["μ"];
        ex=hopping_x(CTM,Cdag,C,A,AA,grad_ctm_setting);
        ey=hopping_y(CTM,Cdag,C,A,AA,grad_ctm_setting);
        px=hopping_x(CTM,Cdag,Cdag_,A,AA,grad_ctm_setting);
        py=hopping_y(CTM,Cdag,Cdag_,A,AA,grad_ctm_setting);
        e0=ob_onsite(CTM,occu,A,AA,grad_ctm_setting);

        E=real(t1*ex+t1'*ex' +t1*ey+t1'*ey'+ -2*μ*e0   +γ*px+γ'px'+ γ*py+γ'py');
    elseif energy_setting.model=="spinless_t1_t2"
        Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonians_spinless_Z2();
        t1 =parameters["t1"];
        t2=parameters["t2"];
        μ=parameters["μ"];

        ex=hopping_x(CTM,Cdag,C,A,AA,grad_ctm_setting);
        ey=hopping_y(CTM,Cdag,C,A,AA,grad_ctm_setting);
        e_right_top=hopping_right_top(CTM,Cdag,-1*C,A,AA,grad_ctm_setting);#compared with exact result, here a minus sign to ensure correct result
        e_right_bot=hopping_right_bot(CTM,Cdag,C,A,AA,grad_ctm_setting);
        e0=ob_onsite(CTM,occu,A,AA,grad_ctm_setting);

        E=real(t1*ex+t1'*ex' +t1*ey+t1'*ey'+ -2*μ*e0   +t2*e_right_top+t2'e_right_top'+ t2*e_right_bot+t2'e_right_bot');
    end

    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM);
    E_tem=deepcopy(E)
    return E
end





function energy_CTM(x, chi, parameters, ctm_setting, energy_setting, init, init_CTM)
    A=x.T;
    norm_A=norm(A)
    A= A/norm_A;

    CTM, AA, U_L,U_D,U_R,U_U,ite_num,ite_err=fermi_CTMRG(A,chi,init, init_CTM, ctm_setting);

    #@assert ite_err<3*(1e-5)

    if energy_setting.model=="spinless_Hubbard"
        Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonians_spinless_Z2();
        t1 =parameters["t1"];
        μ=parameters["μ"];
        
        ex=hopping_x(CTM,Cdag,C,A,AA,ctm_setting);
        ey=hopping_y(CTM,Cdag,C,A,AA,ctm_setting);
        e0=ob_onsite(CTM,occu,A,AA,ctm_setting);

        E=real(t1*ex+t1'*ex' +t1*ey+t1'*ey' -2*μ*e0);

        return E, ex,ey,e0, ite_num,ite_err,CTM
    elseif energy_setting.model=="spinless_Hubbard_pairing"
        Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonians_spinless_Z2();
        t1 =parameters["t1"];
        γ=parameters["γ"];
        μ=parameters["μ"];
        
        ex=hopping_x(CTM,Cdag,C,A,AA,ctm_setting);
        ey=hopping_y(CTM,Cdag,C,A,AA,ctm_setting);
        px=hopping_x(CTM,Cdag,Cdag_,A,AA,ctm_setting);
        py=hopping_y(CTM,Cdag,Cdag_,A,AA,ctm_setting);
        e0=ob_onsite(CTM,occu,A,AA,ctm_setting);

        E=real(t1*ex+t1'*ex' +t1*ey+t1'*ey' -2*μ*e0   +γ*px+γ'px'+ γ*py+γ'py');
    elseif energy_setting.model=="spinless_t1_t2"
        Ident, occu, Cdag, C, Cdag_ =@ignore_derivatives Hamiltonians_spinless_Z2();
        t1 =parameters["t1"];
        t2=parameters["t2"];
        μ=parameters["μ"];
        
        ex=hopping_x(CTM,Cdag,C,A,AA,ctm_setting);
        ey=hopping_y(CTM,Cdag,C,A,AA,ctm_setting);
        e_right_top=hopping_right_top(CTM,Cdag,-1*C,A,AA,ctm_setting);#compared with exact result, here a minus sign to ensure correct result
        e_right_bot=hopping_right_bot(CTM,Cdag,C,A,AA,ctm_setting);
        e0=ob_onsite(CTM,occu,A,AA,ctm_setting);

        E=real(t1*ex+t1'*ex' +t1*ey+t1'*ey'+ -2*μ*e0   +t2*e_right_top+t2'e_right_top'+ t2*e_right_bot+t2'e_right_bot');

        return E, ex,ey,e_right_top,e_right_bot, e0, ite_num,ite_err,CTM
    end
end





