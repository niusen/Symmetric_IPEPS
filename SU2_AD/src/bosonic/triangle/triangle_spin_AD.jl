function create_isometry(V1,V2)
    #V1 is larger than V2
    #@assert dim(V1)>=dim(V2)
    tt=TensorMap(randn,V1,V2);
    for cc=1:length(tt.data.values)
        mm=tt.data.values[cc];
        tt.data.values[cc]=Matrix(I, size(mm,1), size(mm,2));
    end
    return tt
end

function initial_fPEPS_spinful_U1_SU2_2site_reducedD(Vx,Vy,init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    Vp=Rep[U₁ × SU₂]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (1, 1/2)=>2, (-1, 1/2)=>2, (0, 1)=>1)';
    if init_statenm=="nothing" 
        println("Random initial state");flush(stdout);
        
        if init_complex_tensor
            A=TensorMap(randn,Vx*Vy*Vx'*Vy',Vp)+TensorMap(randn,Vx*Vy*Vx'*Vy',Vp)*im;
        else
            A=TensorMap(randn,Vx*Vy*Vx'*Vy',Vp);
        end

        A=permute(A,(1,2,3,4,5,));
        A=A/norm(A);
        
        state=Square_iPEPS(A);
        return state
    else
        
        println("load state: "*init_statenm);flush(stdout);
        data=load(init_statenm);

        A0=data["A"];
        if (Vx==space(A0,1))&(Vy==space(A0,2))
            A=A0;
        else
            println("change bond dimension of initial state")
            Ureduce=create_isometry(Vx,space(A0,1));
            @tensor A[:]:=A0[1,-2,3,-4,-5]*Ureduce[-1,1]*Ureduce'[3,-3];

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
        e_diagonalb=hopping_diagonalb(CTM,Cdag,-1*C,A,AA,grad_ctm_setting);#compared with exact result, here a minus sign to ensure correct result
        e_diagonala=hopping_diagonala(CTM,Cdag,C,A,AA,grad_ctm_setting);
        e0=ob_onsite(CTM,occu,A,AA,grad_ctm_setting);

        E=real(t1*ex+t1'*ex' +t1*ey+t1'*ey'+ -2*μ*e0   +t2*e_diagonalb+t2'e_diagonalb'+ t2*e_diagonala+t2'e_diagonala');
    elseif energy_setting.model=="spinful_triangle_lattice_2site"
        Ident4, NA, NB, n_double_A, n_double_B, CdagA_CB, Cdag_A, C_A, Cdag_B, C_B = @ignore_derivatives Hamiltonians_spinful_U1_SU2_2site(M);
        t1 =parameters["t1"];
        t2=parameters["t2"];
        ϕ=parameters["ϕ"]
        μ=parameters["μ"];
        U=parameters["U"];

        ex1=hopping_x(CTM,Cdag_B,C_A,A,AA,grad_ctm_setting);
        ex2=ob_onsite(CTM,CdagA_CB,A,AA,grad_ctm_setting);

        ey1=hopping_y(CTM,Cdag_A,C_A,A,AA,grad_ctm_setting);
        ey2=hopping_y(CTM,Cdag_B,C_B,A,AA,grad_ctm_setting);

        e_diagonalb1=hopping_diagonalb(CTM,Cdag_B,C_A,A,AA,grad_ctm_setting);
        e_diagonalb2=hopping_y(CTM,Cdag_A,C_B,A,AA,grad_ctm_setting);
        
        eU1=ob_onsite(CTM,n_double_A-(1/2)*NA+(1/4)*Ident4,A,AA,grad_ctm_setting);
        eU2=ob_onsite(CTM,n_double_B-(1/2)*NB+(1/4)*Ident4,A,AA,grad_ctm_setting);

        #println([ex1,ex2,ey1,ey2,e_diagonalb1,e_diagonalb2,eU1,eU2])
        
        E_hop=t1*exp(im*ϕ)*ex1+t1*exp(im*ϕ)*ex2+t1*ey1-t1*ey2+t2*e_diagonalb1-t2*e_diagonalb2;
        E=real((E_hop+E_hop')/2)+real(U*(eU1+eU2))/2;
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
        e_diagonalb=hopping_diagonalb(CTM,Cdag,-1*C,A,AA,ctm_setting);#compared with exact result, here a minus sign to ensure correct result
        e_diagonala=hopping_diagonala(CTM,Cdag,C,A,AA,ctm_setting);
        e0=ob_onsite(CTM,occu,A,AA,ctm_setting);

        E=real(t1*ex+t1'*ex' +t1*ey+t1'*ey'+ -2*μ*e0   +t2*e_diagonalb+t2'e_diagonalb'+ t2*e_diagonala+t2'e_diagonala');

        return E, ex,ey,e_diagonalb,e_diagonala, e0, ite_num,ite_err,CTM


    elseif energy_setting.model=="spinful_triangle_lattice_2site"
        Ident4, NA, NB, n_double_A,n_double_B, CdagA_CB, Cdag_A, C_A, Cdag_B, C_B = @ignore_derivatives Hamiltonians_spinful_U1_SU2_2site(M);
        t1 =parameters["t1"];
        t2=parameters["t2"];
        ϕ=parameters["ϕ"]
        μ=parameters["μ"];
        U=parameters["U"];

        ex1=hopping_x(CTM,Cdag_B,C_A,A,AA,ctm_setting);
        ex2=ob_onsite(CTM,CdagA_CB,A,AA,ctm_setting);

        ey1=hopping_y(CTM,Cdag_A,C_A,A,AA,ctm_setting);
        ey2=hopping_y(CTM,Cdag_B,C_B,A,AA,ctm_setting);

        e_diagonalb1=hopping_diagonalb(CTM,Cdag_B,C_A,A,AA,ctm_setting);
        e_diagonalb2=hopping_y(CTM,Cdag_A,C_B,A,AA,ctm_setting);

        e01=ob_onsite(CTM,NA,A,AA,ctm_setting);
        e02=ob_onsite(CTM,NB,A,AA,ctm_setting);

        eU1=ob_onsite(CTM,n_double_A-(1/2)*NA+(1/4)*Ident4,A,AA,grad_ctm_setting);
        eU2=ob_onsite(CTM,n_double_B-(1/2)*NB+(1/4)*Ident4,A,AA,grad_ctm_setting);

        E_hop=t1*exp(im*ϕ)*ex1+t1*exp(im*ϕ)*ex2+t1*ey1-t1*ey2+t2*e_diagonalb1-t2*e_diagonalb2;
        E=real((E_hop+E_hop')/2)+real(U*(eU1+eU2))/2;
        return E, ex1,ex2,ey1,ey2,e_diagonalb1,e_diagonalb2, e01,e02,eU1,eU2, ite_num,ite_err,CTM
    end
end






