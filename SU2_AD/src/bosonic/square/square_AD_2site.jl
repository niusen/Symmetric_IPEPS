function initial_SU2_state(Vspace,init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    if init_statenm=="nothing" 
        println("Random initial state");flush(stdout);
        Vp=SU2Space(0=>1,1=>1);
        if init_complex_tensor
            A=TensorMap(randn,Vv*Vv*Vv*Vv,Vp)+TensorMap(randn,Vv*Vv*Vv*Vv,Vp)*im;
        else
            A=TensorMap(randn,Vv*Vv*Vv*Vv,Vp);
        end

        A=permute(A,(1,2,3,4,5,));
        A=A/norm(A);
        
        U=unitary(Vv',Vv);
        @tensor A[:]:=A[-1,-2,1,2,-5]*U[-3,1]*U[-4,2];
        
        

        state=Square_iPEPS(A);
        return state
    else
        Vp=SU2Space(0=>1,1=>1);
        println("load state: "*init_statenm);flush(stdout);
        data=load(init_statenm);

        A0=data["A"];
        if (Vspace==space(A0,1))|(Vspace==space(A0,1)')
            A=A0;
        else
            println("Extend bond dimension of initial state")
            if space(A0,1)==GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'
                if Vspace==SU2Space(0=>2,1/2=>1)
                    M=convert(Array,A0);
                    A=TensorMap(M,Vspace'*Vspace*Vspace*Vspace',Vp')

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

function initial_U1_SU2_state(Vspace,init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    if init_statenm=="nothing" 
        println("Random initial state");flush(stdout);
        Vp=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(0,1)=>1)';
        if init_complex_tensor
            A=TensorMap(randn,Vv*Vv*Vv*Vv,Vp)+TensorMap(randn,Vv*Vv*Vv*Vv,Vp)*im;
        else
            A=TensorMap(randn,Vv*Vv*Vv*Vv,Vp);
        end

        A=permute(A,(1,2,3,4,5,));
        A=A/norm(A);
        
        U=unitary(Vv',Vv);
        @tensor A[:]:=A[-1,-2,1,2,-5]*U[-3,1]*U[-4,2];
        
        

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
            if space(A0,1)==SU2Space(0=>1,1/2=>1)
                if Vspace==SU2Space(0=>2,1/2=>1)
                    M=zeros(4,4,4,4,2)*im;
                    M[2:4,2:4,2:4,2:4,1:2]=convert(Array,A0);
                    A=TensorMap(M,Vspace*Vspace,Vspace*Vspace*SU2Space(1/2=>1));
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

    CTM, AA, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A,chi,init,[],grad_ctm_setting);

    if isa(energy_setting,Square_2site_Energy_settings)
        E=energy_2site(parameters,A,AA,CTM);
        E=real(E);
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

    CTM, AA, U_L,U_D,U_R,U_U,ite_num,ite_err=CTMRG(A,chi,init, init_CTM, ctm_setting);

    #@assert ite_err<3*(1e-5)
    if isa(energy_setting,Square_2site_Energy_settings)
        E=energy_2site(parameters,A,AA,CTM);
        E=real(E);
        return E, ite_num,ite_err,CTM
    end
end






