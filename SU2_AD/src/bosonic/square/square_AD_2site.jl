function initial_SU2_anistropic_state(Vx,Vy,init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    if init_statenm=="nothing" 
        println("Random initial state");flush(stdout);
        Vp=SU2Space(0=>1,1=>1);
        if init_complex_tensor
            A=TensorMap(randn,Vx'*Vy*Vx*Vy',Vp')+TensorMap(randn,Vx'*Vy*Vx*Vy',Vp')*im;
        else
            A=TensorMap(randn,Vx'*Vy*Vx*Vy',Vp');
        end

        A=permute(A,(1,2,3,4,5,));
        A=A/norm(A);

        state=Square_iPEPS(A);
        return state
    else
        Vp=SU2Space(0=>1,1=>1);
        println("load state: "*init_statenm);flush(stdout);
        data=load(init_statenm);

        A0=data["A"];
        if (Vx==space(A0,1)')&(Vy==space(A0,4)')
            A=A0;
        else
            println("Extend bond dimension of initial state")
            if space(A0,1)==Rep[SU₂](0=>2, 1/2=>1)'
                if (Vx==SU2Space(0=>2,1/2=>1))&(Vy==SU2Space(0=>4,1/2=>3,1=>1))
                    M=convert(Array,A0);
                    M0=zeros(dim(Vx),dim(Vy),dim(Vx),dim(Vy),dim(Vp))*im;
                    M0[:,[1,2,5,6],:,[1,2,5,6],:]=M;
                    A=TensorMap(M0,Vx'*Vy*Vx*Vy',Vp')

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

function initial_SU2_state(Vspace,init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    if init_statenm=="nothing" 
        println("Random initial state");flush(stdout);
        Vp=SU2Space(0=>1,1=>1);
        if init_complex_tensor
            A=TensorMap(randn,Vv'*Vv*Vv*Vv',Vp')+TensorMap(randn,Vv'*Vv*Vv*Vv',Vp')*im;
        else
            A=TensorMap(randn,Vv'*Vv*Vv*Vv',Vp');
        end

        A=permute(A,(1,2,3,4,5,));
        A=A/norm(A);

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
            elseif space(A0,1)==GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1,(-1,1/2)=>2,(-3,1/2)=>2,(-2,1)=>1)
                if Vspace==SU2Space(0=>5,1/2=>4,1=>1)
                    # M=convert(Array,A0);
                    # A=TensorMap(M,Vspace'*Vspace,Vspace'*Vspace*Vp')

                    #if convert directly, outofmemory error will occur
                    Va=space(A0,1);
                    Vb=SU2Space(0=>5,1/2=>4,1=>1);

                    V0_a=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1);
                    Vhalf_a=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-1,1/2)=>2,(-3,1/2)=>2);
                    V1_a=GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((-2,1)=>1);

                    P0_a=TensorMap(randn,V0_a,Va)*0;
                    P0_a.data.values[1][1]=1;
                    P0_a.data.values[2]=[1 0 0;0 1 0;0 0 1];
                    P0_a.data.values[3][1]=1;

                    Phalf_a=TensorMap(randn,Vhalf_a,Va)*0;
                    Phalf_a.data.values[1]=[1 0;0 1];
                    Phalf_a.data.values[2]=[1 0;0 1];

                    P1_a=TensorMap(randn,V1_a,Va)*0;
                    P1_a.data.values[1][1]=1;

                    spaces_a=(V0_a,Vhalf_a,V1_a);
                    Projectors_a=(P0_a,Phalf_a,P1_a);

                    V0_b=SU2Space(0=>5);
                    Vhalf_b=SU2Space(1/2=>4);
                    V1_b=SU2Space(1=>1);

                    P0_b=TensorMap(randn,V0_b,Vb)*0;
                    P0_b.data.values[1]=Matrix(I,5,5);

                    Phalf_b=TensorMap(randn,Vhalf_b,Vb)*0;
                    Phalf_b.data.values[1]=Matrix(I,4,4);

                    P1_b=TensorMap(randn,V1_b,Vb)*0;
                    P1_b.data.values[1][1]=1;

                    spaces_b=(V0_b,Vhalf_b,V1_b);
                    Projectors_b=(P0_b,Phalf_b,P1_b);

                    A=TensorMap(randn,Vb*Vb'*Vb'*Vb,Vp')*0*im;
                    A=permute(A,(1,2,3,4,5,));
                    global A_comp
                    for c1=1:3
                        for c2=1:3
                            for c3=1:3
                                for c4=1:3
                                    #println([c1,c2,c3,c4])
                                    @tensor A_comp[:]:=Projectors_a[c1][-1,1]*Projectors_a[c2]'[2,-2]*Projectors_a[c3]'[3,-3]*Projectors_a[c4][-4,4]*A0[1,2,3,4,-5];
                                    if norm(A_comp)==0
                                        continue;
                                    end
                                    A_comp_dense=convert(Array,permute(A_comp,(1,2,3,4,),(5,)));
                                    A_comp_=TensorMap(A_comp_dense,spaces_b[c1]*spaces_b[c2]'*spaces_b[c3]'*spaces_b[c4],Vp');
                                    # println(space(Projectors_b[c1]'))
                                    # println(space(Projectors_b[c2]))
                                    # println(space(Projectors_b[c3]))
                                    # println(space(Projectors_b[c4]'))
                                    # println(space(A_comp_))
                                    @tensor A_comp__[:]:=Projectors_b[c1]'[-1,1]*Projectors_b[c2][2,-2]*Projectors_b[c3][3,-3]*Projectors_b[c4]'[-4,4]*A_comp_[1,2,3,4,-5];
                                    A=A+A_comp__
                                end
                            end
                        end
                    end
                end
            elseif space(A0,1)==SU2Space(0=>2,1/2=>1)'#initial D=4
                if Vv==SU2Space(0=>3,1/2=>2,1=>1);
                    M=convert(Array,A0);
                    M0=zeros(dim(Vv),dim(Vv),dim(Vv),dim(Vv),dim(Vp))*im;
                    pos=[1,2,4,5];
                    M0[pos,pos,pos,pos,:]=M;
                    A=TensorMap(M0,Vv'*Vv*Vv*Vv',Vp')
                end
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
            if space(A0,1)==GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)';
                if Vspace==GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0, 0)=>1, (2, 0)=>2, (1, 1/2)=>1,(3, 1/2)=>1,(2,1)=>1)';
                    M=zeros(10,10,10,10,4)*im;
                    pos=[1,2,4,5];
                    M[pos,pos,pos,pos,1:4]=convert(Array,A0);
                    A=TensorMap(M,Vspace*Vspace',Vspace*Vspace'*GradedSpace[Irrep[U₁]⊠Irrep[SU₂]]((0,0)=>1,(0,1)=>1)');

                end
            elseif space(A0,1)==SU2Space(0=>1,1/2=>1)
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






