function initial_fPEPS_state_spinful_SU2(Vspace,init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    Vp=SU2Space(0=>2,1/2=>1)';
    global Lx,Ly
    if init_statenm=="nothing" 
        
        println("Random initial state");flush(stdout);
        
        state=Matrix{Square_iPEPS}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                if init_complex_tensor
                    A=TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp)+TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp)*im;
                else
                    A=TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp);
                end
                A=permute(A,(1,2,3,4,5,));
                A=A/norm(A);
                state[cx,cy]=Square_iPEPS(A);
            end
        end
        return state
    else
        println("load state: "*init_statenm);flush(stdout);
        x=load(init_statenm)["x"];
        state=similar(x);
        for cc in eachindex(x)
            ansatz=x[cc];
            A0=ansatz.T;
            if space(A0,1)==Vspace
                A=A0;
            else
                println("Extend bond dimension of initial state")
                if space(A0,1)==Rep[ℤ₂](0=>1, 1=>1)
                    if Vspace==Rep[ℤ₂](0=>2, 1=>2)
                        M=zeros(4,4,4,4,2)*im;
                        M[[1,3],[1,3],[1,3],[1,3],1:2]=convert(Array,A0);
                        A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
                    elseif Rep[ℤ₂](0=>3, 1=>3)
                        M=zeros(6,6,6,6,2)*im;
                        M[[1,4],[1,4],[1,4],[1,4],1:2]=convert(Array,A0);
                        A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
                    end
                elseif space(A0,1)==Rep[ℤ₂](0=>2, 1=>2)
                    if Vspace==Rep[ℤ₂](0=>3, 1=>3)
                        M=zeros(6,6,6,6,2)*im;
                        M[[1,2,4,5],[1,2,4,5],[1,2,4,5],[1,2,4,5],1:2]=convert(Array,A0);
                        A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
                    end
                end
                A=permute(A,(1,2,3,4,5,));
            end

            if init_complex_tensor
                A_noise=TensorMap(randn,codomain(A),domain(A))+im*TensorMap(randn,codomain(A),domain(A));
            else
                A_noise=TensorMap(randn,codomain(A),domain(A));
            end

            A_new=A+A_noise*init_noise*norm(A)/norm(A_noise);
            A_new=permute(A_new,(1,2,3,4,5,));
            ansatz_new=Square_iPEPS(A_new);
            state[cc]=ansatz_new;
        end
        return state
    end
end



function initial_fPEPS_state_spinless_U1(Vspace_set,init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    Vp=Rep[U₁](-1/2=>1,1/2=>1)';
    global Lx,Ly
    if init_statenm=="nothing" 
        
        println("Random initial state");flush(stdout);
        
        state=Matrix{Square_iPEPS}(undef,Lx,Ly);
        if (Lx==2)&(Ly==1)
            if init_complex_tensor
                A=TensorMap(randn,Vspace_set[1]*Vspace_set[2]*Vspace_set[3]*Vspace_set[4],Vp)+TensorMap(randn,Vspace_set[1]*Vspace_set[2]*Vspace_set[3]*Vspace_set[4],Vp)*im;
            else
                A=TensorMap(randn,Vspace_set[1]*Vspace_set[2]*Vspace_set[3]*Vspace_set[4],Vp);
            end
            
            A=permute(A,(1,2,3,4,5,));
            A=A/norm(A);
            state[1,1]=Square_iPEPS(A);
            ################3
            if init_complex_tensor
                A=TensorMap(randn,Vspace_set[5]*Vspace_set[6]*Vspace_set[7]*Vspace_set[8],Vp)+TensorMap(randn,Vspace_set[5]*Vspace_set[6]*Vspace_set[7]*Vspace_set[8],Vp)*im;
            else
                A=TensorMap(randn,Vspace_set[5]*Vspace_set[6]*Vspace_set[7]*Vspace_set[8],Vp);
            end
            A=permute(A,(1,2,3,4,5,));
            A=A/norm(A);
            state[2,1]=Square_iPEPS(A);
        end
        return state
    else
        println("load state: "*init_statenm);flush(stdout);
        x=load(init_statenm)["x"];
        state=similar(x);
        for cc in eachindex(x)
            ansatz=x[cc];
            A=ansatz.T;
            # if space(A0,1)==Vspace
            #     A=A0;
            # else
            #     println("Extend bond dimension of initial state")
            #     if space(A0,1)==Rep[ℤ₂](0=>1, 1=>1)
            #         if Vspace==Rep[ℤ₂](0=>2, 1=>2)
            #             M=zeros(4,4,4,4,2)*im;
            #             M[[1,3],[1,3],[1,3],[1,3],1:2]=convert(Array,A0);
            #             A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
            #         elseif Rep[ℤ₂](0=>3, 1=>3)
            #             M=zeros(6,6,6,6,2)*im;
            #             M[[1,4],[1,4],[1,4],[1,4],1:2]=convert(Array,A0);
            #             A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
            #         end
            #     elseif space(A0,1)==Rep[ℤ₂](0=>2, 1=>2)
            #         if Vspace==Rep[ℤ₂](0=>3, 1=>3)
            #             M=zeros(6,6,6,6,2)*im;
            #             M[[1,2,4,5],[1,2,4,5],[1,2,4,5],[1,2,4,5],1:2]=convert(Array,A0);
            #             A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
            #         end
            #     end
            #     A=permute(A,(1,2,3,4,5,));
            # end

            if init_complex_tensor
                A_noise=TensorMap(randn,codomain(A),domain(A))+im*TensorMap(randn,codomain(A),domain(A));
            else
                A_noise=TensorMap(randn,codomain(A),domain(A));
            end

            A_new=A+A_noise*init_noise*norm(A)/norm(A_noise);
            A_new=permute(A_new,(1,2,3,4,5,));
            ansatz_new=Square_iPEPS(A_new);
            state[cc]=ansatz_new;
        end
        return state
    end
end

function initial_fPEPS_state_spinless_Z2(Vspace,init_statenm="nothing",init_noise=0,init_complex_tensor=false)
    Vp=Rep[ℤ₂](0=>1,1=>1)';
    global Lx,Ly
    if init_statenm=="nothing" 
        
        println("Random initial state");flush(stdout);
        
        state=Matrix{Square_iPEPS}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                if init_complex_tensor
                    A=TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp)+TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp)*im;
                else
                    A=TensorMap(randn,Vv*Vv'*Vv'*Vv,Vp);
                end
                A=permute(A,(1,2,3,4,5,));
                A=A/norm(A);
                state[cx,cy]=Square_iPEPS(A);
            end
        end
        return state
    else
        println("load state: "*init_statenm);flush(stdout);
        x=load(init_statenm)["x"];
        state=similar(x);
        for cc in eachindex(x)
            ansatz=x[cc];
            A0=ansatz.T;
            if space(A0,1)==Vspace
                A=A0;
            else
                println("Extend bond dimension of initial state")
                if space(A0,1)==Rep[ℤ₂](0=>1, 1=>1)
                    if Vspace==Rep[ℤ₂](0=>2, 1=>2)
                        M=zeros(4,4,4,4,2)*im;
                        M[[1,3],[1,3],[1,3],[1,3],1:2]=convert(Array,A0);
                        A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
                    elseif Rep[ℤ₂](0=>3, 1=>3)
                        M=zeros(6,6,6,6,2)*im;
                        M[[1,4],[1,4],[1,4],[1,4],1:2]=convert(Array,A0);
                        A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
                    end
                elseif space(A0,1)==Rep[ℤ₂](0=>2, 1=>2)
                    if Vspace==Rep[ℤ₂](0=>3, 1=>3)
                        M=zeros(6,6,6,6,2)*im;
                        M[[1,2,4,5],[1,2,4,5],[1,2,4,5],[1,2,4,5],1:2]=convert(Array,A0);
                        A=TensorMap(M,Vspace*Vspace'*Vspace'*Vspace,Vp);
                    end
                end
                A=permute(A,(1,2,3,4,5,));
            end

            if init_complex_tensor
                A_noise=TensorMap(randn,codomain(A),domain(A))+im*TensorMap(randn,codomain(A),domain(A));
            else
                A_noise=TensorMap(randn,codomain(A),domain(A));
            end

            A_new=A+A_noise*init_noise*norm(A)/norm(A_noise);
            A_new=permute(A_new,(1,2,3,4,5,));
            ansatz_new=Square_iPEPS(A_new);
            state[cc]=ansatz_new;
        end
        return state
    end
end

function cost_fun(x::Matrix{T})  where T<:iPEPS_ansatz_immutable #variational parameters are vector of TensorMap
    global Lx,Ly
    global chi, parameters, energy_setting, grad_ctm_setting

    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=x[cx,cy].T;
            norm_A=norm(A)
            A=A/norm_A;

            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end
    
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init,[],grad_ctm_setting);
    E_total,  _=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
    E=real(E_total);

    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM_cell);
    E_tem=deepcopy(E)
    return E
end





function energy_CTM(x, chi, parameters, ctm_setting, energy_setting, init, init_CTM)
    global Lx,Ly
    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=x[cx,cy].T;
            norm_A=norm(A)
            A=A/norm_A;

            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end


    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init, init_CTM,ctm_setting);
    

    if energy_setting.model=="spinless_Hubbard"
        E_total,  ex_set, ey_set, e0_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, e0_set, ite_num,ite_err,CTM_cell
    elseif energy_setting.model=="spinless_Hubbard_pairing"
        E_total,  ex_set, ey_set, px_set, py_set, e0_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, px_set, py_set, e0_set, ite_num,ite_err,CTM_cell
    elseif energy_setting.model=="spinless_triangle_lattice"
        E_total,  ex_set, ey_set, e_right_bot_set, e0_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, e_right_bot_set, e0_set, ite_num,ite_err,CTM_cell
    elseif energy_setting.model=="spinful_triangle_lattice"
        E_total,  ex_set, ey_set, e_right_bot_set, e0_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        E=real(E_total);
        return E, ex_set, ey_set, e_right_bot_set, e0_set, ite_num,ite_err,CTM_cell
    end
end






function cost_fun_testt(x::Matrix{T})  where T<:iPEPS_ansatz #variational parameters are vector of TensorMap
    global Lx,Ly
    global chi, parameters, energy_setting, grad_ctm_setting

    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A=x[cx,cy].T;
            norm_A=norm(A)
            A=A/norm_A;

            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end
    
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init,[],grad_ctm_setting);
    E_total,  _=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
    E=real(E_total);

    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM_cell);
    E_tem=deepcopy(E)
    return E
end
