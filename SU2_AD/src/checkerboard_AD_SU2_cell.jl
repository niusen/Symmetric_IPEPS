function build_A_checkerboard(x::Checkerboard_iPESS_immutable)
    BL=x.B_L;
    BU=x.B_U;
    Tm=x.Tm;
    @tensor PEPS_tensor[:] := BL[-1,1,-5]*BU[-4,2,-6]*Tm[1,-2,-3,2];
    A_unfused=PEPS_tensor;
    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2]*U_phy[-5,1,2];
    return A_unfused,A_fused,U_phy
end

function build_A_checkerboard(x::Checkerboard_iPESS)
    BL=x.B_L;
    BU=x.B_U;
    Tm=x.Tm;
    @tensor PEPS_tensor[:] := BL[-1,1,-5]*BU[-4,2,-6]*Tm[1,-2,-3,2];
    A_unfused=PEPS_tensor;
    U_phy=@ignore_derivatives unitary(fuse(space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6)), space(PEPS_tensor, 5) ⊗ space(PEPS_tensor, 6));
    @tensor A_fused[:] :=PEPS_tensor[-1,-2,-3,-4,1,2]*U_phy[-5,1,2];
    return A_unfused,A_fused,U_phy
end



function initial_SU2_state(Vspace,init_statenm="nothing",init_noise=0)
    if init_statenm=="nothing" 
        global Lx,Ly
        println("Random initial state");flush(stdout);
        Vp=SU2Space(1=>1);
        state=Matrix{Checkerboard_iPESS}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                bL=TensorMap(randn,Vv*Vv,Vp)*(1+0*im);
                bU=TensorMap(randn,Vv*Vv,Vp)*(1+0*im);
                tM=TensorMap(randn,Vv',Vv*Vv*Vv)*(1+0*im);
                bL=permute(bL,(1,2,3,));
                bU=permute(bU,(1,2,3,));
                tM=permute(tM,(1,2,3,4,));
                state[cx,cy]=Checkerboard_iPESS(bL,bU,tM);
            end
        end
        return state
    else
        
        println("load state: "*init_statenm);flush(stdout);
        x=load(init_statenm)["x"];
        state=similar(x);
        for cc in eachindex(x)
            ansatz=x[cc];
            B_L=ansatz.B_L;
            B_U=ansatz.B_U;
            T_m=ansatz.Tm;

            @assert space(B_L,1)==Vspace
            BL_noise=TensorMap(randn,codomain(B_L),domain(B_L));
            BU_noise=TensorMap(randn,codomain(B_U),domain(B_U));
            Tm_noise=TensorMap(randn,codomain(T_m),domain(T_m));
            
            BL=B_L+BL_noise*init_noise*norm(B_L)/norm(BL_noise);
            BU=B_U+BU_noise*init_noise*norm(B_U)/norm(BU_noise);
            Tm=T_m+Tm_noise*init_noise*norm(T_m)/norm(Tm_noise);

            ansatz_new=Checkerboard_iPESS(BL,BU,Tm);
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
            A_unfused,A_fused,U_phy=build_A_checkerboard(x[cx, cy]);
            A_unfused_cell=fill_tuple(A_unfused_cell, A_unfused, cx,cy);
            A_fused_cell=fill_tuple(A_fused_cell, A_fused, cx,cy);
        end
    end

    global chi, parameters, energy_setting, grad_ctm_setting
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

    CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_fused_cell,chi,init,[],grad_ctm_setting)
    E_total, E_plaquatte_cell=evaluate_ob(parameters, U_phy, x, A_fused_cell, AA_fused_cell, CTM_cell, grad_ctm_setting);
    E=E_total/(Lx*Ly);
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
            A_unfused,A_fused,U_phy=build_A_checkerboard(x[cx, cy]);
            A_unfused_cell=fill_tuple(A_unfused_cell, A_unfused, cx,cy);
            A_fused_cell=fill_tuple(A_fused_cell, A_fused, cx,cy);
        end
    end

    global chi, parameters, energy_setting, grad_ctm_setting
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);

    CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_fused_cell,chi,init,[],grad_ctm_setting)
    E_total, E_plaquatte_cell=evaluate_ob(parameters, U_phy, x, A_fused_cell, AA_fused_cell, CTM_cell, grad_ctm_setting);
    E=E_total/(Lx*Ly);
    #println(E)
    println("E0= "*string(E));flush(stdout);
    global E_tem, CTM_tem
    CTM_tem=deepcopy(CTM_cell);
    E_tem=deepcopy(E)
    return E
end



function energy_CTM(x, chi, parameters, ctm_setting, init, init_CTM)
    global Lx,Ly,U_phy

    A_unfused_cell=initial_tuple_cell(Lx,Ly);
    A_fused_cell=initial_tuple_cell(Lx,Ly);

    for cx=1:Lx
        for cy=1:Ly
            global U_phy
        A_unfused,A_fused,U_phy=build_A_checkerboard(x[cx, cy]);
        A_unfused_cell=fill_tuple(A_unfused_cell, A_unfused, cx,cy);
        A_fused_cell=fill_tuple(A_fused_cell, A_fused, cx,cy);
        end
    end
    CTM_cell, AA_fused_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=CTMRG_cell(A_fused_cell,chi,init, init_CTM, ctm_setting);

    #@assert ite_err<3*(1e-5)

    E_total, E_plaquatte_cell=evaluate_ob(parameters, U_phy, x, A_fused_cell, AA_fused_cell, CTM_cell, ctm_setting);
    energy=E_total/(Lx*Ly);

    return real(energy), real(E_plaquatte_cell), ite_num, ite_err, CTM_cell
end




