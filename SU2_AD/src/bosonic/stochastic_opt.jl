

function stochastic_opt(x0::Matrix{T}, delta, maxiter, gtol) where T<:iPEPS_ansatz
    global chi,D
    println("stochastic optimization")
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    
    x = deepcopy(x0);
    gvec = similar(x);
    gnorm=10000;
    iter = 0
    while iter < maxiter && gnorm > gtol
        println("optim iteration "*string(iter))
        x=normalize_ansatz(x);

        gvec=g!(gvec,x);#get grad
        x_updated=x-get_random_grad(gvec,delta);#get random grad
        println("norm of random grad:"*string(norm(x_updated-x)))
        E_updated=fx(x_updated);
        x=x_updated;

        iter += 1
        gnorm = norm(gvec);
    end
    return x
end

function fx(x::Matrix{T}) where T<:iPEPS_ansatz
    global CTM_tem,LS_ctm_setting,energy_setting
    if optim_setting.linesearch_CTM_method=="from_converged_CTM"
        init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
        CTM0=deepcopy(CTM_tem);
    elseif optim_setting.linesearch_CTM_method=="restart"
        init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
        CTM0=[];
    end
    if isa(x[1],Kagome_iPESS)
        E,E_up, E_down,ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
        println("E= "*string(E)*", "*"E_up= "*string(E_up[:])*", "*"E_down= "*string(E_down[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    elseif isa(x[1],Checkerboard_iPESS)
        E,E_plaquatte_cell,ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, init, CTM0); 
        println("E= "*string(E)*", "*"E_plaquatte_cell= "*string(E_plaquatte_cell[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    elseif isa(x[1],Triangle_iPESS)
        if energy_setting.model == "spinful_triangle_lattice";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonal1_set= "*string(e_diagonal1_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"eU_set= "*string(eU_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
            println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
        elseif energy_setting.model == "standard_triangle_Hubbard";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, triangle_up_set, triangle_dn_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
            println("E= "*string(E));flush(stdout);
            println("ex_set= "*string(ex_set[:])); flush(stdout);
            println("ey_set= "*string(ey_set[:]));flush(stdout);
            println("e_diagonal1_set= "*string(e_diagonal1_set[:]));flush(stdout);
            println("e0_set= "*string(e0_set[:]));flush(stdout);
            println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
            println("eU_set= "*string(eU_set[:])); flush(stdout);
            println("triangle_up_set= "*string(triangle_up_set[:])); flush(stdout);
            println("triangle_dn_set= "*string(triangle_dn_set[:])); flush(stdout);
            
        end
    elseif isa(x[1],Square_iPEPS)
        if energy_setting.model =="triangle_J1_J2_Jchi"
            E,E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set,ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"E_LU_RU_LD= "*string(E_LU_RU_LD_set[:])*", "*"E_LD_RU_RD "*string(E_LD_RU_RD_set[:])*", "*"E_LU_LD_RD= "*string(E_LU_LD_RD_set[:])*", "*"E_LU_RU_RD= "*string(E_LU_RU_RD_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinless_Hubbard";
            E, ex_set, ey_set, e0_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinless_Hubbard_pairing";
            E, ex_set, ey_set, px_set, py_set, e0_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"px_set= "*string(px_set[:])*", "*"py_set= "*string(py_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinless_triangle_lattice";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonal1_set= "*string(e_diagonal1_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinful_triangle_lattice";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonal1_set= "*string(e_diagonal1_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"eU_set= "*string(eU_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
            println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
        elseif energy_setting.model == "standard_triangle_Hubbard";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, triangle_up_set, triangle_dn_set, ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            # println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonal1_set= "*string(e_diagonal1_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"eU_set= "*string(eU_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
            # println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
            println("ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
            println("E= "*string(E));flush(stdout);
            println("ex_set= "*string(ex_set[:])); flush(stdout);
            println("ey_set= "*string(ey_set[:]));flush(stdout);
            println("e_diagonal1_set= "*string(e_diagonal1_set[:]));flush(stdout);
            println("e0_set= "*string(e0_set[:]));flush(stdout);
            println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);
            println("eU_set= "*string(eU_set[:])); flush(stdout);
            println("triangle_up_set= "*string(triangle_up_set[:])); flush(stdout);
            println("triangle_dn_set= "*string(triangle_dn_set[:])); flush(stdout);
        end

    end
    global E_history,E_all_history,delta_history
    E_all_history=vcat(E_all_history,E);
    delta_history=vcat(delta_history,delta);
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        # filenm="Optim_cell_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
        #jldsave(filenm; B_a=x[1],B_b=x[2],B_c=x[3],T_u=x[4],T_d=x[5]);
        global save_filenm
        jldsave(save_filenm; x,E_all_history,E_history,delta_history);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end





function random_tensor_sign(T::TensorMap)
    T=deepcopy(T);
    function generate_number(a)
        if a>0
            b=rand(1);
        elseif a<0
            b=-rand(1);
        elseif a==0
            b=0;
        end
        return b[1]
    end
    if sectortype(space(T,1)) == Trivial
        mm=T.data;
        for dd in eachindex(mm)
            a=mm[dd];
            if isa(a,Float64)
                a_new=generate_number(a);
            elseif isa(a,ComplexF64)
                a_new=generate_number(real(a))+im*generate_number(imag(a));
            else
                error("unknown number type")
            end
            mm[dd]=a_new;
        end
        T=TensorMap(mm,codomain(T),domain(T));
    else
        for cc=1:length(T.data.values)
            mm=T.data.values[cc];
            for dd in eachindex(mm)
                a=mm[dd];
                if isa(a,Float64)
                    a_new=generate_number(a);
                elseif isa(a,ComplexF64)
                    a_new=generate_number(real(a))+im*generate_number(imag(a));
                else
                    error("unknown number type")
                end
                mm[dd]=a_new;
            end
            T.data.values[cc]=mm;
        end
    end
    return T
end

function get_random_grad(x::Matrix{T},delta) where T<:iPEPS_ansatz
    x_new=deepcopy(x);
    for cc in eachindex(x)
        ansatz=x[cc];
        if isa(x[cc],Kagome_iPESS)
            B1=ansatz.B1;
            B2=ansatz.B2;
            B3=ansatz.B3;
            Tup=ansatz.Tup;
            Tdn=ansatz.Tdn;

            B1=random_tensor_sign(B1)*delta;
            B2=random_tensor_sign(B2)*delta;
            B3=random_tensor_sign(B3)*delta;
            Tup=random_tensor_sign(Tup)*delta;
            Tdn=random_tensor_sign(Tdn)*delta;
            ansatz_new=Kagome_iPESS(B1,B2,B3,Tup,Tdn);
        elseif isa(x[cc],Checkerboard_iPESS)
            BL=ansatz.B_L;
            BU=ansatz.B_U;
            Tm=ansatz.Tm;

            BL=random_tensor_sign(BL)*delta;
            BU=random_tensor_sign(BU)*delta;
            Tm=random_tensor_sign(Tm)*delta;
            ansatz_new=Checkerboard_iPESS(BL,BU,Tm);
        elseif isa(x[cc],Triangle_iPESS)
            iPEss=x[cc];
            bm=iPEss.Bm;
            tm=iPEss.Tm;
            bm=random_tensor_sign(bm)*delta;
            tm=random_tensor_sign(tm)*delta;
            ansatz_new=Triangle_iPESS(bm,tm);
        elseif isa(x[cc],Square_iPEPS)
            A=ansatz.T;
            A=random_tensor_sign(A)*delta;
            ansatz_new=Square_iPEPS(A);
        else
            error("unknown type")
        end
        x_new[cc]=ansatz_new;
    end
    return x_new
end





