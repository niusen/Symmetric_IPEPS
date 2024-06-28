

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

        global n_mps_sweep;
        n_mps_sweep=0;
        gvec0=gradient(x ->cost_fun_global(x), x)[1];

        #convert gvec to Matrix{Triangle_iPESS}
        Lx,Ly=size(gvec);
        gvec=Matrix{Triangle_iPESS}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                gvec[cx,cy]=Triangle_iPESS(gvec0[cx,cy].Bm, gvec0[cx,cy].Tm);
            end
        end

        #gvec=g!(gvec,x);#get grad
        x_updated=x-get_random_grad(gvec,delta);#get random grad
        println("norm of random grad:"*string(norm(x_updated-x)))




        E_updated=fx(x_updated,Lx,Ly);
        x=x_updated;


        global use_canonical_form
        if use_canonical_form
            println("convert to canonical form")
            x,_=fermiPEPS_gauge_fix_simple(x,100);
            # psi_double,_=construct_double_layer_swap_new(psi,Lx,Ly);
        end




        iter += 1
        gnorm = norm(gvec);
    end
    return x
end




function stochastic_opt(x0::Matrix{TensorMap}, delta, maxiter, gtol) 
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
        x=normalize_tensor_group(x);

        global n_mps_sweep;
        n_mps_sweep=0;
        gvec0=gradient(x ->cost_fun_global(x), x)[1];

        #convert gvec to Matrix{Triangle_iPESS}
        Lx,Ly=size(gvec);
        gvec=Matrix{TensorMap}(undef,Lx,Ly);
        for cx=1:Lx
            for cy=1:Ly
                gvec[cx,cy]=gvec0[cx,cy];
            end
        end

        #gvec=g!(gvec,x);#get grad
        x_updated=x-get_random_grad(gvec,delta);#get random grad
        println("norm of random grad:"*string(norm(x_updated-x)))


        E_updated=fx(x_updated);
        x=x_updated;

        iter += 1
        gnorm = norm(gvec);
    end
    return x
end

function fx(x::Matrix{T},Lx,Ly) where T<:iPEPS_ansatz
    global n_mps_sweep;
    n_mps_sweep=5;

    if isa(x[1],Triangle_iPESS)
        psi_PEPS=PESS_to_PEPS_matrix(x);
        psi_double,_=construct_double_layer_swap(psi_PEPS,psi_PEPS,Lx,Ly);
        E,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_old(psi_PEPS,psi_double)
        println("E_total="*string(E));
        println(Ex_set)
        println(Ey_set)
        println(E_ld_ru_set)
        println(occu_set)
        println(EU_set);flush(stdout);
        
        E=real(E);
 
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




function fx(x::Matrix{TensorMap}) 
    global n_mps_sweep;
    n_mps_sweep=5;

    global psi_double
    E,Ex_set,Ey_set,E_ld_ru_set,occu_set,EU_set=energy_disk_global(x,psi_double);

    println("E_total="*string(E));
    println(Ex_set)
    println(Ey_set)
    println(E_ld_ru_set)
    println(occu_set)
    println(EU_set);flush(stdout);
    
    E=real(E);
 
    
    global E_history,E_all_history,delta_history
    E_all_history=vcat(E_all_history,E);
    delta_history=vcat(delta_history,delta);
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        # filenm="Optim_cell_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
        #jldsave(filenm; B_a=x[1],B_b=x[2],B_c=x[3],T_u=x[4],T_d=x[5]);
        global save_filenm
        jldsave(save_filenm; psi=x,E_all_history,E_history,delta_history);
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

function get_random_grad(x::Matrix,delta) 
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
        elseif isa(x[cc],TensorMap)
            A=ansatz;
            A=random_tensor_sign(A)*delta;
            ansatz_new=A;
        else
            error("unknown type")
        end
        x_new[cc]=ansatz_new;
    end
    return x_new
end





