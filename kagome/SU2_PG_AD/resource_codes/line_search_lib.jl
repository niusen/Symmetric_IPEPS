using LinearAlgebra: norm, dot

function gdoptimize(f, g!, fg!, x0::Vector{TensorMap}, linesearch, maxiter::Int = 20, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 
    global chi,D
    println("D="*string(D));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    x = deepcopy(x0)
    gvec = similar(x)
    g!(gvec, x)
    fx = f(x)
    gnorm = norm(gvec)
    gtol = max(g_rtol*gnorm, g_atol)

    # Univariate line search functions
    ϕ(α) = f(x .+ α.*s)
    function dϕ(α)
        g!(gvec, x .+ α.*s)
        return real(dot(gvec, s)) #I am not sure if taking real part is reasonable. If the output is complex the algorithm fails.
    end
    function ϕdϕ(α)
        phi = fg!(gvec, x .+ α.*s)
        dphi = real(dot(gvec, s)) #I am not sure if taking real part is reasonable. If the output is complex the algorithm fails.
        return (phi, dphi)
    end

    s = similar(gvec) # Step direction

    iter = 0
    while iter < maxiter && gnorm > gtol
        println("optim iteration "*string(iter))
        x=normalize_tensor_group(x);

        iter += 1
        s = (-1)*gvec

        dϕ_0 = dot(s, gvec)
        #α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
        α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1/5, fx, dϕ_0)

        x = x + α*s
        g!(gvec, x)
        gnorm = norm(gvec)
    end

    return (fx, x, iter)
end

function f(x)
    global CTM_tem,LS_ctm_setting
    if optim_setting.linesearch_CTM_method=="from_converged_CTM"
        init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
        CTM0=deepcopy(CTM_tem);
    elseif optim_setting.linesearch_CTM_method=="restart"
        init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
        CTM0=[];
    end
    E,E_up, E_down,ite_num,ite_err,_=energy_CTM(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
    println("E= "*string(E)*", "*"E_up= "*string(real(E_up))*", "*"E_down= "*string(real(E_down))*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    global E_history
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
        jldsave(filenm; B_a=x[1],B_b=x[2],B_c=x[3],T_u=x[4],T_d=x[5]);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end

function g!(gvec, x)# this function changes the value of gvec
    println("compute grad")
    global E_tem, CTM_tem
    E_tem,∂E,CTM_tem=get_grad(x);
    #gvec=∂E;#this will not change the input variable
    for cc=1:length(gvec)
        setindex!(gvec,∂E[cc],cc);#this will change the input variable
    end
    return gvec
end

function fg!(gvec, x)
    #println("one fg!")
    g!(gvec, x)
    f(x)
end


