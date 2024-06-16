


function gdoptimize_2site(psi_left,psi_double,   f, g!, fg!, x0::TensorMap, linesearch, maxiter::Int = 20, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 
    global chi,Dmax
    println("Dmax="*string(Dmax));flush(stdout);
    println("chi="*string(chi));flush(stdout);
    x = deepcopy(x0)
    gvec = similar(x)
    g!(gvec, x, psi_left,psi_double)
    fx = f(x, psi_left,psi_double)
    gnorm = norm(gvec)
    gtol = max(g_rtol*gnorm, g_atol)

    # Univariate line search functions
    ϕ(α) = f(x + α*s, psi_left,psi_double)
    function dϕ(α)
        g!(gvec, x + α*s, psi_left,psi_double)
        return real(dot(gvec, s)) #I am not sure if taking real part is reasonable. If the output is complex the algorithm fails.
    end
    function ϕdϕ(α)
        phi = fg!(gvec, x + α*s, psi_left,psi_double)
        dphi = real(dot(gvec, s)) #I am not sure if taking real part is reasonable. If the output is complex the algorithm fails.
        return (phi, dphi)
    end

    s = similar(gvec) # Step direction

    iter = 0
    while iter < maxiter && gnorm > gtol
        println("optim iteration "*string(iter))
        x=x/norm(x);

        iter += 1
        s = (-1)*gvec

        dϕ_0 = real(dot(s, gvec))
        #α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)
        α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1/3, fx, dϕ_0)

        x = x + α*s
        g!(gvec, x, psi_left,psi_double)
        gnorm = norm(gvec)
    end

    return (fx, x, iter)
end


function f_2site(x::TensorMap,psi_left,psi_double)
    global n_mps_sweep
    n_mps_sweep=5;#for line search

    E=cost_fun_bond(x,psi_left,psi_double); 

    println("E= "*string(E));flush(stdout);

    global E_history#,save_opt_filenm
    global px,py
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        #save_filenm="Optim_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"

        # jldsave(save_opt_filenm; psi,px,py,x);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end


function g!_2site(gvec::TensorMap, x, psi_left,psi_double)# this function changes the value of gvec
    println("compute grad")
    global n_mps_sweep
    n_mps_sweep=0;
    
    E_tem,∂E=get_grad_2site(x,psi_left,psi_double);
    #gvec=∂E;#this will not change the input variable

    for (k,block) in blocks(gvec)
        copyto!(block,blocks(∂E)[k]);
    end

    println("norm of grad: "*string(norm(gvec)))
    return gvec
end


function get_grad_2site(x::TensorMap,psi_left,psi_double)
    global n_mps_sweep
    n_mps_sweep=0;

    ∂E=gradient(x ->cost_fun_bond(x,psi_left,psi_double), x)[1];

    #E=fun(state_vec)
    global E_tem
    # x_tem=x;

    # println("norm of grad: "*string(norm(∂E)))
    if isa(∂E, Vector{Float64})
        @assert !isnan(norm(∂E))
    elseif isa(∂E, Vector)
        for elem in ∂E
            @assert !isnan(norm(elem))
        end
    end
    
    return E_tem,∂E
end





function fg!_2site(gvec, x,psi_left,psi_double)
    #println("one fg!")
    g!_2site(gvec, x,psi_left,psi_double)
    f_2site(x,psi_left,psi_double)
end







