


function separate_gdoptimize(f, g!, fg!, x0::iPEPS_ansatz, linesearch, maxiter::Int = 500, g_rtol::Float64 = 1e-8, g_atol::Float64 = 1e-16) 
    global chi,D,Dx,Dy
    if @isdefined(D)
        println("D="*string(D));flush(stdout);
    elseif @isdefined(Dx)&@isdefined(Dy)
        println("Dx,Dy="*string([Dx,Dy]));flush(stdout);
    end
    println("chi="*string(chi));flush(stdout);
    x = deepcopy(x0)
    gvec = similar(x)
    g!(gvec, x)
    fx = f(x)
    gnorm = norm(gvec)
    gtol = max(g_rtol*gnorm, g_atol)

    # Univariate line search functions
    ϕ(α) = f(x + α*s)
    function dϕ(α)
        g!(gvec, x + α*s)
        return real(dot(gvec, s)) #I am not sure if taking real part is reasonable. If the output is complex the algorithm fails.
    end
    function ϕdϕ(α)
        phi = fg!(gvec, x + α*s)
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

function f_separate(x::iPEPS_ansatz)
    global CTM_tem,LS_ctm_setting,energy_setting
    if optim_setting.linesearch_CTM_method=="from_converged_CTM"
        init=initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true);
        CTM0=deepcopy(CTM_tem);
    elseif optim_setting.linesearch_CTM_method=="restart"
        init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
        CTM0=[];
    end
    if isa(x,Kagome_iPESS)
        E,E_up, E_down,ite_num,ite_err,_=energy_CTM_separate(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
        println("E= "*string(E)*", "*"E_up= "*string(E_up[:])*", "*"E_down= "*string(E_down[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    elseif isa(x,Checkerboard_iPESS)
        E,E_plaquatte_cell,ite_num,ite_err,_=energy_CTM_separate(x, chi, parameters, LS_ctm_setting, init, CTM0); 
        println("E= "*string(E)*", "*"E_plaquatte_cell= "*string(E_plaquatte_cell[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
    elseif isa(x,Square_iPEPS)
        if energy_setting.model =="triangle_J1_J2_Jchi"
            E,E_LU_RU_LD_set, E_LD_RU_RD_set, E_LU_LD_RD_set, E_LU_RU_RD_set,ite_num,ite_err,_=energy_CTM_separate(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"E_LU_RU_LD= "*string(E_LU_RU_LD_set[:])*", "*"E_LD_RU_RD "*string(E_LD_RU_RD_set[:])*", "*"E_LU_LD_RD= "*string(E_LU_LD_RD_set[:])*", "*"E_LU_RU_RD= "*string(E_LU_RU_RD_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinless_Hubbard";
            E, ex_set, ey_set, e0_set, ite_num,ite_err,_=energy_CTM_separate(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinless_Hubbard_pairing";
            E, ex_set, ey_set, px_set, py_set, e0_set, ite_num,ite_err,_=energy_CTM_separate(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"px_set= "*string(px_set[:])*", "*"py_set= "*string(py_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinless_triangle_lattice";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, ite_num,ite_err,_=energy_CTM_separate(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonal1_set= "*string(e_diagonal1_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        elseif energy_setting.model == "spinful_triangle_lattice";
            E, ex_set, ey_set, e_diagonal1_set, e0_set, eU_set, ite_num,ite_err,_=energy_CTM_separate(x, chi, parameters, LS_ctm_setting, energy_setting, init, CTM0); 
            println("E= "*string(E)*", "*"ex_set= "*string(ex_set[:])*", "*"ey_set= "*string(ey_set[:])*", "*"e_diagonal1_set= "*string(e_diagonal1_set[:])*", "*"e0_set= "*string(e0_set[:])*", "*"eU_set= "*string(eU_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        end

    end
    global E_history,save_filenm
    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        # filenm="Optim_cell_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
        #jldsave(filenm; B_a=x[1],B_b=x[2],B_c=x[3],T_u=x[4],T_d=x[5]);
        global psi;
        jldsave(save_filenm; x=psi);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end
    return E;
end

function g!_separate(gvec::iPEPS_ansatz, x) # this function changes the value of gvec
    println("compute grad")
    global E_tem, CTM_tem
    E_tem,∂E,CTM_tem=get_grad(x);
    #gvec=∂E;#this will not change the input variable
    Fields=fieldnames(typeof(gvec));
    for i in Fields
        Value=getfield(∂E, i)
        setfield!(gvec,i,Value)
    end
    # println("norm of grad: "*string(norm(gvec)))
    return gvec
end
function fg!_separate(gvec, x)
    #println("one fg!")
    g!_separate(gvec, x)
    f_separate(x)
end


function NamedTuple_to_Struc(∂E,x)
    ∂E_new=deepcopy(x);
    Keys=keys(∂E);
    for cc in Keys
        setfield!(∂E_new,cc,getindex(∂E,cc))
    end
    return ∂E_new
end


function get_grad(x::iPEPS_ansatz) 
    global Lx,Ly
    global px,py,psi

    x=normalize_tensor_group(x);
    #∂E = cost_fun'(x);
    

    ∂E=gradient(x ->cost_fun_separate(x), x)[1];#this works when x is a mutable structure. The output is a NamedTuple, not a structure, due to that the cost function takes out some fields of the input structure.
    # ∂E=NamedTuple_to_Struc_cell(∂E,x);
    #E=fun(state_vec)
    global E_tem, CTM_tem
    # x_tem=x;
    println("norm of grad: "*string(norm(∂E)))
    if isa(∂E, Vector{Float64})
        @assert !isnan(norm(∂E))
    elseif isa(∂E, Vector)
        for elem in ∂E
            @assert !isnan(norm(elem))
        end
    end
    
    return E_tem,∂E,CTM_tem
end





