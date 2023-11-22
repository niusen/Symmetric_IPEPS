# my_retract is not an in place function which should not change x
function my_retract(x,dx,α::Number)
    ψ = deepcopy(x)
    ψ=ψ+ dx*α
    #env = leading_boundary(ψ, alg_ctm,env)
    return ψ,dx
end

my_inner(x,dx1,dx2) = real(dot(dx1,dx2))

function my_add!(Y::Array, X, a)
    for cc=1:length(Y)
        setindex!(Y,Y[cc]+a*X[cc],cc);#this will change the input variable
    end
    return Y
end

function my_add!(Y::iPEPS_ansatz, X, a)
    Fields=fieldnames(typeof(Y));
    for i in Fields
        setfield!(Y,i,getfield(Y, i)+a*getfield(X, i))
    end
    return Y
end

function my_scale!(η::Array, β)
    for cc=1:length(η)
        setindex!(η,η[cc]*β,cc);#this will change the input variable
    end
    return η
end

function my_scale!(η::iPEPS_ansatz, β)
    Fields=fieldnames(typeof(η));
    for i in Fields
        setfield!(η,i,β*getfield(η, i))
    end
    return η
end


function cfun(x::Kagome_iPESS)
    global E_history
    E,∂E,_=get_grad(x)

    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        filenm="OptimKit_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
        jldsave(filenm; B_a=x[1],B_b=x[2],B_c=x[3],T_u=x[4],T_d=x[5]);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end

    return E,∂E
end

function cfun(x::Square_iPEPS)
    global E_history
    E,∂E,_=get_grad(x)

    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        filenm="OptimKit_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
        jldsave(filenm; A=x.T);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end

    return E,∂E
end

function cfun(x::Matrix{T}) where T<:iPEPS_ansatz
    global E_history
    E,∂E,_=get_grad(x)

    if E<minimum(E_history)
        E_history=vcat(E_history,E);
        filenm="Optim_cell_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
        jldsave(filenm; x);
        global starting_time
        Now=now();
        Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
        println("Time consumed: "*string(Time));flush(stdout);
    end

    return E,∂E
end


function optimkit_op(state_vec)
    optimize(
        cfun, 
        state_vec,
        ConjugateGradient(verbosity=3); 
        inner=my_inner,
        retract=my_retract,
        scale! = my_scale!,
        add! = my_add!
    )
    return ψ
end
