# my_retract is not an in place function which should not change x
function my_retract(x,dx,α::Number)
    ψ = deepcopy(x)
    ψ=ψ+ dx*α
    #env = leading_boundary(ψ, alg_ctm,env)
    return ψ,dx
end

my_inner(x,dx1,dx2) = real(dot(dx1,dx2))

function my_add!(Y, X, a)
    for cc=1:length(Y)
        setindex!(Y,Y[cc]+a*X[cc],cc);#this will change the input variable
    end
    return Y
end

function my_scale!(η, β)
    for cc=1:length(η)
        setindex!(η,η[cc]*β,cc);#this will change the input variable
    end
    return η
end


function cfun(x)
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
