# my_retract is not an in place function which should not change x
function my_retract(x,dx,α::Number)
    ψ = deepcopy(x)
    ψ=ψ+ dx*α
    #env = leading_boundary(ψ, alg_ctm,env)
    return ψ,dx
end


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



###############################
function unwrap_thunk(x)
    if isa(x, Thunk)
        return unthunk(x)
    else
        return x
    end
end

function zero_like_field(x)
    try
        return zero(x)
    catch
        return 0 * x
    end
end

function zero_struct_like(x::T) where T
    fns = fieldnames(T)
    vals = map(fn -> zero_like_field(getfield(x, fn)), fns)
    return T(vals...)
end

function NamedTuple_to_Struc_special_optimkit(∂E::NamedTuple, x::T) where T
    fns = fieldnames(T)

    vals = map(fns) do fn
        if haskey(∂E, fn)
            unwrap_thunk(getfield(∂E, fn))
        else
            zero_like_field(getfield(x, fn))
        end
    end

    return T(vals...)
end

function NamedTuple_to_Struc_cell_optimkit(∂E, x)
    ∂E_new = similar(x)

    for cc in eachindex(x)
        if ∂E[cc] === nothing
            ∂E_new[cc] = zero_struct_like(x[cc])
        else
            ∂E_new[cc] = NamedTuple_to_Struc_special_optimkit(∂E[cc], x[cc])
        end
    end

    return ∂E_new
end
#######################################
function scale_ansatz(a::Triangle_iPESS_immutable, β::Number)
    return Triangle_iPESS_immutable(a.Bm * β, a.Tm * β)
end

function add_ansatz(a::Triangle_iPESS_immutable, b::Triangle_iPESS_immutable)
    return Triangle_iPESS_immutable(a.Bm + b.Bm, a.Tm + b.Tm)
end

function sub_ansatz(a::Triangle_iPESS_immutable, b::Triangle_iPESS_immutable)
    return Triangle_iPESS_immutable(a.Bm - b.Bm, a.Tm - b.Tm)
end

Base.:*(a::Triangle_iPESS_immutable, β::Number) = scale_ansatz(a, β)
Base.:*(β::Number, a::Triangle_iPESS_immutable) = scale_ansatz(a, β)
Base.:+(a::Triangle_iPESS_immutable, b::Triangle_iPESS_immutable) = add_ansatz(a, b)
Base.:-(a::Triangle_iPESS_immutable, b::Triangle_iPESS_immutable) = sub_ansatz(a, b)

function real_inner(a, b)
    return real(dot(a, b))
end

function real_inner(a::Triangle_iPESS_immutable, b::Triangle_iPESS_immutable)
    return real_inner(a.Tm, b.Tm) + real_inner(a.Bm, b.Bm)
end

function real_inner(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable)
    return real_inner(a.T, b.T)
end

function real_inner(a::AbstractArray, b::AbstractArray)
    s = 0.0
    @assert size(a) == size(b)

    for i in eachindex(a, b)
        s += real_inner(a[i], b[i])
    end

    return s
end

my_inner(x, dx1, dx2) = real_inner(dx1, dx2)


function cost_fun_optimkit(x::Matrix{T})  where T<:iPEPS_ansatz_immutable #variational parameters are vector of TensorMap
    global Lx,Ly
    global chi, parameters, energy_setting, grad_ctm_setting

    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            if isa(x[cx,cy],Square_iPEPS_immutable)
                A=x[cx,cy].T;
            elseif isa(x[cx,cy],Triangle_iPESS_immutable)
                tm=x[cx,cy].Tm;#|LU><M|
                bm=x[cx,cy].Bm;#|Md><|RD
                A=permute(tm*bm,(1,5,4,2,3,));#L,D,R,U,d,
            end
            norm_A=norm(A)
            A=A/norm_A;

            A_cell=fill_tuple(A_cell, A, cx,cy);
        end
    end
    
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true);
    

    if energy_setting.model =="simple_test";
        E=simple_cfun(A_cell);
        E=real(E);
        
    elseif energy_setting.model in ("Triangle_Hofstadter_Hubbard_spinHall",);
        CTM_cell, AA_cell, U_L_cell,U_D_cell,U_R_cell,U_U_cell,ite_num,ite_err=Fermionic_CTMRG_cell(A_cell,chi,init,[],grad_ctm_setting);

        E_total,  ex_up_set, ey_up_set, e_diagonala_up_set, ex_dn_set, ey_dn_set, e_diagonala_dn_set, e0_set, eU_set, sx_set,sy_set,sz_set=evaluate_ob_cell(parameters, A_cell::Tuple, AA_cell, CTM_cell, LS_ctm_setting, energy_setting);
        E=real(E_total);
        println("E= "*string(E));flush(stdout);
        println("ex_up_set= "*string(ex_up_set[:])*", "*"ey_up_set= "*string(ey_up_set[:])*", "*"e_diagonal1_up_set= "*string(e_diagonala_up_set[:]));flush(stdout);
        println("ex_dn_set= "*string(ex_dn_set[:])*", "*"ey_dn_set= "*string(ey_dn_set[:])*", "*"e_diagonal1_dn_set= "*string(e_diagonala_dn_set[:]));flush(stdout);
        println("e0_set= "*string(e0_set[:])*", "*"eU_set= "*string(eU_set[:])*", "*"ctm_ite_num= "*string(ite_num)*", "*"ctm_ite_err= "*string(ite_err));flush(stdout);
        println("occu="*string(sum(e0_set)/length(e0_set)));flush(stdout);

        S2=sqrt.(sx_set.^2+sy_set.^2+sz_set.^2);
        println("S2= "*string(abs.(S2))*", sx= "*string(sx_set)*", sy= "*string(sy_set)*", sz= "*string(sz_set));flush(stdout); 
    end



    println("E0= "*string(E));flush(stdout);
    global E_tem
    E_tem=deepcopy(E)



    # global E_history
    # if E<minimum(E_history)
    #     E_history=vcat(E_history,E);
    #     # filenm="Optim_cell_LS_D_"*string(D)*"_chi_"*string(chi)*".jld2"
    #     #jldsave(filenm; B_a=x[1],B_b=x[2],B_c=x[3],T_u=x[4],T_d=x[5]);
    #     global save_filenm
    #     jldsave(save_filenm; x);
    #     global starting_time
    #     Now=now();
    #     Time=Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)));
    #     println("Time consumed: "*string(Time));flush(stdout);
    # end


    return E
end

# function costfun_grad(x::Matrix{T}) where T<:iPEPS_ansatz_immutable #variational parameters are vector of TensorMap

#     E=cost_fun_optimkit(x::Matrix{T});
#     ∂E=gradient(x ->cost_fun_optimkit(x), x)[1];#this works when x is a mutable structure. The output is a NamedTuple, not a structure, due to that the cost function takes out some fields of the input structure.


#     ∂E = NamedTuple_to_Struc_cell_optimkit(∂E, x);

#     grad_norm = sqrt(max(my_inner(x, ∂E, ∂E), 0.0))
#     println("norm of grad = ", grad_norm)

#     return E, ∂E

# end

function costfun_grad(x::Matrix{T}) where T<:iPEPS_ansatz_immutable
    out = Zygote.withgradient(y -> cost_fun_optimkit(y), x)

    E = out.val
    ∂E = out.grad[1]
    ∂E = NamedTuple_to_Struc_cell_optimkit(∂E, x)

    grad_norm = sqrt(max(my_inner(x, ∂E, ∂E), 0.0))
    println("norm of grad = ", grad_norm)
    flush(stdout)

    if E < minimum(E_history)
        global E_history
        E_history = vcat(E_history, E)

        global save_filenm
        jldsave(save_filenm; x)

        global starting_time
        Now = now()
        Time = Dates.canonicalize(
            Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time))
        )
        println("Time consumed: " * string(Time)); flush(stdout)
    end

    return E, ∂E
end

"""
    LBFGS(m::Int = 8; 
          acceptfirst::Bool = true,
          maxiter::Int=MAXITER[], # 1_000_000
          gradtol::Real=GRADTOL[], # 1e-8
          verbosity::Int=VERBOSITY[], # 1
          ls_maxiter::Int=LS_MAXITER[], # 10
          ls_maxfg::Int=LS_MAXFG[], # 20
          ls_verbosity::Int=LS_VERBOSITY[], # 1
          linesearch = HagerZhangLineSearch(maxiter=ls_maxiter, maxfg=ls_maxfg, verbosity=ls_verbosity))

LBFGS optimization algorithm.

## Parameters
- `m::Int`: The number of previous iterations to store for the limited memory BFGS approximation.
- `maxiter::Int`: The maximum number of iterations.
- `gradtol::T`: The tolerance for the norm of the gradient.
- `verbosity::Int`: The verbosity level of the optimization algorithm.
- `acceptfirst::Bool`: Whether to accept the first step of the line search.
- `ls_maxiter::Int`: The maximum number of iterations for the line search.
- `ls_maxfg::Int`: The maximum number of function evaluations for the line search.
- `ls_verbosity::Int`: The verbosity level of the line search algorithm.
- `linesearch`: The line search algorithm to use; if a custom value is provided,
  it overrides `ls_maxiter`, `ls_maxfg`, and `ls_verbosity`.

Both `verbosity` and `ls_verbosity` use the following scheme:
- 0: no output
- 1: only warnings upon non-convergence
- 2: convergence information at the end of the algorithm
- 3: progress information after each iteration
- 4: more detailed information (only for the linesearch)
"""

function optimkit_op(state_vec)
    x_opt, fx, gx, numfg, grad_history=optimize(
        costfun_grad, 
        state_vec,
        LBFGS(8;); 
        inner=my_inner,
        retract=my_retract,
        scale! = my_scale!,
        add! = my_add!
    )
    return x_opt, fx, gx, numfg, grad_history
end
