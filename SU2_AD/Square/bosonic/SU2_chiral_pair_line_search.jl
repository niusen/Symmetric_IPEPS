using Revise, TensorKit, Zygote
using LinearAlgebra: I, diagm
using JLD2, ChainRulesCore
using KrylovKit
using JSON
using Random
using LineSearches, OptimKit
using Zygote: @ignore_derivatives
using Dates

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_chiral_pair_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "line_search_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "optimkit_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "stochastic_opt.jl"))

@show Random.seed!(555)

D = 9
chi = 54
optimizer = "backtracking" # "backtracking", "optimkit", or "stochastic"
@show optimizer

stochastic_delta = 1e-3
stochastic_maxiter = 100
stochastic_gtol = 1e-5


###########################
import LinearAlgebra.BLAS as BLAS
n_cpu=10;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C"*string(n_cpu)*"_D"*string(D))
pid=getpid();
println("pid="*string(pid));
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
###########################

J1 = 2 * cos(0.06 * pi) * cos(0.14 * pi)
J2 = 2 * cos(0.06 * pi) * sin(0.14 * pi)
Jchi = 2 * sin(0.06 * pi) * 2

parameters = Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)])

grad_ctm_setting = grad_CTMRG_settings()
grad_ctm_setting.CTM_conv_tol = 1e-6
grad_ctm_setting.CTM_ite_nums = 10
grad_ctm_setting.CTM_trun_tol = 1e-8
grad_ctm_setting.svd_lanczos_tol = 1e-8
grad_ctm_setting.projector_strategy = "4x4"
grad_ctm_setting.conv_check = "singular_value"
grad_ctm_setting.CTM_ite_info = true
grad_ctm_setting.CTM_conv_info = true
grad_ctm_setting.CTM_trun_svd = false
grad_ctm_setting.construct_double_layer = true
grad_ctm_setting.grad_checkpoint = true
dump(grad_ctm_setting)

LS_ctm_setting = LS_CTMRG_settings()
LS_ctm_setting.CTM_conv_tol = 1e-6
LS_ctm_setting.CTM_ite_nums = 80
LS_ctm_setting.CTM_trun_tol = 1e-8
LS_ctm_setting.svd_lanczos_tol = 1e-8
LS_ctm_setting.projector_strategy = "4x4"
LS_ctm_setting.conv_check = "singular_value"
LS_ctm_setting.CTM_ite_info = false
LS_ctm_setting.CTM_conv_info = true
LS_ctm_setting.CTM_trun_svd = false
LS_ctm_setting.construct_double_layer = true
LS_ctm_setting.grad_checkpoint = true
dump(LS_ctm_setting)

backward_settings = Backward_settings()
backward_settings.grad_inverse_tol = 1e-8
backward_settings.grad_regulation_epsilon = 1e-12
backward_settings.show_ite_grad_norm = false
dump(backward_settings)

optim_setting = Optim_settings()
optim_setting.init_statenm = "SU2_chiral_pair_D9_chi_54_-1.9538.jld2"
optim_setting.init_noise = 0.0
optim_setting.linesearch_CTM_method = "restart"
dump(optim_setting)

energy_setting = Square_Energy_settings()
energy_setting.model = "triangle_J1_J2_Jchi_chiral_pair"
dump(energy_setting)

global chi, multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = grad_ctm_setting.CTM_trun_tol

global backward_settings

global Vv
if D == 3
    Vv = SU2Space(0 => 1, 1 / 2 => 1)
elseif D == 4
    Vv = SU2Space(0 => 2, 1 / 2 => 1)
elseif D == 5
    Vv = SU2Space(0 => 1, 1 / 2 => 2)
elseif D == 6
    Vv = SU2Space(0 => 2, 1 / 2 => 2)
elseif D == 9
    Vv = SU2Space(0 => 2, 1 / 2 => 2, 1 => 1)
end
@assert dim(Vv) == D

global starting_time
starting_time = now()

function _chiral_pair_ctm_converged(ite_err, ctm_setting)
    if ismissing(ite_err)
        return true
    end
    return ite_err <= ctm_setting.CTM_conv_tol
end

function f(x::Square_iPEPS)
    global CTM_tem, LS_ctm_setting, optim_setting, chi, parameters

    if optim_setting.linesearch_CTM_method == "from_converged_CTM"
        init = initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true)
        CTM0 = deepcopy(CTM_tem)
    elseif optim_setting.linesearch_CTM_method == "restart"
        init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
        CTM0 = []
    else
        error("Unknown linesearch_CTM_method=" * string(optim_setting.linesearch_CTM_method))
    end

    E, E_triangles, obs, ite_num, ite_err, _ =
        energy_CTM_chiral_pair_with_observables(x, chi, parameters, LS_ctm_setting, init, CTM0)

    converged = _chiral_pair_ctm_converged(ite_err, LS_ctm_setting)
    println("E= " * string(E) *
            ", E_triangle= " * string(real.(collect(E_triangles))) *
            ", ctm_ite_num= " * string(ite_num) *
            ", ctm_ite_err= " * string(ite_err) *
            ", ctm_converged= " * string(converged))
    flush(stdout)

    if !converged
        println("Reject line-search trial because CTMRG did not converge.")
        flush(stdout)
        return Inf
    end

    global E_history, save_filenm
    if E < minimum(E_history)
        print_chiral_pair_observables(obs)
        E_history = vcat(E_history, E)
        jldsave(
            save_filenm;
            A=x.T,
            energy=E,
            parameters=parameters,
            observables=obs,
        )
        global starting_time
        Now = now()
        Time = Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(Now) - Dates.DateTime(starting_time)))
        println("Saved best chiral-pair state to " * save_filenm)
        println("Time consumed: " * string(Time))
        flush(stdout)
    end

    return E
end

function gdoptimize_chiral_pair(
    f,
    g!,
    fg!,
    x0::iPEPS_ansatz,
    linesearch,
    maxiter::Int=500,
    g_rtol::Float64=1e-8,
    g_atol::Float64=1e-16,
)
    global chi, D
    println("D=" * string(D))
    println("chi=" * string(chi))
    flush(stdout)

    x = deepcopy(x0)
    gvec = similar(x)
    g!(gvec, x)
    fx = f(x)
    gnorm = norm(gvec)
    gtol = max(g_rtol * gnorm, g_atol)

    phi(alpha) = f(x + alpha * s)
    function dphi(alpha)
        g!(gvec, x + alpha * s)
        return real(dot(gvec, s))
    end
    function phidphi(alpha)
        phi_value = fg!(gvec, x + alpha * s)
        dphi_value = real(dot(gvec, s))
        return (phi_value, dphi_value)
    end

    s = similar(gvec)
    iter = 0
    while iter < maxiter && gnorm > gtol
        println("optim iteration " * string(iter) *
                ", fx= " * string(fx) *
                ", grad_norm= " * string(gnorm))
        flush(stdout)

        x = normalize_ansatz(x)
        iter += 1
        s = (-1) * gvec
        dphi_0 = real(dot(s, gvec))
        alpha, fx = linesearch(phi, dphi, phidphi, 1.0, fx, dphi_0)
        println("accepted backtracking alpha = " * string(alpha))
        flush(stdout)

        x = x + alpha * s
        g!(gvec, x)
        gnorm = norm(gvec)
    end

    return fx, x, iter
end

function stochastic_grad_step_chiral_pair(gvec::Square_iPEPS, delta::Real)
    return Square_iPEPS(random_tensor_sign(gvec.T) * delta)
end

function stochastic_optimize_chiral_pair(
    x0::Square_iPEPS,
    delta::Real,
    maxiter::Int,
    gtol::Real,
)
    global chi, D
    println("stochastic gradient optimization")
    println("D=" * string(D))
    println("chi=" * string(chi))
    println("delta=" * string(delta))
    flush(stdout)

    x = deepcopy(x0)
    gvec = similar(x)
    gnorm = Inf
    iter = 0

    while iter < maxiter && gnorm > gtol
        println("\nstochastic iteration " * string(iter))
        x = normalize_ansatz(x)

        gvec = g!(gvec, x)
        gnorm = norm(gvec)
        random_step = stochastic_grad_step_chiral_pair(gvec, delta)
        x_updated = x - random_step

        println("norm of grad = " * string(gnorm))
        println("norm of random grad step = " * string(norm(x_updated - x)))
        flush(stdout)

        E_updated = f(x_updated)
        if isfinite(E_updated)
            x = x_updated
        else
            println("Reject stochastic step because the trial energy is not finite.")
            flush(stdout)
        end

        iter += 1
    end

    return x
end

init_complex_tensor = true
init_C4_symetry = false

state_vec = initial_SU2_chiral_pair_state(
    Vv,
    optim_setting.init_statenm,
    optim_setting.init_noise;
    init_complex_tensor=init_complex_tensor,
    init_C4_symetry=init_C4_symetry,
)
println("initial state virtual spaces: " *
        string((space(state_vec.T, 1), space(state_vec.T, 2), space(state_vec.T, 3), space(state_vec.T, 4))))
println("initial state physical space: " * string(space(state_vec.T, 5)))
flush(stdout)
state_vec = normalize_ansatz(state_vec)

global E_history
E_history = [10000.0]



global save_filenm
save_filenm = if optimizer == "optimkit"
    "OptimKit_SU2_chiral_pair_D" * string(D) * "_chi_" * string(chi) * ".jld2"
elseif optimizer == "stochastic"
    "Stochastic_SU2_chiral_pair_D" * string(D) * "_chi_" * string(chi) * ".jld2"
else
    "SU2_chiral_pair_D" * string(D) * "_chi_" * string(chi) * ".jld2"
end

function _scale_square_immutable(a::Square_iPEPS_immutable, beta::Number)
    return Square_iPEPS_immutable(a.T * beta)
end

function _add_square_immutable(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable)
    return Square_iPEPS_immutable(a.T + b.T)
end

function _sub_square_immutable(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable)
    return Square_iPEPS_immutable(a.T - b.T)
end

Base.:*(a::Square_iPEPS_immutable, beta::Number) = _scale_square_immutable(a, beta)
Base.:*(beta::Number, a::Square_iPEPS_immutable) = _scale_square_immutable(a, beta)
Base.:+(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable) = _add_square_immutable(a, b)
Base.:-(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable) = _sub_square_immutable(a, b)

function chiral_pair_cost_fun_optimkit(x::Matrix{Square_iPEPS_immutable})
    global chi, parameters, grad_ctm_setting
    @assert size(x) == (1, 1) "This chiral-pair OptimKit runner is for a 1x1 iPEPS unit cell."

    A = x[1, 1].T
    A = A / norm(A)

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err =
        chiral_pair_ctmrg(A, chi, init, [], grad_ctm_setting)

    E, E_triangles, triangles = evaluate_chiral_pair_triangle(
        A,
        AA,
        U_L,
        U_D,
        U_R,
        U_U,
        CTM,
        grad_ctm_setting,
        parameters,
    )
    E = real(E)

    println("  trial energy E= " * string(E) *
            ", E_triangle= " * string(real.(collect(E_triangles))) *
            ", ctm_ite_num= " * string(ite_num) *
            ", ctm_ite_err= " * string(ite_err))
    flush(stdout)

    global E_tem, CTM_tem
    E_tem = deepcopy(E)
    CTM_tem = deepcopy(CTM)
    return E
end

function chiral_pair_costfun_grad_optimkit(x::Matrix{Square_iPEPS_immutable})
    OPTIMKIT_FG_COUNT[] += 1
    println("\nOptimKit fg evaluation $(OPTIMKIT_FG_COUNT[])")
    println("  This is a cost/gradient evaluation, often a line-search trial; it is not necessarily an accepted optimization step.")
    flush(stdout)

    out = Zygote.withgradient(y -> chiral_pair_cost_fun_optimkit(y), x)
    E = out.val
    grad = NamedTuple_to_Struc_cell_optimkit(out.grad[1], x)

    grad_norm = sqrt(max(my_inner(x, grad, grad), 0.0))
    println("  trial grad_norm = " * string(grad_norm))
    flush(stdout)

    if E < minimum(E_history)
        global E_history
        E_history = vcat(E_history, E)

        A = x[1, 1].T / norm(x[1, 1].T)
        state = Square_iPEPS(A)
        init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
        E_save, _, obs, ite_num, ite_err, _ =
            energy_CTM_chiral_pair_with_observables(state, chi, parameters, LS_ctm_setting, init, [])

        global save_filenm
        jldsave(
            save_filenm;
            x=x,
            A=A,
            energy=E_save,
            parameters=parameters,
            observables=obs,
        )

        println("  accepted new best energy = " * string(E_save))
        print_chiral_pair_observables(obs)

        global starting_time
        now_time = now()
        elapsed = Dates.canonicalize(
            Dates.CompoundPeriod(Dates.DateTime(now_time) - Dates.DateTime(starting_time))
        )
        println("  Saved best chiral-pair state to " * save_filenm *
                ", ctm_ite_num= " * string(ite_num) *
                ", ctm_ite_err= " * string(ite_err))
        println("  Time consumed: " * string(elapsed))
        flush(stdout)
    end

    return E, grad
end

function chiral_pair_optimkit_op(x)
    x_opt, fx, gx, numfg, grad_history = optimize(
        chiral_pair_costfun_grad_optimkit,
        x,
        LBFGS(8; verbosity=3);
        inner=my_inner,
        retract=my_retract,
        scale! = my_scale!,
        add! = my_add!,
    )
    return x_opt, fx, gx, numfg, grad_history
end

if optimizer == "backtracking"
    ls = BackTracking(order=3)
    println(ls)
    fx_bt3, x_bt3, iter_bt3 = gdoptimize_chiral_pair(f, g!, fg!, state_vec, ls)
elseif optimizer == "optimkit"
    x0 = Matrix{Square_iPEPS_immutable}(undef, 1, 1)
    x0[1, 1] = Square_iPEPS_convert(state_vec)
    x_opt, fx, gx, numfg, grad_history = chiral_pair_optimkit_op(x0)
elseif optimizer == "stochastic"
    x_stochastic = stochastic_optimize_chiral_pair(
        state_vec,
        stochastic_delta,
        stochastic_maxiter,
        stochastic_gtol,
    )
else
    error("Unknown optimizer=" * string(optimizer) * ". Use \"backtracking\", \"optimkit\", or \"stochastic\".")
end
