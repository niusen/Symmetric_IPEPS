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
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_AD_SU2.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "line_search_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "optimkit_lib.jl"))

Random.seed!(555)

D = 3
chi = 54

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
LS_ctm_setting.CTM_ite_nums = 50
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
optim_setting.init_statenm = "nothing";#"D3.jld2"
optim_setting.init_noise = 0
optim_setting.linesearch_CTM_method = "from_converged_CTM"
dump(optim_setting)

energy_setting = Square_Energy_settings()
energy_setting.model = "triangle_J1_J2_Jchi"
dump(energy_setting)

global chi, parameters, energy_setting, grad_ctm_setting, LS_ctm_setting
global multiplet_tol, projector_trun_tol, backward_settings
multiplet_tol = 1e-5
projector_trun_tol = grad_ctm_setting.CTM_trun_tol

global Vv
if D == 3
    Vv = SU2Space(0=>1, 1/2=>1)
elseif D == 4
    Vv = SU2Space(0=>2, 1/2=>1)
elseif D == 5
    Vv = SU2Space(0=>1, 1/2=>2)
elseif D == 6
    Vv = SU2Space(0=>1, 1/2=>1, 1=>1)
elseif D == 8
    Vv = SU2Space(0=>1, 1/2=>2, 1=>1)
elseif D == 11
    Vv = SU2Space(0=>1, 1/2=>2, 1=>2)
elseif D == 16
    Vv = SU2Space(0=>1, 1/2=>3, 1=>3)
end
@assert dim(Vv) == D

function scale_square_ansatz(a::Square_iPEPS_immutable, beta::Number)
    return Square_iPEPS_immutable(a.T * beta)
end

function add_square_ansatz(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable)
    return Square_iPEPS_immutable(a.T + b.T)
end

function sub_square_ansatz(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable)
    return Square_iPEPS_immutable(a.T - b.T)
end

Base.:*(a::Square_iPEPS_immutable, beta::Number) = scale_square_ansatz(a, beta)
Base.:*(beta::Number, a::Square_iPEPS_immutable) = scale_square_ansatz(a, beta)
Base.:+(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable) = add_square_ansatz(a, b)
Base.:-(a::Square_iPEPS_immutable, b::Square_iPEPS_immutable) = sub_square_ansatz(a, b)

function square_optimkit_ctmrg(A, chi, init, init_CTM, ctm_setting)
    out = CTMRG(A, chi, init, init_CTM, ctm_setting)
    if length(out) == 8
        return out
    elseif length(out) == 6
        CTM, AA, U_L, U_D, U_R, U_U = out
        return CTM, AA, U_L, U_D, U_R, U_U, missing, missing
    else
        error("Unexpected CTMRG return length: " * string(length(out)))
    end
end

function square_cost_fun_optimkit(x::Matrix{Square_iPEPS_immutable})
    global chi, parameters, energy_setting, grad_ctm_setting
    @assert size(x) == (1, 1) "This square OptimKit runner is for a 1x1 iPEPS unit cell."
    @assert energy_setting.model == "triangle_J1_J2_Jchi"

    A = x[1, 1].T
    A = A / norm(A)

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err = square_optimkit_ctmrg(A, chi, init, [], grad_ctm_setting)

    H_Heisenberg, H123chiral, H12, H31, H23 = @ignore_derivatives Hamiltonians(space(A, 1))
    J1 = parameters["J1"]
    J2 = parameters["J2"]
    Jchi = parameters["Jchi"]
    H_triangle = J1 / 4 * (H12 + H31) + J2 / 2 * H23 + Jchi * H123chiral

    E_T1, E_T2, E_T3, E_T4 = evaluate_triangle(
        H_triangle,
        A,
        AA,
        U_L,
        U_D,
        U_R,
        U_U,
        CTM,
        grad_ctm_setting,
    )

    E = real(E_T1 + E_T2 + E_T3 + E_T4)
    println("E= " * string(E) *
            ", E_triangle= " * string(real.([E_T1, E_T2, E_T3, E_T4])) *
            ", ctm_ite_num= " * string(ite_num) *
            ", ctm_ite_err= " * string(ite_err))
    flush(stdout)

    global E_tem, CTM_tem
    E_tem = deepcopy(E)
    CTM_tem = deepcopy(CTM)
    return E
end

function square_costfun_grad_optimkit(x::Matrix{Square_iPEPS_immutable})
    OPTIMKIT_FG_COUNT[] += 1
    println("\noptim iteration $(OPTIMKIT_FG_COUNT[])")
    flush(stdout)

    out = Zygote.withgradient(y -> square_cost_fun_optimkit(y), x)
    E = out.val
    grad = NamedTuple_to_Struc_cell_optimkit(out.grad[1], x)

    grad_norm = sqrt(max(my_inner(x, grad, grad), 0.0))
    println("norm of grad = " * string(grad_norm))
    flush(stdout)

    if E < minimum(E_history)
        global E_history
        E_history = vcat(E_history, E)

        global save_filenm
        best_A = x[1, 1].T / norm(x[1, 1].T)
        jldsave(save_filenm; x=x, A=best_A, energy=E)

        global starting_time
        now_time = now()
        elapsed = Dates.canonicalize(
            Dates.CompoundPeriod(Dates.DateTime(now_time) - Dates.DateTime(starting_time))
        )
        println("Saved best state to " * save_filenm)
        println("Time consumed: " * string(elapsed))
        flush(stdout)
    end

    return E, grad
end

function square_optimkit_op(x)
    x_opt, fx, gx, numfg, grad_history = optimize(
        square_costfun_grad_optimkit,
        x,
        LBFGS(8; verbosity=3);
        inner=my_inner,
        retract=my_retract,
        scale! = my_scale!,
        add! = my_add!,
    )
    return x_opt, fx, gx, numfg, grad_history
end

init_complex_tensor = true
init_C4_symetry = false

state_vec = initial_SU2_state(
    Vv,
    optim_setting.init_statenm,
    optim_setting.init_noise,
    init_complex_tensor,
    init_C4_symetry,
)
state_vec = normalize_ansatz(state_vec)

x = Matrix{Square_iPEPS_immutable}(undef, 1, 1)
x[1, 1] = Square_iPEPS_convert(state_vec)

global save_filenm
save_filenm = "OptimKit_SU2_D$(D)_chi_$(chi).jld2"

global starting_time
starting_time = now()

global E_history
E_history = [10000.0]

x_opt, fx, gx, numfg, grad_history = square_optimkit_op(x)
