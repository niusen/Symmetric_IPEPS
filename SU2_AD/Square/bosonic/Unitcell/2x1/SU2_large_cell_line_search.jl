using TensorKit, Zygote
using LinearAlgebra: I, diagm, norm, dot
using JLD2, ChainRulesCore
using KrylovKit
using Random
using LineSearches, OptimKit
using Zygote: @ignore_derivatives
using Dates

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_model.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_model_cell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "Settings_cell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "optimkit_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "line_search_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "line_search_lib_cell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "stochastic_opt.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_large_cell_optimization.jl"))

Random.seed!(555)

D = 3
chi = 40
Lx = 2
Ly = 1
optimizer = "optimkit" # "backtracking", "optimkit", or "stochastic"

optimkit_maxiter = 500
optimkit_verbosity = 3

backtracking_maxiter = 500
backtracking_grad_rtol = 1e-8
backtracking_grad_atol = 1e-16

stochastic_delta = 1e-3
stochastic_maxiter = 100
stochastic_gtol = 1e-5

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
optim_setting.init_statenm = "nothing"
optim_setting.init_noise = 0.0
optim_setting.linesearch_CTM_method = "restart" # "restart" or "from_converged_CTM"
dump(optim_setting)

energy_setting = Square_Energy_settings()
energy_setting.model = "triangle_J1_J2_Jchi"
dump(energy_setting)

algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"
dump(algrithm_CTMRG_settings)

global chi, Lx, Ly, multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = grad_ctm_setting.CTM_trun_tol
global backward_settings, algrithm_CTMRG_settings

Vv = square_SU2_virtual_space(D)
@assert dim(Vv) == D

init_complex_tensor = true
state_vec = initial_SU2_large_cell_state(
    Vv,
    optim_setting.init_statenm,
    optim_setting.init_noise,
    init_complex_tensor,
)
state_vec = normalize_ansatz(state_vec)

println("initial state virtual spaces:")
for cx in 1:Lx, cy in 1:Ly
    A = state_vec[cx, cy].T
    println("  site " * string((cx, cy)) * ": " *
            string((space(A, 1), space(A, 2), space(A, 3), space(A, 4))))
end
println("initial state physical space: " * string(space(state_vec[1, 1].T, 5)))
flush(stdout)

global save_filenm
save_filenm = if optimizer == "optimkit"
    "OptimKit_SU2_cell_" * string(Lx) * "x" * string(Ly) * "_D" *
        string(D) * "_chi_" * string(chi) * ".jld2"
elseif optimizer == "stochastic"
    "Stochastic_SU2_cell_" * string(Lx) * "x" * string(Ly) * "_D" *
        string(D) * "_chi_" * string(chi) * ".jld2"
else
    "SU2_cell_" * string(Lx) * "x" * string(Ly) * "_D" *
        string(D) * "_chi_" * string(chi) * ".jld2"
end

global starting_time
starting_time = now()

global E_history
E_history = [10000.0]

if optimizer == "backtracking"
    ls = BackTracking(order=3)
    println(ls)
    fx_bt3, x_bt3, iter_bt3 = gdoptimize_square_large_cell(
        large_cell_f,
        large_cell_g!,
        large_cell_fg!,
        state_vec,
        ls;
        maxiter=backtracking_maxiter,
        g_rtol=backtracking_grad_rtol,
        g_atol=backtracking_grad_atol,
    )
elseif optimizer == "optimkit"
    x0 = immutable_square_cell(state_vec)
    x_opt, fx, gx, numfg, grad_history = large_cell_optimkit_op(
        x0;
        maxiter=optimkit_maxiter,
        verbosity=optimkit_verbosity,
    )
elseif optimizer == "stochastic"
    x_stochastic = stochastic_optimize_square_large_cell(
        state_vec,
        stochastic_delta,
        stochastic_maxiter,
        stochastic_gtol,
    )
else
    error("Unknown optimizer=" * string(optimizer))
end

nothing
