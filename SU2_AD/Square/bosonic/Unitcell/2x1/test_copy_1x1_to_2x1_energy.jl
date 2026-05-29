using TensorKit, Zygote
using LinearAlgebra: I, diagm, norm
using JLD2, ChainRulesCore
using KrylovKit
using Zygote: @ignore_derivatives

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_model.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_model_cell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_AD_SU2.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "Settings_cell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_large_cell_optimization.jl"))

default_state = joinpath(@__DIR__, "..", "..", "Optim_RVB_coefficients_D3_chi_54_-0.9834.jld2")
init_statenm = length(ARGS) >= 1 ? ARGS[1] : default_state
chi = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 54

Lx = 2
Ly = 1

J1 = 2 * cos(0.06 * pi) * cos(0.14 * pi)
J2 = 2 * cos(0.06 * pi) * sin(0.14 * pi)
Jchi = 2 * sin(0.06 * pi) * 2
parameters = Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)])

LS_ctm_setting = LS_CTMRG_settings()
LS_ctm_setting.CTM_conv_tol = 1e-6
LS_ctm_setting.CTM_ite_nums = 150
LS_ctm_setting.CTM_trun_tol = 1e-8
LS_ctm_setting.svd_lanczos_tol = 1e-8
LS_ctm_setting.projector_strategy = "4x4"
LS_ctm_setting.conv_check = "singular_value"
LS_ctm_setting.CTM_ite_info = false
LS_ctm_setting.CTM_conv_info = true
LS_ctm_setting.CTM_trun_svd = false
LS_ctm_setting.construct_double_layer = true
LS_ctm_setting.grad_checkpoint = true

energy_setting = Square_Energy_settings()
energy_setting.model = "triangle_J1_J2_Jchi"

algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"

global chi, Lx, Ly, multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = LS_ctm_setting.CTM_trun_tol
global algrithm_CTMRG_settings

function _load_single_site_square_tensor(statenm::AbstractString)
    data = load(statenm)
    if haskey(data, "A")
        return data["A"]
    elseif haskey(data, "x")
        x = data["x"]
        A = _as_square_tensor(x[1])
        return A
    elseif haskey(data, "A_cell")
        return _as_square_tensor(data["A_cell"][1][1])
    end
    error("State file must contain `A`, `x`, or `A_cell`.")
end

function _copy_single_site_to_2x1(A)
    state = Matrix{Square_iPEPS}(undef, 2, 1)
    state[1, 1] = Square_iPEPS(deepcopy(A))
    state[2, 1] = Square_iPEPS(deepcopy(A))
    return state
end

A = _load_single_site_square_tensor(init_statenm)
A = A / norm(A)
D = dim(space(A, 1))
@assert space(A, 2) == space(A, 1)
@assert space(A, 3) == space(A, 1)'
@assert space(A, 4) == space(A, 1)'
@assert space(A, 5) == SU2Space(1 / 2 => 1)'

println("Test copying a 1x1 SU2 square CSL state into a 2x1 cell")
println("  state file      = " * init_statenm)
println("  D               = " * string(D))
println("  chi             = " * string(chi))
println("  virtual spaces  = " * string((space(A, 1), space(A, 2), space(A, 3), space(A, 4))))
println("  physical space  = " * string(space(A, 5)))
flush(stdout)

single_state = Square_iPEPS(A)
init_single = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
E_1x1, E_T1, E_T2, E_T3, E_T4, ite_num_1x1, ite_err_1x1, _ =
    energy_CTM(single_state, chi, parameters, LS_ctm_setting, energy_setting, init_single, [])

cell_state = _copy_single_site_to_2x1(A)
init_cell = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
E_2x1, obs_2x1, ite_num_2x1, ite_err_2x1, _, _ =
    square_large_cell_energy_with_observables(cell_state, LS_ctm_setting, init_cell, [])

println("\n1x1 energy:")
println("  E          = " * string(E_1x1))
println("  E_triangle = " * string(real.([E_T1, E_T2, E_T3, E_T4])))
println("  ctm_ite_num= " * string(ite_num_1x1))
println("  ctm_ite_err= " * string(ite_err_1x1))

println("\n2x1 copied-cell energy:")
println("  E          = " * string(E_2x1))
println("  E_triangle = " * string(real.(obs_2x1.E_triangle[:])))
println("  ctm_ite_num= " * string(ite_num_2x1))
println("  ctm_ite_err= " * string(ite_err_2x1))
print_square_large_cell_observables(obs_2x1)

println("\nenergy consistency:")
println("  E_2x1 - E_1x1 = " * string(E_2x1 - E_1x1))
println("  relative diff = " * string(abs(E_2x1 - E_1x1) / max(abs(E_1x1), 1e-14)))
flush(stdout)

nothing
