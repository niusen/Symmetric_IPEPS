using TensorKit
import TensorKit: ×
using Zygote
using Zygote: @ignore_derivatives
using LinearAlgebra: I, diag, diagm, norm
using KrylovKit
using ChainRulesCore
using Random

const REPO = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
include(joinpath(REPO, "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(REPO, "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(REPO, "src", "bosonic", "Settings.jl"))
include(joinpath(REPO, "src", "bosonic", "Settings_cell.jl"))
include(joinpath(REPO, "src", "bosonic", "CTMRG.jl"))
include(joinpath(REPO, "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(REPO, "src", "bosonic", "square", "square_model.jl"))
include(joinpath(REPO, "src", "bosonic", "square", "simple_update_lib.jl"))
include(joinpath(REPO, "src", "bosonic", "square", "full_update_J1.jl"))
include(joinpath(REPO, "src", "bosonic", "square", "full_update_J1_cell.jl"))

Random.seed!(8128)
global Lx = 2
global Ly = 2
global chi = 8
global multiplet_tol = 1.0e-5
global projector_trun_tol = 1.0e-8
global backward_settings = Backward_settings()
global algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"

Vp = SU2Space(1 / 2 => 1)
Vv = SU2Space(0 => 1, 1 / 2 => 1)
A_set, _, _ = initial_iPEPS(Lx, Ly, Vp, Vv)
A_set = [A_set[cx, cy] / norm(A_set[cx, cy]) for cx in 1:Lx, cy in 1:Ly]

ctm_setting = LS_CTMRG_settings()
ctm_setting.CTM_conv_tol = 1.0e-4
# An initialized boundary is sufficient for comparing the two contraction
# graphs and keeps this probe independent of fork-specific CTM truncation.
ctm_setting.CTM_ite_nums = 0
ctm_setting.CTM_trun_tol = 1.0e-8
ctm_setting.svd_lanczos_tol = 1.0e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = false
ctm_setting.CTM_conv_info = true
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true

environment = _square_fu_environment_cell(A_set, chi, ctm_setting)
for bond in (
    SquareJ1CellBond(:x, CartesianIndex(1, 1), CartesianIndex(2, 1)),
    SquareJ1CellBond(:y, CartesianIndex(1, 1), CartesianIndex(1, 2)),
)
    A1, A2 = A_set[bond.site1], A_set[bond.site2]
    rho_reference = _square_fu_two_site_density_cell(
        environment.CTM, A1, A1, A2, A2, bond, Lx, Ly,
    )
    rho_streaming = _square_fu_two_site_density_cell_low_memory(
        environment.CTM, A1, A2, bond, Lx, Ly,
    )
    relative_error = norm(rho_streaming - rho_reference) / norm(rho_reference)
    println("$(bond.direction) density-matrix relative error = $relative_error")
    @assert relative_error < 1.0e-12
end

energy_reference = square_J1_energy_cell(A_set, environment; low_memory=false)
energy_streaming = square_J1_energy_cell(A_set, environment; low_memory=true)
energy_error = abs(energy_streaming.energy_per_site - energy_reference.energy_per_site)
println("energy reference = $(energy_reference.energy_per_site)")
println("energy streaming = $(energy_streaming.energy_per_site)")
println("energy absolute error = $energy_error")
@assert energy_error < 1.0e-12
