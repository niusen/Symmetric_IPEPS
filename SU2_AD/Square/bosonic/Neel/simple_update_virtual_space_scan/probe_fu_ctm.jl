using TensorKit
import TensorKit: ×
using Zygote
using LinearAlgebra: I, diag, diagm, dot, norm
using KrylovKit
using ChainRulesCore
using JLD2
using Random
using Zygote: @ignore_derivatives

repo = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
include(joinpath(repo, "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(repo, "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(repo, "src", "bosonic", "Settings.jl"))
include(joinpath(repo, "src", "bosonic", "Settings_cell.jl"))
include(joinpath(repo, "src", "bosonic", "CTMRG.jl"))
include(joinpath(repo, "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(repo, "src", "bosonic", "square", "square_model.jl"))
include(joinpath(repo, "src", "bosonic", "square", "simple_update_lib.jl"))
include(joinpath(repo, "src", "bosonic", "square", "full_update_J1.jl"))
include(joinpath(repo, "src", "bosonic", "square", "full_update_J1_cell.jl"))

Random.seed!(666)
global Lx = 2
global Ly = 2
global chi = 8
global multiplet_tol = 1e-5
global projector_trun_tol = 1e-8
global backward_settings = Backward_settings()
global algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"

Vp = SU2Space(1 / 2 => 1)
Vv = SU2Space(0 => 1, 1 / 2 => 1)
A_original, _, _ = initial_iPEPS(Lx, Ly, Vp, Vv)
A_original = [A_original[cx, cy] / norm(A_original[cx, cy]) for cx in 1:Lx, cy in 1:Ly]

ctm_setting = LS_CTMRG_settings()
ctm_setting.CTM_conv_tol = 1e-4
ctm_setting.CTM_ite_nums = 5
ctm_setting.CTM_trun_tol = 1e-8
ctm_setting.svd_lanczos_tol = 1e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = true
ctm_setting.CTM_conv_info = true
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true

environment = _square_fu_environment_cell(A_original, chi, ctm_setting)
gate = prepare_gate_Heisenberg(0.01, "SU2")
settings = SquareJ1FullUpdateSettings(
    Dmax=parse(Int, get(ENV, "FU_PROBE_DMAX", "3")),
    multiplet_tol=1e-5,
    maxiter=parse(Int, get(ENV, "FU_PROBE_LOCAL_MAXITER", "1")),
    verbose=true,
)

probe_direction = Symbol(get(ENV, "FU_PROBE_DIRECTION", "both"))
all_bonds = (
    SquareJ1CellBond(:x, CartesianIndex(1, 1), CartesianIndex(2, 1)),
    SquareJ1CellBond(:y, CartesianIndex(1, 1), CartesianIndex(1, 2)),
)
probe_bonds = probe_direction === :both ? all_bonds :
    filter(bond -> bond.direction === probe_direction, all_bonds)
for bond in probe_bonds
    A_work = copy(A_original)
    A1_new, A2_new, report = square_J1_full_update_cell_bond(
        A_work, environment, gate, bond; settings=settings,
    )
    A_work[bond.site1] = A1_new
    A_work[bond.site2] = A2_new
    _square_fu_validate_cell(A_work)
    println(
        "CTM reduced FU ", bond.direction,
        ": loss ", report.loss_initial, " -> ", report.loss_final,
        ", space ", report.old_bond_space, " -> ", report.new_bond_space,
    )
    if parse(Bool, get(ENV, "FU_PROBE_REBUILD", "false"))
        ctm_setting.CTM_ite_nums = 1
        rebuilt = _square_fu_environment_cell(A_work, chi, ctm_setting)
        println("rebuilt CTM after ", bond.direction, " update; error=", rebuilt.ite_err)
    end
end
