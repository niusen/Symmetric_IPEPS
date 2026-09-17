"""
Small-fixed-D variational optimization of the 2×2 SU(2) square-J1 iPEPS.

Edit the configuration below and run `julia Run_variational_J1_SU2_cell.jl`.
The saved `A_set` is already an iPEPS state: load it directly in
`Run_full_update_J1_SU2_cell.jl` and choose a larger `Dmax` there.
"""

using TensorKit
import TensorKit: ×
using TensorKitSectors
using Zygote
using Zygote: @ignore_derivatives
using LinearAlgebra: BLAS, I, diag, diagm, dot, norm
using KrylovKit
using ChainRulesCore
using OptimKit
using JLD2
using Random
using Dates
using Sockets: gethostname

const NEEL_DIR = @__DIR__
const SU2_AD_DIR = normpath(joinpath(NEEL_DIR, "..", "..", ".."))

include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "Settings.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "Settings_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "AD_lib.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "optimkit_lib.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "CTMRG.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_model.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "simple_update_lib.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "full_update_J1.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "full_update_J1_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_initial_states.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_configured_initial.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_measurements_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_variational_cell.jl"))

# ---------------------------------------------------------------------------
# Configuration.  This script deliberately optimizes fixed virtual spaces.
# ---------------------------------------------------------------------------

# Use the explicitly configured even/odd spaces below by default.  The named
# :minimal_y_staggered seed has odd space 1/2=>1 and ignores custom_* fields.
initial_state_kind = :custom_matching
# This is a random state; set initial_state_file to resume an optimized iPEPS.
custom_matching = :y_staggered
custom_even_multiplets = [0 => 1, 1 => 1]  # D=4
custom_odd_multiplets = [1 // 2 => 2]      # D=4

initial_state_file = nothing
# initial_state_file = "Variational_J1_2x2_...jld2"

cell_Lx = 2
cell_Ly = 2
random_seed = 666
n_cpu = 4
environment_chi = 24
J1 = 1.0
multiplet_tolerance = 1.0e-5

ctm_tolerance = 1.0e-6
ctm_max_iterations = 50
ctm_cell_method = "continuous_update"
ctm_checkpoint = true

max_variational_iterations = 10
gradient_tolerance = 1.0e-6

# A new timestamped checkpoint prevents an unrelated run from being replaced.
output_file = nothing

# ---------------------------------------------------------------------------

BLAS.set_num_threads(n_cpu)
Random.seed!(random_seed)

if isnothing(initial_state_file)
    (cell_Lx, cell_Ly) == (2, 2) || error(
        "named/custom parity matchings currently require a 2×2 cell",
    )
    T_set, _, _ = if initial_state_kind === :custom_matching
        square_J1_configured_matching_cell(
            custom_matching, custom_even_multiplets, custom_odd_multiplets;
            seed=random_seed,
        )
    else
        square_J1_named_initial_state(initial_state_kind, random_seed)
    end
    A_initial = [T_set[cx, cy] for cx in 1:cell_Lx, cy in 1:cell_Ly]
    source_description = string(initial_state_kind)
else
    source_file = isabspath(initial_state_file) ? initial_state_file :
        joinpath(NEEL_DIR, initial_state_file)
    isfile(source_file) || error("initial_state_file does not exist: $source_file")
    A_initial = square_J1_load_configured_cell(source_file, cell_Lx, cell_Ly)
    source_description = source_file
end
_square_fu_validate_cell(A_initial)

initial_Dmax = maximum(
    dim(space(A_initial[cx, cy], leg))
    for cx in 1:cell_Lx, cy in 1:cell_Ly, leg in 1:4
)
stamp = Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")
default_output = "Variational_J1_$(cell_Lx)x$(cell_Ly)_Dinit_$(initial_Dmax)_chi_$(environment_chi)_$(stamp).jld2"
save_filename = isnothing(output_file) ? joinpath(NEEL_DIR, default_output) :
    (isabspath(output_file) ? output_file : joinpath(NEEL_DIR, output_file))
isfile(save_filename) && error("refusing to overwrite existing state: $save_filename")

grad_ctm_setting = grad_CTMRG_settings()
LS_ctm_setting = LS_CTMRG_settings()
for setting in (grad_ctm_setting, LS_ctm_setting)
    setting.CTM_conv_tol = ctm_tolerance
    setting.CTM_ite_nums = ctm_max_iterations
    setting.CTM_trun_tol = 1.0e-8
    setting.svd_lanczos_tol = 1.0e-8
    setting.projector_strategy = "4x4"
    setting.conv_check = "singular_value"
    setting.CTM_ite_info = false
    setting.CTM_conv_info = true
    setting.CTM_trun_svd = false
    setting.construct_double_layer = true
    setting.grad_checkpoint = ctm_checkpoint
end

square_J1_prepare_ctm_globals!(
    A_initial, environment_chi, grad_ctm_setting;
    multiplet_tolerance=multiplet_tolerance,
    cell_method=ctm_cell_method,
)
H_Heisenberg, _, _, _, _ = Hamiltonians(space(A_initial[1, 1], 1))
H = permute(H_Heisenberg, (1, 2), (3, 4))

println("PID=$(getpid())")
@show hostnm=gethostname()
println("Starting fixed-space square-J1 variational optimization")
println("  initial_state=$source_description")
if isnothing(initial_state_file) && initial_state_kind === :custom_matching
    println("  custom_matching=$custom_matching")
    println("  custom_even_multiplets=$custom_even_multiplets")
    println("  custom_odd_multiplets=$custom_odd_multiplets")
end
println("  cell=$(cell_Lx)x$(cell_Ly), initial_Dmax=$initial_Dmax, chi=$environment_chi")
println("  J1=$J1, CTM_tol=$ctm_tolerance, CTM_maxiter=$ctm_max_iterations")
println("  CTM_checkpoint=$ctm_checkpoint, max_iterations=$max_variational_iterations")
println("  output_file=$save_filename")
println("initial virtual bonds:")
for group in square_J1_bond_groups(cell_Lx, cell_Ly), bond in group
    V = bond.direction === :x ?
        space(A_initial[bond.site1], 3)' : space(A_initial[bond.site1], 2)
    println("  $(bond.direction)$(Tuple(bond.site1))->$(Tuple(bond.site2)): D=$(dim(V)), V=$V")
end
flush(stdout)

function save_variational_step(A_set, evaluation, energy, ctm_iterations, ctm_error)
    jldsave(
        save_filename;
        A_set,
        energy,
        evaluation,
        ctm_iterations,
        ctm_error,
        initial_state=source_description,
        initial_state_kind,
        custom_matching,
        custom_even_multiplets,
        custom_odd_multiplets,
        random_seed,
        cell_Lx,
        cell_Ly,
        initial_Dmax,
        environment_chi,
        J1,
        multiplet_tolerance,
    )
    println("saved variational state at fg evaluation $evaluation: E=$energy")
    flush(stdout)
    return nothing
end

A_final, E_final, E_best, numfg, grad_history = square_J1_variational_optimize_cell(
    A_initial,
    H,
    environment_chi,
    grad_ctm_setting,
    LS_ctm_setting;
    J1,
    max_iterations=max_variational_iterations,
    gradient_tolerance,
    callback=save_variational_step,
)
println("Final OptimKit energy/site=$E_final, best saved LS energy/site=$E_best, numfg=$numfg")
println("For Full Update, set initial_state_file=\"$save_filename\" and Dmax > $initial_Dmax")
flush(stdout)
