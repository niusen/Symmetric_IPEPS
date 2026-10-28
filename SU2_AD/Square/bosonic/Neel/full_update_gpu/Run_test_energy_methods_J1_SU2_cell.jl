"""
Small-D comparison of legacy AA_open energy and reduced-bond energy.

Edit settings here and run directly. No SU/FU/variational optimization is run.
For each case, build CTM ONCE and pass that same environment to both methods.
The script was written for server execution; it is not automatically run.
"""

# ---------------------------------------------------------------------------
# Configuration (no command-line arguments)
# ---------------------------------------------------------------------------
run_device = "cpu"             # or "cuda:0" / "cuda:1"
n_cpu = 2
source_dir = nothing            # auto-find a parent containing src/bosonic;
                               # alternatively set the server's source directory.
initial_state_file = nothing    # filename next to THIS script, or absolute path
# initial_state_file = "FullUpdate_GPU_J1_2x2_Dmax_4_chi_60.jld2"

# Ignored when loading a state. Cover mixed parity and both bond orientations.
test_initial_states = [:mixed_min, :minimal_x_staggered, :minimal_y_staggered]
use_complex_random_state = true # independent real/imaginary tensors, not a global phase
random_seed = 666
cell_Lx = 2
cell_Ly = 2
environment_chi = 16
ctm_tolerance = 1.0e-8
ctm_max_iterations = 50
J1 = 1.0
comparison_atol = 1.0e-10
comparison_rtol = 1.0e-9
maximum_test_D = 4             # guard: legacy AA_open can OOM at large D
print_gpu_memory = false
# No input checkpoint is overwritten. A timestamped comparison report is saved.
# ---------------------------------------------------------------------------

using TensorKit
import TensorKit: ×
using TensorKitSectors
using Zygote                    # dependencies of existing includes; no gradient call
using Zygote: @ignore_derivatives
using LinearAlgebra: BLAS, I, diag, diagm, dot, norm
using KrylovKit
using ChainRulesCore
using JLD2
using Random
using Dates
using Sockets: gethostname

if lowercase(strip(run_device)) != "cpu"
    using CUDA, cuTENSOR, Adapt
end

if isnothing(source_dir)
    global source_dir = let candidate = @__DIR__
        while !isfile(joinpath(candidate, "src", "bosonic", "Settings.jl"))
            parent = dirname(candidate)
            parent == candidate && error("Cannot locate sources; set source_dir in this script")
            candidate = parent
        end
        candidate
    end
end
isfile(joinpath(source_dir, "src", "bosonic", "Settings.jl")) ||
    error("source_dir does not contain src/bosonic/Settings.jl: $source_dir")

include(joinpath(source_dir, "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(source_dir, "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(source_dir, "src", "bosonic", "Settings.jl"))
include(joinpath(source_dir, "src", "bosonic", "Settings_cell.jl"))
include(joinpath(source_dir, "src", "bosonic", "AD_lib.jl"))
include(joinpath(source_dir, "src", "bosonic", "CTMRG.jl"))
include(joinpath(source_dir, "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "square_model.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "simple_update_lib.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "full_update_J1.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "full_update_J1_cell.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "square_J1_initial_states.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "square_J1_configured_initial.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "gpu", "square_J1_gpu_utils.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "gpu", "square_J1_gpu_tensor_builders.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "gpu", "CTMRG_unitcell_offload.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "gpu", "square_J1_full_update_als.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "gpu", "square_J1_bond_energy_reduced.jl"))
include(joinpath(source_dir, "src", "bosonic", "square", "gpu", "square_J1_full_update_gpu.jl"))

BLAS.set_num_threads(n_cpu)
Random.seed!(random_seed)
Base.Sys.set_process_title("C$(n_cpu)_test_J1_energy_methods")
square_J1_select_device!(run_device)
println("PID=$(getpid())")
@show hostnm=gethostname()
println("number of cpus: $(BLAS.get_num_threads()), device=$run_device")
println("cell=$(cell_Lx)x$(cell_Ly), chi=$environment_chi, J1=$J1, maximum_test_D=$maximum_test_D")
println("atol=$comparison_atol, rtol=$comparison_rtol, random_seed=$random_seed")
println("source_dir=$source_dir, initial_state_file=$initial_state_file")

ctm_setting = LS_CTMRG_settings()
ctm_setting.CTM_conv_tol = ctm_tolerance
ctm_setting.CTM_ite_nums = ctm_max_iterations
ctm_setting.CTM_trun_tol = 1.0e-8
ctm_setting.svd_lanczos_tol = 1.0e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = false
ctm_setting.CTM_conv_info = true
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true
ctm_setting.grad_checkpoint = false
ctm_memory = SquareJ1CTMMemorySettings(
    offload_double_layer=true, offload_intermediates=true,
    intermediate_min_bytes=2048 * 1024^2, verbose=false,
)
global Lx = cell_Lx
global Ly = cell_Ly
global chi = environment_chi
global multiplet_tol = 1.0e-5
global projector_trun_tol = ctm_setting.CTM_trun_tol
global backward_settings = Backward_settings()
global algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"

cases = isnothing(initial_state_file) ? test_initial_states : [:loaded]
isempty(cases) && error("test_initial_states is empty")
results = NamedTuple[]
report_file = joinpath(@__DIR__,
    "Energy_methods_comparison_$(cell_Lx)x$(cell_Ly)_chi$(environment_chi)_$(Dates.format(now(), "yyyy_mm_dd_HH_MM_SS")).jld2")

for (case_index, kind) in pairs(cases)
    A_set, description = if kind === :loaded
        state_path = isabspath(initial_state_file) ? initial_state_file :
            joinpath(@__DIR__, initial_state_file)
        square_J1_load_configured_cell(state_path, cell_Lx, cell_Ly), state_path
    else
        (cell_Lx, cell_Ly) == (2, 2) || error("named test initial states require a 2x2 cell")
        real_set, _, _ = square_J1_named_initial_state(kind, random_seed + 2 * case_index)
        if use_complex_random_state
            imag_set, _, _ = square_J1_named_initial_state(kind, random_seed + 2 * case_index + 1)
            complex_set = [real_set[x, y] + im * imag_set[x, y]
                           for x in 1:cell_Lx, y in 1:cell_Ly]
            [complex_set[x, y] / norm(complex_set[x, y])
             for x in 1:cell_Lx, y in 1:cell_Ly], string(kind, "_complex")
        else
            [real_set[x, y] for x in 1:cell_Lx, y in 1:cell_Ly], string(kind, "_real")
        end
    end
    _square_fu_validate_cell(A_set)
    actual_Dmax = maximum(dim(space(A, leg)) for A in A_set for leg in 1:4)
    actual_Dmax <= maximum_test_D || error(
        "D=$actual_Dmax exceeds maximum_test_D=$maximum_test_D; use a small-D state for the legacy comparison",
    )
    println("\nTesting $description: actual_Dmax=$actual_Dmax")
    flush(stdout)

    # Build ONCE. Convergence affects physical accuracy, not equality of these
    # two contractions of the very same finite-chi network.
    environment = square_J1_gpu_environment_cell(
        A_set, environment_chi, ctm_setting; device=run_device, ctm_memory,
    )
    println("ctm_ite_num=$(environment.ite_num), ctm_ite_err=$(environment.ite_err)")
    ctm_converged = !ismissing(environment.ite_err) &&
                    isfinite(environment.ite_err) && environment.ite_err < ctm_tolerance
    if !ctm_converged
        @warn "CTM has not met tolerance; comparing the same unconverged CTM is still an algebraic check"
    end
    comparison = square_J1_compare_energy_methods(
        A_set, environment; J1, device=run_device, atol=comparison_atol,
        rtol=comparison_rtol, debug_memory=print_gpu_memory,
    )
    push!(results, (; description, actual_Dmax, ctm_converged,
                    ctm_iterations=environment.ite_num, ctm_error=environment.ite_err,
                    comparison))
    jldsave(report_file; results, cell_Lx, cell_Ly, environment_chi, J1,
            run_device, random_seed, comparison_atol, comparison_rtol,
            ctm_tolerance, ctm_max_iterations, maximum_test_D,
            test_initial_states, use_complex_random_state, initial_state_file)
    A_set = environment = nothing
    square_J1_reclaim_device_memory!(aggressive=true)
end

println("Comparison report: $report_file")
all(result -> result.comparison.passed, results) ||
    error("Energy-method comparison FAILED: see per-bond differences above")
println("PASS: all x/y bonds and total energies agree within the specified tolerances.")
flush(stdout)
