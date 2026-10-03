"""
GPU launcher for the bosonic square-lattice J1 Full Update.

Edit the configuration block and run this file directly. The CPU Full Update
launcher and all pre-existing CTMRG/Full Update source files are unchanged.
"""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

run_device = "cuda:0"
# All stages use `run_device` by default.  Set any individual stage to "cpu"
# when that stage exceeds GPU memory; the other stages remain on the GPU.
ctm_device = run_device
full_update_device = run_device
observable_device = run_device
print_gpu_memory = false

# Park inactive CTMRG tensors in CPU RAM; contractions and SVD stay on ctm_device.
# false/false uses the copied CTMRG's original all-device storage path.
ctm_offload_double_layer = true     # CPU AA/rotations; one GPU direction at a time
ctm_offload_intermediates = true    # CPU MM/RM while waiting for their next use
ctm_print_memory = true             # per-direction/projector memory diagnostics

initial_state_kind = :custom_matching
custom_matching = :y_staggered
custom_even_multiplets = [0 => 1, 1 => 1]  # D=4
custom_odd_multiplets = [1 // 2 => 2]      # D=4

# Relative paths are resolved from this GPU script directory.
initial_state_file = nothing
# initial_state_file = "Variational_J1_2x2_Dinit_4_chi_60_....jld2"

cell_Lx = 2
cell_Ly = 2
random_seed = 666
n_cpu = 4

Dmax = 4
multiplet_tolerance = 1.0e-5
environment_chi = 60
imaginary_time = 5.0
time_step = 0.02
J1 = 1.0

ctm_tolerance = 1.0e-6
ctm_max_iterations = 50
ctm_cell_method = "continuous_update"

als_sweeps = 10
als_convergence_tolerance = 1.0e-12
verbose = true

save_file = "FullUpdate_GPU_J1_$(cell_Lx)x$(cell_Ly)_Dmax_$(Dmax)_chi_$(environment_chi).jld2"

# ---------------------------------------------------------------------------

using TensorKit
import TensorKit: ×
using TensorKitSectors
using Zygote
using Zygote: @ignore_derivatives
using LinearAlgebra: BLAS, I, diag, diagm, dot, norm
using KrylovKit
using ChainRulesCore
using JLD2
using Random
using Dates
using Sockets: gethostname

device_specs = (run_device, ctm_device, full_update_device, observable_device)
if any(device -> lowercase(strip(device)) != "cpu", device_specs)
    using CUDA, cuTENSOR, Adapt
end

const GPU_RUN_DIR = @__DIR__
const SU2_AD_DIR = normpath(joinpath(GPU_RUN_DIR, "..", "..", "..", ".."))

include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "Settings.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "Settings_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "AD_lib.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "CTMRG.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_model.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "simple_update_lib.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "full_update_J1.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "full_update_J1_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_initial_states.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_configured_initial.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "gpu", "square_J1_gpu_utils.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "gpu", "square_J1_gpu_tensor_builders.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "gpu", "CTMRG_unitcell_offload.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "gpu", "square_J1_full_update_als.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "gpu", "square_J1_full_update_gpu.jl"))

BLAS.set_num_threads(n_cpu)
Random.seed!(random_seed)
Base.Sys.set_process_title("C$(n_cpu)_GPU_FU_J1_D$(Dmax)_chi$(environment_chi)")
square_J1_select_device!(run_device)

if isnothing(initial_state_file)
    (cell_Lx, cell_Ly) == (2, 2) || error(
        "named/custom parity matchings currently require a 2×2 cell",
    )
    T_set, _, _ = if initial_state_kind === :custom_matching
        square_J1_configured_matching_cell(
            custom_matching,
            custom_even_multiplets,
            custom_odd_multiplets;
            seed=random_seed,
        )
    else
        square_J1_named_initial_state(initial_state_kind, random_seed)
    end
    A_initial = [T_set[cx, cy] for cx in 1:cell_Lx, cy in 1:cell_Ly]
    source_description = string(initial_state_kind)
else
    source_file = isabspath(initial_state_file) ? initial_state_file :
        joinpath(GPU_RUN_DIR, initial_state_file)
    isfile(source_file) || error("initial_state_file does not exist: $source_file")
    A_initial = square_J1_load_configured_cell(source_file, cell_Lx, cell_Ly)
    source_description = source_file
end
_square_fu_validate_cell(A_initial)

initial_Dmax = maximum(
    dim(space(A_initial[cx, cy], leg))
    for cx in 1:cell_Lx, cy in 1:cell_Ly, leg in 1:4
)
save_filename = isabspath(save_file) ? save_file : joinpath(GPU_RUN_DIR, save_file)

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
    offload_double_layer=ctm_offload_double_layer,
    offload_intermediates=ctm_offload_intermediates,
    verbose=ctm_print_memory,
)

global Lx = cell_Lx
global Ly = cell_Ly
global chi = environment_chi
global multiplet_tol = multiplet_tolerance
global projector_trun_tol = ctm_setting.CTM_trun_tol
global backward_settings = Backward_settings()
global algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = ctm_cell_method

fu_settings = SquareJ1FullUpdateSettings(
    Dmax=Dmax,
    multiplet_tol=multiplet_tolerance,
    maxiter=als_sweeps,
    loss_tolerance=als_convergence_tolerance,
    refresh_environment=true,
    verbose=verbose,
)

println("PID=$(getpid())")
@show hostnm=gethostname()
println("number of cpus: $(BLAS.get_num_threads())")
println("Starting GPU bosonic square-lattice J1 Full Update")
println("  run_device=$run_device")
println("  ctm_device=$ctm_device")
println("  ctm_offload_double_layer=$ctm_offload_double_layer, ctm_offload_intermediates=$ctm_offload_intermediates")
println("  ctm_print_memory=$ctm_print_memory")
println("  full_update_device=$full_update_device")
println("  observable_device=$observable_device")
println("  initial_state=$source_description")
if isnothing(initial_state_file) && initial_state_kind === :custom_matching
    println("  custom_matching=$custom_matching")
    println("  custom_even_multiplets=$custom_even_multiplets")
    println("  custom_odd_multiplets=$custom_odd_multiplets")
end
println("  cell=$(cell_Lx)x$(cell_Ly), initial_Dmax=$initial_Dmax, Dmax=$Dmax")
println("  chi=$environment_chi, J1=$J1, tau=$imaginary_time, dt=$time_step")
println("  CTM_tol=$ctm_tolerance, CTM_maxiter=$ctm_max_iterations")
println("  als_sweeps=$als_sweeps, ALS_tol=$als_convergence_tolerance")
println("  local_optimizer=ALS (no automatic differentiation)")
println("  save_file=$save_filename")
println("initial virtual bonds:")
for group in square_J1_bond_groups(cell_Lx, cell_Ly), bond in group
    V = bond.direction === :x ?
        space(A_initial[bond.site1], 3)' : space(A_initial[bond.site1], 2)
    Dstar = sum(dim(V, sector) for sector in sectors(V))
    println(
        "  $(bond.direction)$(Tuple(bond.site1))->$(Tuple(bond.site2)): " *
        "D*=$Dstar, D=$(dim(V)), V=$V",
    )
end
print_gpu_memory && square_J1_print_device_memory("Initial CUDA memory:")
flush(stdout)

starting_time = now()
best_energy = Ref(Inf)

function save_and_measure_gpu(A_set_now, environment, step, reports)
    energies = square_J1_gpu_energy_cell(
        A_set_now,
        environment;
        J1,
        device=observable_device,
        low_memory=true,
        debug_memory=print_gpu_memory,
    )
    println(
        "E= $(energies.energy_per_site), " *
        "ex_set= $(energies.Ex[:]), ey_set= $(energies.Ey[:])",
    )
    if energies.energy_per_site < best_energy[]
        best_energy[] = energies.energy_per_site
        jldsave(
            save_filename;
            A_set=A_set_now,
            A_cell=square_fu_cell_to_tuple(A_set_now),
            Lx=cell_Lx,
            Ly=cell_Ly,
            initial_D=initial_Dmax,
            Dmax,
            init_kind=initial_state_kind,
            init_filename=source_description,
            multiplet_tol=multiplet_tolerance,
            chi=environment_chi,
            J1,
            tau=imaginary_time,
            dt=time_step,
            completed_sweeps=step,
            reports,
            Ex=energies.Ex,
            Ey=energies.Ey,
            energy_per_site=energies.energy_per_site,
            run_device,
            ctm_device,
            full_update_device,
            observable_device,
            ctm_offload_double_layer,
            ctm_offload_intermediates,
        )
        elapsed = Dates.canonicalize(Dates.CompoundPeriod(now() - starting_time))
        println("Saved lower-energy CPU checkpoint; time consumed: $elapsed")
    end
    print_gpu_memory && square_J1_print_device_memory("CUDA memory after sweep $step:")
    flush(stdout)
    return nothing
end

A_final, environment_final, history = square_J1_full_update_cell_gpu(
    A_initial,
    environment_chi,
    imaginary_time,
    time_step,
    ctm_setting;
    J1,
    settings=fu_settings,
    ctm_device,
    full_update_device,
    ctm_memory,
    callback=save_and_measure_gpu,
)

println("GPU Full Update finished; best measured energy/site=$(best_energy[])")
flush(stdout)
