using TensorKit
import TensorKit: ×
using TensorKitSectors
using LinearAlgebra
using JLD2
using MAT
using Random
using Dates

println("PID=$(getpid())")
flush(stdout)

const OBS_DIR = @__DIR__
const NEEL_DIR = normpath(joinpath(OBS_DIR, ".."))
const SCAN_DIR = joinpath(NEEL_DIR, "simple_update_virtual_space_scan")

include(joinpath(SCAN_DIR, "scan_lib.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_correl_cell.jl"))
include(joinpath(OBS_DIR, "observables_lib.jl"))

# ---------------------------------------------------------------------------
# Configuration: edit these values directly; no command-line input is needed.
# ---------------------------------------------------------------------------

input_state_file = joinpath(
    SCAN_DIR,
    "results",
    "paper_y_staggered_seed666_fine_long",
    "final.jld2",
)
output_prefix = joinpath(OBS_DIR, "J1_SU2_simple_update_observables")

environment_chi = 32
ctm_tolerance = 1.0e-6
ctm_max_iterations = 150
ctm_cell_method = "continuous_update"

correlation_distance = 20
transfer_n_values_per_spin = 6
transfer_spins = (0, 1 / 2, 1, 3 / 2, 2)
random_seed = 555
save_mat = true

# ---------------------------------------------------------------------------

Random.seed!(random_seed)
isfile(input_state_file) || error("input state does not exist: $input_state_file")
data = load(input_state_file)
A_set = if haskey(data, "T_set")
    data["T_set"]
elseif haskey(data, "A_set")
    data["A_set"]
else
    error("Simple Update analysis expects `T_set` (or `A_set`) in the checkpoint")
end
A_set isa AbstractMatrix || error("the loaded state must be an Lx×Ly matrix")
cell_Lx, cell_Ly = size(A_set)

ctm_setting = scan_ctm_settings(
    tolerance=ctm_tolerance,
    maxiter=ctm_max_iterations,
    verbose=false,
)
scan_prepare_globals(environment_chi, ctm_setting)
global Lx = cell_Lx
global Ly = cell_Ly
global chi = environment_chi
algrithm_CTMRG_settings.CTM_cell_ite_method = ctm_cell_method

println("Square-lattice J1 SU(2) Simple-Update state analysis")
println("parameters:")
println("  input_state_file=$input_state_file")
println("  output_prefix=$output_prefix")
println("  cell=$(cell_Lx)x$(cell_Ly)")
println("  chi=$environment_chi")
println("  CTM_conv_tol=$ctm_tolerance")
println("  CTM_ite_nums=$ctm_max_iterations")
println("  CTM_cell_ite_method=$ctm_cell_method")
println("  correlation_distance=$correlation_distance")
println("  transfer_n_values_per_spin=$transfer_n_values_per_spin")
println("  transfer_spins=$transfer_spins")
println("virtual spaces:")
for cy in 1:cell_Ly, cx in 1:cell_Lx
    println(
        "  site ($cx,$cy): " *
        join(("leg$leg=$(space(A_set[cx, cy], leg))" for leg in 1:4), ", "),
    )
end
flush(stdout)

started = now()
environment = scan_environment(A_set, environment_chi, ctm_setting)
println(
    "CTMRG finished: iterations=$(environment.ite_num), " *
    "error=$(environment.ite_err)",
)
if !ismissing(environment.ite_err) && environment.ite_err > ctm_tolerance
    @warn "CTMRG did not reach the requested tolerance; observables may be unreliable" error=environment.ite_err tolerance=ctm_tolerance
end
flush(stdout)

observables = square_J1_analyze_observables(
    A_set,
    environment;
    distance=correlation_distance,
    transfer_n_values=transfer_n_values_per_spin,
    transfer_spins=transfer_spins,
)

metadata = (
    model="square-lattice spin-1/2 J1 Heisenberg",
    symmetry="SU(2)",
    source="Simple Update iPEPS T_set (lambda is not reabsorbed)",
    input_state_file=abspath(input_state_file),
    cell=(cell_Lx, cell_Ly),
    environment_chi=environment_chi,
    ctm_tolerance=ctm_tolerance,
    ctm_iterations=environment.ite_num,
    ctm_error=environment.ite_err,
    correlation_distance=correlation_distance,
    transfer_n_values_per_spin=transfer_n_values_per_spin,
    transfer_spins=transfer_spins,
    random_seed=random_seed,
)

jld2_file = output_prefix * ".jld2"
jldsave(jld2_file; metadata, observables)

if save_mat
    mat_file = output_prefix * ".mat"
    matwrite(mat_file, Dict(
        "energy_per_site" => observables.local_observables.energy_per_site,
        "rho_one" => observables.local_observables.rho_one,
        "spin_one" => observables.local_observables.spin_one,
        "rho_two_x" => observables.local_observables.rho_two_x,
        "rho_two_y" => observables.local_observables.rho_two_y,
        "spin_spin_local_x" => observables.local_observables.spin_spin_x,
        "spin_spin_local_y" => observables.local_observables.spin_spin_y,
        "spin_spin_components_x" => observables.local_observables.spin_spin_components_x,
        "spin_spin_components_y" => observables.local_observables.spin_spin_components_y,
        "transfer_x_eigenvalues" => observables.transfer_x.eigenvalues,
        "transfer_x_normalized" => observables.transfer_x.normalized_eigenvalues,
        "transfer_x_magnitudes" => observables.transfer_x.magnitudes,
        "transfer_x_spin" => observables.transfer_x.spin,
        "transfer_x_correlation_lengths" => observables.transfer_x.correlation_lengths,
        "transfer_y_eigenvalues" => observables.transfer_y.eigenvalues,
        "transfer_y_normalized" => observables.transfer_y.normalized_eigenvalues,
        "transfer_y_magnitudes" => observables.transfer_y.magnitudes,
        "transfer_y_spin" => observables.transfer_y.spin,
        "transfer_y_correlation_lengths" => observables.transfer_y.correlation_lengths,
        "spin_separations" => observables.correlations.spin_separations,
        "spin_spin_x" => observables.correlations.spin_x,
        "spin_spin_y" => observables.correlations.spin_y,
        "dimer_separations" => observables.correlations.dimer_separations,
        "dimer_dimer_raw_x" => observables.correlations.dimer_raw_x,
        "dimer_dimer_raw_y" => observables.correlations.dimer_raw_y,
        "dimer_dimer_connected_x" => observables.correlations.dimer_connected_x,
        "dimer_dimer_connected_y" => observables.correlations.dimer_connected_y,
    ); compress=true)
    println("Saved MATLAB data to $mat_file")
end

elapsed = Dates.canonicalize(Dates.CompoundPeriod(now() - started))
println("E/site=$(observables.local_observables.energy_per_site)")
println("local <S>=$(observables.local_observables.spin_one)")
println("local <S.S> x=$(observables.local_observables.spin_spin_x[:])")
println("local <S.S> y=$(observables.local_observables.spin_spin_y[:])")
println("transfer |lambda/lambda0| x=$(observables.transfer_x.magnitudes)")
println("transfer spin x=$(observables.transfer_x.spin)")
println("transfer |lambda/lambda0| y=$(observables.transfer_y.magnitudes)")
println("transfer spin y=$(observables.transfer_y.spin)")
println("Saved JLD2 data to $jld2_file")
println("Time consumed: $elapsed")
flush(stdout)
