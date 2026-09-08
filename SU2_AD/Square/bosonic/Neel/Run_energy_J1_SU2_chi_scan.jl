"""
CTMRG energy scan for a saved square-lattice SU(2) J1 iPEPS.

Edit the configuration block below, then run this file directly.  Checkpoints
from Simple Update (`T_set`) and Full Update (`A_set`) are both accepted.
The chi values must be strictly increasing.  When `reuse_previous_ctm=true`,
the converged CTM at one chi is used to initialize the next chi.
"""

using TensorKit
import TensorKit: ×
using TensorKitSectors
using Zygote
using Zygote: @ignore_derivatives
using LinearAlgebra: I, diag, diagm, dot, norm
using KrylovKit
using ChainRulesCore
using JLD2
using MAT
using Dates
using Random

const RUN_DIR = @__DIR__
const NEEL_DIR = RUN_DIR
const SU2_AD_DIR = normpath(joinpath(NEEL_DIR, "..", "..", ".."))
const RESULT_ROOT = joinpath(NEEL_DIR, "energy_observables_results")

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
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_measurements_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_correl.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_observables_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_transfer_spectrum_cell.jl"))

# ---------------------------------------------------------------------------
# Configuration: edit values here; no command-line arguments are used.
# ---------------------------------------------------------------------------

# Put the state file in the same folder as this script and enter its name here.
filenm = "final.jld2"
chi_list = [16, 24, 32, 48, 64]
reuse_previous_ctm = true
J1 = 1.0

ctm_tolerance = 1.0e-6
ctm_max_iterations = 150
ctm_cell_method = "continuous_update"
multiplet_tolerance = 1.0e-5

# Transfer-matrix spectrum: number of eigenvalues requested in each SU(2)
# spin sector.  Correlation lengths are reported separately for x and y.
transfer_n_values_per_spin = 6
transfer_spins = (0, 1 / 2, 1, 3 / 2, 2)
transfer_cut = 1
transfer_random_seed = 555

n_cpu = 10
process_name = "C$(n_cpu)_E_square_J1_chi_scan"

# `nothing` generates a name containing cell size, actual state D, chi range,
# and a timestamp.  CTMs are reused in memory but are not saved by default.
output_name = nothing
save_last_ctm = false

# ---------------------------------------------------------------------------

import LinearAlgebra.BLAS as BLAS
BLAS.set_num_threads(n_cpu)
Base.Sys.set_process_title(process_name)
println("number of cpus: $(BLAS.get_num_threads())")
pid = getpid()
println("pid=" * string(pid))
println("process name=$process_name")
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
flush(stdout)

function _energy_scan_resolve_file(filename)
    return isabspath(filename) ? filename : normpath(joinpath(RUN_DIR, filename))
end

function _energy_scan_load_state(filename)
    data = load(filename)
    if haskey(data, "T_set")
        return data["T_set"], "Simple Update T_set"
    elseif haskey(data, "A_set")
        return data["A_set"], "Full Update A_set"
    end
    error("checkpoint must contain either T_set (Simple Update) or A_set (Full Update)")
end

function _energy_scan_D_tag(A_set)
    dimensions = sort!(unique([
        dim(space(A_set[cx, cy], leg))
        for cx in axes(A_set, 1), cy in axes(A_set, 2), leg in 1:4
    ]))
    return "D" * join(dimensions, "-")
end

function _energy_scan_write_csv_record(io, record)
    println(
        io,
        join((
            record.chi,
            record.energy_per_site,
            record.Ex_mean,
            record.Ey_mean,
            record.ctm_iterations,
            record.ctm_error,
            record.reused_initial_CTM,
            record.xi_x,
            record.inverse_xi_x,
            record.xi_y,
            record.inverse_xi_y,
            record.transfer_subleading_spin_x,
            record.transfer_subleading_spin_y,
            record.elapsed_seconds,
        ), ','),
    )
    flush(io)
    return nothing
end

function _energy_scan_write_mat(filename, record)
    matwrite(filename, Dict(
        "chi" => record.chi,
        "energy_per_site" => record.energy_per_site,
        "Ex" => record.Ex,
        "Ey" => record.Ey,
        "Ex_mean" => record.Ex_mean,
        "Ey_mean" => record.Ey_mean,
        "ctm_iterations" => record.ctm_iterations,
        "ctm_error" => record.ctm_error,
        "xi_x" => record.xi_x,
        "inverse_xi_x" => record.inverse_xi_x,
        "xi_y" => record.xi_y,
        "inverse_xi_y" => record.inverse_xi_y,
        "transfer_x_eigenvalues" => record.transfer_x.eigenvalues,
        "transfer_x_normalized" => record.transfer_x.normalized_eigenvalues,
        "transfer_x_magnitudes" => record.transfer_x.magnitudes,
        "transfer_x_spin" => record.transfer_x.spin,
        "transfer_x_correlation_lengths" => record.transfer_x.correlation_lengths,
        "transfer_y_eigenvalues" => record.transfer_y.eigenvalues,
        "transfer_y_normalized" => record.transfer_y.normalized_eigenvalues,
        "transfer_y_magnitudes" => record.transfer_y.magnitudes,
        "transfer_y_spin" => record.transfer_y.spin,
        "transfer_y_correlation_lengths" => record.transfer_y.correlation_lengths,
    ); compress=true)
    return nothing
end

function run_energy_J1_SU2_chi_scan()
    state_file = _energy_scan_resolve_file(filenm)
    isfile(state_file) || error("state file does not exist: $state_file")
    A_set, state_kind = _energy_scan_load_state(state_file)
    A_set isa AbstractMatrix || error("loaded iPEPS must be an Lx×Ly matrix")
    cell_Lx, cell_Ly = size(A_set)

    isempty(chi_list) && error("chi_list cannot be empty")
    all(chi_value -> chi_value > 0, chi_list) || error("all chi values must be positive")
    issorted(chi_list) || error("chi_list must be ordered from small to large")
    allunique(chi_list) || error("chi_list must not contain duplicates")

    D_tag = _energy_scan_D_tag(A_set)
    chi_tag = "chi$(first(chi_list))-$(last(chi_list))"
    default_name = "Energy_J1_SU2_cell$(cell_Lx)x$(cell_Ly)_$(D_tag)_$(chi_tag)_" *
        Dates.format(now(), "yyyymmdd_HHMMSS")
    run_name = isnothing(output_name) ? default_name : output_name
    output_dir = joinpath(RESULT_ROOT, run_name)
    mkpath(output_dir)
    file_prefix = "Energy_J1_SU2_cell$(cell_Lx)x$(cell_Ly)_$(D_tag)_$(chi_tag)"
    csv_file = joinpath(output_dir, file_prefix * ".csv")
    result_file = joinpath(output_dir, file_prefix * ".jld2")
    ctm_file = joinpath(output_dir, file_prefix * "_last_CTM.jld2")

    println("Square-lattice J1 SU(2) CTMRG energy scan")
    println("parameters:")
    println("  filenm=$filenm")
    println("  state_kind=$state_kind")
    println("  cell=$(cell_Lx)x$(cell_Ly), $D_tag, J1=$J1")
    println("  chi_list=$chi_list")
    println("  reuse_previous_ctm=$reuse_previous_ctm")
    println("  CTM_conv_tol=$ctm_tolerance, CTM_ite_nums=$ctm_max_iterations")
    println("  CTM_cell_ite_method=$ctm_cell_method")
    println("  multiplet_tol=$multiplet_tolerance")
    println("  transfer_n_values_per_spin=$transfer_n_values_per_spin")
    println("  transfer_spins=$transfer_spins")
    println("  transfer_cut=$transfer_cut")
    println("  transfer_random_seed=$transfer_random_seed")
    println("  output=$output_dir")
    println("virtual spaces:")
    for cy in axes(A_set, 2), cx in axes(A_set, 1)
        println(
            "  site ($cx,$cy): " *
            join(("leg$leg=$(space(A_set[cx, cy], leg))" for leg in 1:4), ", "),
        )
    end
    flush(stdout)

    metadata = (
        model="square-lattice spin-1/2 J1 Heisenberg",
        symmetry="SU(2)",
        input_state_file=abspath(state_file),
        state_kind,
        cell=(cell_Lx, cell_Ly),
        D_tag,
        chi_list=collect(chi_list),
        reuse_previous_ctm,
        J1,
        ctm_tolerance,
        ctm_max_iterations,
        ctm_cell_method,
        multiplet_tolerance,
        transfer_n_values_per_spin,
        transfer_spins,
        transfer_cut,
        transfer_random_seed,
    )
    open(joinpath(output_dir, "config.txt"), "w") do io
        for name in propertynames(metadata)
            println(io, "$name=$(getproperty(metadata, name))")
        end
        println(io, "save_last_ctm=$save_last_ctm")
    end

    records = NamedTuple[]
    previous_CTM = nothing
    open(csv_file, "w") do io
        println(
            io,
            "chi,energy_per_site,Ex_mean,Ey_mean,ctm_iterations,ctm_error," *
            "reused_initial_CTM,xi_x,inverse_xi_x,xi_y,inverse_xi_y," *
            "transfer_subleading_spin_x,transfer_subleading_spin_y,elapsed_seconds",
        )
        for chi_value in chi_list
            initial_CTM = reuse_previous_ctm ? previous_CTM : nothing
            started = time()
            println("Starting chi=$chi_value, reuse_CTM=$(!isnothing(initial_CTM))")
            flush(stdout)
            ctm_settings = square_J1_default_ctm_settings(
                tolerance=ctm_tolerance,
                maxiter=ctm_max_iterations,
                verbose=false,
            )
            environment = square_J1_environment_cell(
                A_set,
                chi_value,
                ctm_settings;
                multiplet_tolerance,
                cell_method=ctm_cell_method,
                initial_CTM,
            )

            current_CTM = environment.CTM
            ctm_iterations = environment.ite_num
            ctm_error = environment.ite_err
            # Discard CTMRG fusion auxiliaries but retain its already-built
            # closed double layers for the established T-AA-T transfer kernel.
            transfer_environment = (CTM=current_CTM, AA=environment.AA)
            environment = nothing
            GC.gc()
            Random.seed!(transfer_random_seed + 2chi_value)
            transfer_x = square_J1_transfer_spectrum_three_line_cell(
                A_set,
                transfer_environment,
                :x;
                n_values=transfer_n_values_per_spin,
                spins=transfer_spins,
                cut=transfer_cut,
            )
            transfer_x_summary = square_J1_transfer_correlation_length(transfer_x)
            GC.gc()
            Random.seed!(transfer_random_seed + 2chi_value + 1)
            transfer_y = square_J1_transfer_spectrum_three_line_cell(
                A_set,
                transfer_environment,
                :y;
                n_values=transfer_n_values_per_spin,
                spins=transfer_spins,
                cut=transfer_cut,
            )
            transfer_y_summary = square_J1_transfer_correlation_length(transfer_y)
            transfer_environment = nothing
            GC.gc()

            energy = square_J1_energy_cell(
                A_set,
                (CTM=current_CTM,);
                J1,
                low_memory=true,
            )
            measurement = merge(energy, (
                chi=chi_value,
                ctm_iterations,
                ctm_error,
                reused_initial_CTM=!isnothing(initial_CTM),
            ))
            record = merge(measurement, (
                Ex_mean=sum(measurement.Ex) / length(measurement.Ex),
                Ey_mean=sum(measurement.Ey) / length(measurement.Ey),
                xi_x=transfer_x_summary.xi,
                inverse_xi_x=transfer_x_summary.inverse_xi,
                xi_y=transfer_y_summary.xi,
                inverse_xi_y=transfer_y_summary.inverse_xi,
                transfer_subleading_spin_x=transfer_x_summary.spin,
                transfer_subleading_spin_y=transfer_y_summary.spin,
                transfer_x,
                transfer_y,
                elapsed_seconds=time() - started,
            ))
            push!(records, record)
            _energy_scan_write_csv_record(io, record)
            jldsave(result_file; metadata, records)
            mat_file = joinpath(
                output_dir,
                "$(file_prefix)_chi$(chi_value)_transfer_spectrum.mat",
            )
            _energy_scan_write_mat(mat_file, record)
            println(
                "chi=$(record.chi), E/site=$(record.energy_per_site), " *
                "Ex=$(record.Ex_mean), Ey=$(record.Ey_mean), " *
                "xi_x=$(record.xi_x), xi_y=$(record.xi_y), " *
                "CTM iterations=$(record.ctm_iterations), error=$(record.ctm_error), " *
                "elapsed=$(round(record.elapsed_seconds; digits=2))s",
            )
            println("  ex_set=$(record.Ex[:])")
            println("  ey_set=$(record.Ey[:])")
            println("  transfer_x |lambda/lambda0|=$(record.transfer_x.magnitudes)")
            println("  transfer_x spin=$(record.transfer_x.spin)")
            println("  transfer_y |lambda/lambda0|=$(record.transfer_y.magnitudes)")
            println("  transfer_y spin=$(record.transfer_y.spin)")
            println("  transfer spectrum MAT=$mat_file")
            isempty(record.transfer_x.failed_sectors) ||
                println("  transfer_x failed sectors=$(record.transfer_x.failed_sectors)")
            isempty(record.transfer_y.failed_sectors) ||
                println("  transfer_y failed sectors=$(record.transfer_y.failed_sectors)")
            if !ismissing(record.ctm_error) && record.ctm_error > ctm_tolerance
                @warn "CTMRG did not reach the requested tolerance" chi=chi_value error=record.ctm_error tolerance=ctm_tolerance
            end
            flush(stdout)

            previous_CTM = reuse_previous_ctm ? current_CTM : nothing
            current_CTM = nothing
            initial_CTM = nothing
            transfer_environment = nothing
            transfer_x_summary = nothing
            transfer_y_summary = nothing
            GC.gc()
        end
    end

    if save_last_ctm && reuse_previous_ctm && !isnothing(previous_CTM)
        last_chi = last(chi_list)
        jldsave(ctm_file; CTM=previous_CTM, last_chi, metadata)
        println("Last CTM saved to $ctm_file")
    end
    previous_CTM = nothing
    GC.gc()

    println("Energy results saved to $result_file")
    println("CSV saved to $csv_file")
    flush(stdout)
    return records
end

run_energy_J1_SU2_chi_scan()
