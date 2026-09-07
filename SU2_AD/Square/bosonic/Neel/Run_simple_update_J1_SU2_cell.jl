"""
Server launcher for the bosonic square-lattice J1 Simple Update.

Edit the configuration block below, then run

    julia Run_simple_update_J1_SU2_cell.jl

No command-line parameters are required.  The state saved by Simple Update
already contains the bond weights in `T_set`; `lambda_x` and `lambda_y` are
kept separately for restarting Simple Update and estimating local energies.
"""

using TensorKit
import TensorKit: ×
using TensorKitSectors
using Zygote
using Zygote: @ignore_derivatives
using LinearAlgebra: I, diag, dot, norm
using KrylovKit
using ChainRulesCore
using JLD2
using Random
using Dates

const RUN_DIR = @__DIR__
const SU2_AD_DIR = normpath(joinpath(RUN_DIR, "..", "..", ".."))
const SIMPLE_RESULTS_DIR = joinpath(RUN_DIR, "simple_update_results")

include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "Settings.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "Settings_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "AD_lib.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "CTMRG.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_model.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "simple_update_lib.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "simple_update_J1_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "full_update_J1.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "full_update_J1_cell.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_initial_states.jl"))
include(joinpath(SU2_AD_DIR, "src", "bosonic", "square", "square_J1_measurements_cell.jl"))

# ---------------------------------------------------------------------------
# Configuration: edit values here; no command-line arguments are used.
# ---------------------------------------------------------------------------

# Used when `initial_state_file === nothing`.
initial_state_kind = :paper_y_staggered
random_seed = 1234
cell_Lx = 2
cell_Ly = 2
n_cpu = 10  # Number of BLAS threads, following the Hofstadter launchers.

# To continue a saved Simple Update state, set this to its initial.jld2 or
# final.jld2.  The file must contain T_set, lambda_x, and lambda_y.
initial_state_file = nothing
# initial_state_file = "simple_update_virtual_space_scan/results/<run>/final.jld2"

# Dmax is the ordinary total virtual dimension.  By default the SVD selects
# the multiplet count freely.  Enable the optional compatibility limit only
# for a deliberate comparison with a fixed maximum number of multiplets.
Dmax = 12
limit_Dstar = false
Dstar_max = 4
multiplet_tolerance = 1.0e-5
convergence_tolerance = -1.0  # Negative disables early stopping.
J1 = 1.0
process_name = "C$(n_cpu)_SU_square_J1_D$(Dmax)"

# This reproduces the legacy-long evolution followed by the fine-long
# continuation used for the best saved states.  Remove stages if a shorter
# run is desired.
schedule = [
    (dt=0.10, tau=30.0),
    (dt=0.05, tau=20.0),
    (dt=0.01, tau=20.0),
    (dt=0.002, tau=4.0),
    (dt=0.002, tau=10.0),
    (dt=0.001, tau=6.0),
    (dt=0.0005, tau=3.0),
    (dt=0.0002, tau=1.0),
]

# Set energy_output_every=1 to print the lambda-environment energy after every
# complete x+y sweep.  A larger interval reduces output and measurement cost.
energy_output_every = 100
# Print all sector-resolved lambda diagonals at this sweep interval.  Set to 0
# to print them only at the initial state and after every schedule stage.
lambda_output_every = 0

# Run CTMRG once on the final T_set.  Lambda is not absorbed a second time.
measure_final_with_ctmrg = true
environment_chi = 60
ctm_tolerance = 1.0e-6
ctm_max_iterations = 150
ctm_cell_method = "continuous_update"

# `nothing` creates a timestamped result directory below simple_update_results.
# Only the named initial/final state files, energy history, and config.txt are
# written; no intermediate tensor states or CTM environment are saved.
output_name = nothing

# ---------------------------------------------------------------------------

import LinearAlgebra.BLAS as BLAS
BLAS.set_num_threads(n_cpu)
println("number of cpus: " * string(BLAS.get_num_threads()))
Base.Sys.set_process_title(process_name)
pid = getpid()
println("pid=" * string(pid))
println("process name=" * process_name)
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm = gethostname()
flush(stdout)

function _simple_run_resolve_state_file(filename)
    isnothing(filename) && return nothing
    return isabspath(filename) ? filename : normpath(joinpath(RUN_DIR, filename))
end

function _simple_run_measure(T_set, lambda_x, lambda_y, coupling)
    return square_J1_simple_energy_cell(
        T_set, lambda_x, lambda_y; J1=coupling,
    )
end

function _simple_run_ctm_measure(T_set, coupling)
    return square_J1_measure_ctm_energy_cell(
        T_set,
        environment_chi;
        J1=coupling,
        tolerance=ctm_tolerance,
        maxiter=ctm_max_iterations,
        verbose=false,
        multiplet_tolerance=multiplet_tolerance,
        cell_method=ctm_cell_method,
    )
end

function _simple_run_write_energy(io, record)
    println(
        io,
        join((
            record.stage,
            record.step,
            record.tau_total,
            record.dt,
            record.error,
            record.energy_per_site,
            record.Ex_mean,
            record.Ey_mean,
        ), ','),
    )
    flush(io)
    return nothing
end

function _simple_run_D_tag(report)
    dimensions = sort!(unique([bond.D for bond in report]))
    return "D" * join(dimensions, "-")
end

function run_simple_update_J1_SU2_cell()
    Random.seed!(random_seed)
    source_file = _simple_run_resolve_state_file(initial_state_file)
    if isnothing(source_file)
        T_set, lambda_x, lambda_y = square_J1_named_initial_state(
            initial_state_kind,
            random_seed,
        )
        source_description = "named:$initial_state_kind"
    else
        isfile(source_file) || error("initial_state_file does not exist: $source_file")
        data = load(source_file)
        missing_keys = filter(
            key -> !haskey(data, key),
            ("T_set", "lambda_x", "lambda_y"),
        )
        isempty(missing_keys) || error(
            "Simple Update restart requires T_set, lambda_x, and lambda_y; " *
            "missing $(join(missing_keys, ", ")) in $source_file",
        )
        T_set = data["T_set"]
        lambda_x = data["lambda_x"]
        lambda_y = data["lambda_y"]
        source_description = abspath(source_file)
    end

    size(T_set) == (cell_Lx, cell_Ly) || error(
        "loaded/constructed cell has size $(size(T_set)), " *
        "but configuration requests $(cell_Lx)x$(cell_Ly)",
    )
    size(lambda_x) == size(T_set) == size(lambda_y) || error(
        "T_set, lambda_x, and lambda_y must have the same cell size",
    )
    Dmax > 0 || error("Dmax must be positive")
    !limit_Dstar || Dstar_max > 0 ||
        error("Dstar_max must be positive when limit_Dstar=true")
    energy_output_every > 0 || error("energy_output_every must be positive")
    lambda_output_every >= 0 || error("lambda_output_every cannot be negative")

    run_name = isnothing(output_name) ?
        "simple_J1_cell$(cell_Lx)x$(cell_Ly)_Dmax$(Dmax)_" *
        Dates.format(now(), "yyyymmdd_HHMMSS") :
        output_name
    output_dir = joinpath(SIMPLE_RESULTS_DIR, run_name)
    mkpath(output_dir)
    file_prefix = "SimpleUpdate_J1_SU2_cell$(cell_Lx)x$(cell_Ly)_seed$(random_seed)"
    energy_file = joinpath(output_dir, "$(file_prefix)_Dmax$(Dmax)_energy_history.csv")

    println("Starting bosonic square-lattice J1 SU(2) Simple Update")
    println("parameters:")
    println("  source=$source_description")
    println("  cell=$(cell_Lx)x$(cell_Ly), seed=$random_seed, J1=$J1")
    println("  Dmax=$Dmax, multiplet_tol=$multiplet_tolerance")
    println("  limit_Dstar=$limit_Dstar" *
            (limit_Dstar ? ", Dstar_max=$Dstar_max" : ""))
    println("  schedule=$schedule")
    println("  energy_output_every=$energy_output_every")
    println("  lambda_output_every=$lambda_output_every")
    println("  final_CTM=$(measure_final_with_ctmrg), chi=$environment_chi")
    println("  output=$output_dir")
    println("initial virtual bonds and lambda diagonals:")
    square_J1_print_bond_spaces(lambda_x, lambda_y; prefix="  ")

    initial_report = square_J1_bond_space_report(lambda_x, lambda_y)
    initial_file = joinpath(
        output_dir,
        "$(file_prefix)_initial_$(_simple_run_D_tag(initial_report)).jld2",
    )
    initial_measurement = _simple_run_measure(T_set, lambda_x, lambda_y, J1)
    jldsave(
        initial_file;
        T_set,
        lambda_x,
        lambda_y,
        initial_state_kind,
        random_seed,
        source=source_description,
        Dmax,
        limit_Dstar,
        Dstar_max,
        initial_report,
        initial_measurement,
    )

    open(joinpath(output_dir, "config.txt"), "w") do io
        println(io, "source=$source_description")
        println(io, "initial_state_kind=$initial_state_kind")
        println(io, "seed=$random_seed")
        println(io, "cell=$(cell_Lx)x$(cell_Ly)")
        println(io, "n_cpu=$n_cpu")
        println(io, "process_name=$process_name")
        println(io, "J1=$J1")
        println(io, "Dmax=$Dmax")
        println(io, "limit_Dstar=$limit_Dstar")
        println(io, "Dstar_max=$(limit_Dstar ? Dstar_max : "unused")")
        println(io, "multiplet_tolerance=$multiplet_tolerance")
        println(io, "convergence_tolerance=$convergence_tolerance")
        println(io, "schedule=$schedule")
        println(io, "energy_output_every=$energy_output_every")
        println(io, "lambda_output_every=$lambda_output_every")
        println(io, "measure_final_with_ctmrg=$measure_final_with_ctmrg")
        println(io, "environment_chi=$environment_chi")
        println(io, "ctm_tolerance=$ctm_tolerance")
        println(io, "ctm_max_iterations=$ctm_max_iterations")
        println(io, "ctm_cell_method=$ctm_cell_method")
    end

    energy_records = NamedTuple[]
    stage_records = NamedTuple[]
    tau_offset = 0.0
    open(energy_file, "w") do energy_io
        println(
            energy_io,
            "stage,step,tau_total,dt,error,energy_per_site,Ex_mean,Ey_mean",
        )
        for (stage, item) in pairs(schedule)
            dt, tau = item.dt, item.tau
            expected_steps = round(Int, tau / dt)
            settings = SquareJ1SimpleUpdateSettings(
                Dmax=Dmax,
                limit_Dstar=limit_Dstar,
                Dstar_max=Dstar_max,
                multiplet_tol=multiplet_tolerance,
                convergence_tol=convergence_tolerance,
                print_every=expected_steps,
                verbose=false,
            )
            last_recorded_step = Ref(0)
            callback = function (T_now, lx_now, ly_now, step, report, error)
                should_measure = step == 1 || step == expected_steps ||
                    step % energy_output_every == 0
                if should_measure
                    measurement = _simple_run_measure(T_now, lx_now, ly_now, J1)
                    record = (
                        stage=stage,
                        step=step,
                        tau_total=tau_offset + step * dt,
                        dt=dt,
                        error=error,
                        energy_per_site=measurement.energy_per_site,
                        Ex_mean=sum(measurement.Ex) / length(measurement.Ex),
                        Ey_mean=sum(measurement.Ey) / length(measurement.Ey),
                    )
                    push!(energy_records, record)
                    _simple_run_write_energy(energy_io, record)
                    last_recorded_step[] = step
                    println(
                        "SU stage=$stage step=$step/$expected_steps " *
                        "tau=$(record.tau_total) error=$error " *
                        "E=$(record.energy_per_site), " *
                        "Ex=$(record.Ex_mean), Ey=$(record.Ey_mean)",
                    )
                    flush(stdout)
                end
                if lambda_output_every > 0 && step % lambda_output_every == 0
                    square_J1_print_bond_spaces(lx_now, ly_now; prefix="  ")
                end
                return nothing
            end

            started = now()
            T_set, lambda_x, lambda_y, history = square_J1_simple_update_cell(
                T_set,
                lambda_x,
                lambda_y,
                tau,
                dt;
                J1=J1,
                settings=settings,
                callback=callback,
            )
            completed_steps = length(history)
            final_error = isempty(history) ? NaN : history[end].error
            if completed_steps > 0 && last_recorded_step[] != completed_steps
                measurement = _simple_run_measure(T_set, lambda_x, lambda_y, J1)
                record = (
                    stage=stage,
                    step=completed_steps,
                    tau_total=tau_offset + completed_steps * dt,
                    dt=dt,
                    error=final_error,
                    energy_per_site=measurement.energy_per_site,
                    Ex_mean=sum(measurement.Ex) / length(measurement.Ex),
                    Ey_mean=sum(measurement.Ey) / length(measurement.Ey),
                )
                push!(energy_records, record)
                _simple_run_write_energy(energy_io, record)
                println(
                    "SU stage=$stage step=$completed_steps/$expected_steps " *
                    "tau=$(record.tau_total) error=$final_error " *
                    "E=$(record.energy_per_site), " *
                    "Ex=$(record.Ex_mean), Ey=$(record.Ey_mean)",
                )
            end
            elapsed_seconds = Dates.value(now() - started) / 1000
            final_report = square_J1_bond_space_report(lambda_x, lambda_y)
            push!(stage_records, (
                stage=stage,
                dt=dt,
                requested_tau=tau,
                completed_steps=completed_steps,
                final_error=final_error,
                elapsed_seconds=elapsed_seconds,
                bond_spaces=final_report,
            ))
            println("stage $stage final virtual bonds and lambda diagonals:")
            square_J1_print_bond_spaces(lambda_x, lambda_y; prefix="  ")
            tau_offset += completed_steps * dt
        end
    end

    simple_measurement = _simple_run_measure(T_set, lambda_x, lambda_y, J1)
    final_report = square_J1_bond_space_report(lambda_x, lambda_y)
    final_file = joinpath(
        output_dir,
        "$(file_prefix)_final_$(_simple_run_D_tag(final_report)).jld2",
    )

    # Save the optimized state before CTMRG so a CTM failure cannot lose the
    # Simple Update result.
    jldsave(
        final_file;
        T_set,
        lambda_x,
        lambda_y,
        initial_state_kind,
        random_seed,
        source=source_description,
        Dmax,
        limit_Dstar,
        Dstar_max,
        schedule,
        stage_records,
        energy_records,
        final_report,
        simple_measurement,
        ctm_status="not_run",
    )

    ctm_status = "disabled"
    ctm_measurement = nothing
    if measure_final_with_ctmrg
        println("Running final CTMRG energy measurement...")
        flush(stdout)
        try
            ctm_measurement = _simple_run_ctm_measure(T_set, J1)
            ctm_status = "ok"
            println(
                "CTM E/site=$(ctm_measurement.energy_per_site), " *
                "chi=$environment_chi, iterations=$(ctm_measurement.ctm_iterations), " *
                "error=$(ctm_measurement.ctm_error)",
            )
        catch exception
            ctm_status = "failed: " * sprint(showerror, exception)
            @warn "Final CTMRG measurement failed; Simple Update state remains saved" exception
        end
    end

    jldsave(
        final_file;
        T_set,
        lambda_x,
        lambda_y,
        initial_state_kind,
        random_seed,
        source=source_description,
        Dmax,
        limit_Dstar,
        Dstar_max,
        schedule,
        stage_records,
        energy_records,
        final_report,
        simple_measurement,
        ctm_status,
        ctm_measurement,
    )
    println("Final SU E/site=$(simple_measurement.energy_per_site)")
    println("Final state saved to $final_file")
    println("Energy history saved to $energy_file")
    flush(stdout)
    return T_set, lambda_x, lambda_y
end

run_simple_update_J1_SU2_cell()
