using TensorKit
import TensorKit: ×
using Zygote
using Zygote: @ignore_derivatives
using LinearAlgebra: I, diag, dot, norm
using KrylovKit
using ChainRulesCore
using JLD2
using Random
using Dates

const SCAN_DIR = @__DIR__
const SU2_AD_DIR = normpath(joinpath(SCAN_DIR, "..", "..", "..", ".."))

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

const PAPER_ENERGY_SU2_DSTAR4 = -0.6686
const QMC_ENERGY = -0.6694

function scan_initial_state(kind::Symbol, seed::Int)
    return square_J1_named_initial_state(kind, seed)
end

scan_ctm_settings(; kwargs...) = square_J1_default_ctm_settings(; kwargs...)

function scan_environment(T_set, chi_value, ctm_settings)
    return square_J1_environment_cell(T_set, chi_value, ctm_settings)
end

scan_J1_energy(T_set, environment) = square_J1_energy_cell(T_set, environment)

function scan_energy(T_set, chi_value::Int; tolerance=1.0e-6, maxiter=120, verbose=false)
    return square_J1_measure_ctm_energy_cell(
        T_set,
        chi_value;
        tolerance,
        maxiter,
        verbose,
    )
end

function scan_space_key(report)
    return join(
        ("$(bond.direction)$(bond.from):$(bond.space)" for bond in report),
        " | ",
    )
end

scan_simple_energy(T_set, lambda_x, lambda_y) =
    square_J1_simple_energy_cell(T_set, lambda_x, lambda_y)

function scan_run_schedule!(
    T_set,
    lambda_x,
    lambda_y,
    schedule,
    case_dir;
    Dmax=12,
    limit_Dstar=false,
    Dstar_max=4,
    multiplet_tolerance=1.0e-5,
)
    stage_records = NamedTuple[]
    for (stage, item) in pairs(schedule)
        dt, tau = item.dt, item.tau
        settings = SquareJ1SimpleUpdateSettings(
            Dmax=Dmax,
            limit_Dstar=limit_Dstar,
            Dstar_max=Dstar_max,
            multiplet_tol=multiplet_tolerance,
            convergence_tol=-1.0,
            print_every=max(1, Int(round(tau / dt))),
            verbose=false,
        )
        started = now()
        T_set, lambda_x, lambda_y, history = square_J1_simple_update_cell(
            T_set, lambda_x, lambda_y, tau, dt; settings,
        )
        report = square_J1_bond_space_report(lambda_x, lambda_y)
        elapsed = Dates.value(now() - started) / 1000
        record = (
            stage=stage,
            dt=dt,
            tau=tau,
            elapsed_seconds=elapsed,
            final_error=isempty(history) ? NaN : history[end].error,
            bond_spaces=report,
        )
        push!(stage_records, record)
        if parse(Bool, get(ENV, "SCAN_SAVE_STAGES", "false"))
            jldsave(
                joinpath(case_dir, "stage_$(stage)_dt_$(dt).jld2");
                T_set,
                lambda_x,
                lambda_y,
                history,
                record,
            )
        end
        println(
            "  stage=$stage dt=$dt tau=$tau error=$(record.final_error) " *
            "elapsed=$(round(elapsed; digits=2))s",
        )
        if parse(Bool, get(ENV, "SCAN_PRINT_BONDS", "true"))
            square_J1_print_bond_spaces(lambda_x, lambda_y; prefix="    ")
        end
    end
    return T_set, lambda_x, lambda_y, stage_records
end

function scan_append_csv(filename, row)
    new_file = !isfile(filename)
    open(filename, "a") do io
        if new_file
            println(io, "case,init,seed,schedule,su_energy,energy,chi,ctm_error,ctm_iterations,space_key,status,message")
        end
        clean(value) = replace(string(value), '"' => "''", '\n' => ' ')
        values = (
            row.case,
            row.init,
            row.seed,
            row.schedule,
            row.su_energy,
            row.energy,
            row.chi,
            row.ctm_error,
            row.ctm_iterations,
            row.space_key,
            row.status,
            row.message,
        )
        println(io, join(("\"$(clean(value))\"" for value in values), ','))
    end
end

function scan_run_case(
    run_dir,
    init_kind::Symbol,
    seed::Int,
    schedule_name::AbstractString,
    schedule;
    chi=16,
    ctm_tolerance=1.0e-5,
    ctm_maxiter=80,
    measure_ctm=true,
)
    case_name = "$(init_kind)_seed_$(seed)_$(schedule_name)"
    case_dir = joinpath(run_dir, case_name)
    mkpath(case_dir)
    println("\n=== $case_name ===")
    try
        T_set, lambda_x, lambda_y = scan_initial_state(init_kind, seed)
        initial_report = square_J1_bond_space_report(lambda_x, lambda_y)
        jldsave(
            joinpath(case_dir, "initial.jld2");
            T_set,
            lambda_x,
            lambda_y,
            init_kind,
            seed,
            initial_report,
        )
        T_set, lambda_x, lambda_y, stage_records = scan_run_schedule!(
            T_set, lambda_x, lambda_y, schedule, case_dir,
        )
        simple_measurement = scan_simple_energy(T_set, lambda_x, lambda_y)
        measurement = measure_ctm ? scan_energy(
            T_set, chi; tolerance=ctm_tolerance, maxiter=ctm_maxiter, verbose=false,
        ) : (
            energy_per_site=NaN,
            Ex=fill(NaN, 2, 2),
            Ey=fill(NaN, 2, 2),
            chi=chi,
            ctm_iterations=-1,
            ctm_error=NaN,
        )
        final_report = square_J1_bond_space_report(lambda_x, lambda_y)
        jldsave(
            joinpath(case_dir, "final.jld2");
            T_set,
            lambda_x,
            lambda_y,
            init_kind,
            seed,
            schedule_name,
            schedule,
            stage_records,
            final_report,
            measurement,
            simple_measurement,
        )
        println(
            "  SU E/site=$(simple_measurement.energy_per_site), " *
            "CTM E/site=$(measurement.energy_per_site), chi=$chi, " *
            "CTM error=$(measurement.ctm_error)",
        )
        return (
            case=case_name,
            init=init_kind,
            seed=seed,
            schedule=schedule_name,
            su_energy=simple_measurement.energy_per_site,
            energy=measurement.energy_per_site,
            chi=chi,
            ctm_error=measurement.ctm_error,
            ctm_iterations=measurement.ctm_iterations,
            space_key=scan_space_key(final_report),
            status="ok",
            message="",
        )
    catch exception
        message = sprint(showerror, exception, catch_backtrace())
        open(joinpath(case_dir, "error.log"), "w") do io
            write(io, message)
        end
        println("  FAILED: ", sprint(showerror, exception))
        return (
            case=case_name,
            init=init_kind,
            seed=seed,
            schedule=schedule_name,
            su_energy=NaN,
            energy=NaN,
            chi=chi,
            ctm_error=NaN,
            ctm_iterations=-1,
            space_key="",
            status="failed",
            message=message,
        )
    finally
        GC.gc()
    end
end
