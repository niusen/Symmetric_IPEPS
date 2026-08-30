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
include(joinpath(SCAN_DIR, "..", "square_J1_initial_states.jl"))

const PAPER_ENERGY_SU2_DSTAR4 = -0.6686
const QMC_ENERGY = -0.6694

function scan_initial_state(kind::Symbol, seed::Int)
    return square_J1_named_initial_state(kind, seed)
end

function scan_ctm_settings(; tolerance=1.0e-6, maxiter=120, verbose=false)
    settings = LS_CTMRG_settings()
    settings.CTM_conv_tol = tolerance
    settings.CTM_ite_nums = maxiter
    settings.CTM_trun_tol = 1.0e-8
    settings.svd_lanczos_tol = 1.0e-8
    settings.projector_strategy = "4x4"
    settings.conv_check = "singular_value"
    settings.CTM_ite_info = verbose
    settings.CTM_conv_info = true
    settings.CTM_trun_svd = false
    settings.construct_double_layer = true
    settings.grad_checkpoint = false
    return settings
end

function scan_prepare_globals(chi_value::Int, ctm_settings)
    global Lx = 2
    global Ly = 2
    global chi = chi_value
    global multiplet_tol = 1.0e-5
    global projector_trun_tol = ctm_settings.CTM_trun_tol
    global backward_settings = Backward_settings()
    global algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
    algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"
    return nothing
end

function scan_environment(T_set, chi_value, ctm_settings)
    global Lx = size(T_set, 1)
    global Ly = size(T_set, 2)
    global chi = chi_value
    A_cell = square_fu_cell_to_tuple(T_set)
    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    result = CTMRG_cell(A_cell, chi_value, init, [], ctm_settings)
    if length(result) == 8
        CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err = result
    else
        CTM, AA, U_L, U_D, U_R, U_U = result
        ite_num, ite_err = missing, missing
    end
    return (
        CTM=CTM,
        AA=AA,
        U_L=U_L,
        U_D=U_D,
        U_R=U_R,
        U_U=U_U,
        ite_num=ite_num,
        ite_err=ite_err,
    )
end

function scan_J1_energy(T_set, environment)
    cell_Lx, cell_Ly = size(T_set)
    H_Heisenberg, _, _, _, _ = Hamiltonians(space(T_set[1, 1], 1))
    H = permute(H_Heisenberg, (1, 2), (3, 4))
    Ex = zeros(Float64, cell_Lx, cell_Ly)
    Ey = zeros(Float64, cell_Lx, cell_Ly)
    for cx in 1:cell_Lx, cy in 1:cell_Ly
        for direction in (:x, :y)
            site1 = CartesianIndex(cx, cy)
            site2 = direction === :x ?
                CartesianIndex(mod1(cx + 1, cell_Lx), cy) :
                CartesianIndex(cx, mod1(cy + 1, cell_Ly))
            bond = SquareJ1CellBond(direction, site1, site2)
            A1, A2 = T_set[site1], T_set[site2]
            rho = _square_fu_two_site_density_cell(
                environment.CTM,
                A1,
                A1,
                A2,
                A2,
                bond,
                cell_Lx,
                cell_Ly,
            )
            norm_rho = real(@tensor rho[1, 2, 1, 2])
            norm_rho != 0 || error("zero norm on $direction bond ($cx,$cy)")
            value = real(@tensor rho[1, 2, 3, 4] * H[1, 2, 3, 4]) / norm_rho
            direction === :x ? (Ex[cx, cy] = value) : (Ey[cx, cy] = value)
        end
    end
    return (energy_per_site=(sum(Ex) + sum(Ey)) / (cell_Lx * cell_Ly), Ex=Ex, Ey=Ey)
end

function scan_energy(T_set, chi_value::Int; tolerance=1.0e-6, maxiter=120, verbose=false)
    ctm_settings = scan_ctm_settings(; tolerance, maxiter, verbose)
    scan_prepare_globals(chi_value, ctm_settings)
    environment = scan_environment(T_set, chi_value, ctm_settings)
    energy = scan_J1_energy(T_set, environment)
    return merge(energy, (
        chi=chi_value,
        ctm_iterations=environment.ite_num,
        ctm_error=environment.ite_err,
    ))
end

function scan_space_key(report)
    return join(
        ("$(bond.direction)$(bond.from):$(bond.space)" for bond in report),
        " | ",
    )
end

function scan_simple_x_energy(Tset, lambda_x, lambda_y, cx, cy, H)
    cell_Lx, cell_Ly = size(Tset)
    site1 = (cx, cy)
    site2 = (mod1(cx + 1, cell_Lx), cy)
    lambda1 = lambda_x[site1...]
    lambda2 = lambda_y[site1...]
    lambda3 = lambda_y[site1[1], mod1(site1[2] + 1, cell_Ly)]
    lambda4 = lambda_y[site2...]
    lambda5 = lambda_x[mod1(site2[1] + 1, cell_Lx), site2[2]]
    lambda6 = lambda_y[site2[1], mod1(site2[2] + 1, cell_Ly)]
    T1, T2 = Tset[site1...], Tset[site2...]
    @tensor T1_env[:] := T1[1,2,-3,3,-5] * lambda1[-1,1] * lambda2[2,-2] * lambda3[-4,3]
    @tensor T2_env[:] := T2[-1,1,2,3,-5] * lambda4[1,-2] * lambda5[2,-3] * lambda6[-4,3]
    u1, s1, v1 = tsvd(permute(T1_env, (1,2,4), (3,5)))
    keep1 = s1 * v1
    u2, s2, v2 = tsvd(permute(T2_env, (1,5), (2,3,4)))
    keep2 = u2 * s2
    @tensor psi[:] := keep1[-1,1,-3] * keep2[1,-2,-4]
    # Negative labels above order the legs as (env1, physical2, physical1, env2).
    psi = permute(psi, (1,4), (3,2))
    physical_isometry = unitary(domain(psi), codomain(H))
    H_psi = physical_isometry * H * physical_isometry'
    return real(dot(psi, psi * H_psi) / dot(psi, psi))
end

function scan_simple_y_energy(Tset, lambda_x, lambda_y, cx, cy, H)
    cell_Lx, cell_Ly = size(Tset)
    upper = (cx, mod1(cy + 1, cell_Ly))
    lower = (cx, cy)
    lambda1 = lambda_x[upper...]
    lambda2 = lambda_x[mod1(upper[1] + 1, cell_Lx), upper[2]]
    lambda3 = lambda_y[upper[1], mod1(upper[2] + 1, cell_Ly)]
    lambda4 = lambda_x[lower...]
    lambda5 = lambda_y[lower...]
    lambda6 = lambda_x[mod1(lower[1] + 1, cell_Lx), lower[2]]
    T1, T2 = Tset[upper...], Tset[lower...]
    @tensor T1_env[:] := T1[1,-2,2,3,-5] * lambda1[-1,1] * lambda2[2,-3] * lambda3[-4,3]
    @tensor T2_env[:] := T2[1,2,3,-4,-5] * lambda4[-1,1] * lambda5[2,-2] * lambda6[3,-3]
    u1, s1, v1 = tsvd(permute(T1_env, (1,3,4), (2,5)))
    keep1 = s1 * v1
    u2, s2, v2 = tsvd(permute(T2_env, (4,5), (1,2,3)))
    keep2 = u2 * s2
    @tensor psi[:] := keep1[-1,1,-3] * keep2[1,-2,-4]
    psi = permute(psi, (1,4), (3,2))
    physical_isometry = unitary(domain(psi), codomain(H))
    H_psi = physical_isometry * H * physical_isometry'
    return real(dot(psi, psi * H_psi) / dot(psi, psi))
end

function scan_simple_energy(Tset, lambda_x, lambda_y)
    H_Heisenberg, _, _, _, _ = Hamiltonians(space(Tset[1,1], 1))
    H = permute(H_Heisenberg, (1,2), (3,4))
    Ex = [scan_simple_x_energy(Tset, lambda_x, lambda_y, cx, cy, H)
          for cx in 1:2, cy in 1:2]
    Ey = [scan_simple_y_energy(Tset, lambda_x, lambda_y, cx, cy, H)
          for cx in 1:2, cy in 1:2]
    return (energy_per_site=(sum(Ex) + sum(Ey)) / 4, Ex=Ex, Ey=Ey)
end

function scan_run_schedule!(
    T_set,
    lambda_x,
    lambda_y,
    schedule,
    case_dir;
    Dstar=4,
    Dmax=12,
    multiplet_tolerance=1.0e-5,
)
    stage_records = NamedTuple[]
    for (stage, item) in pairs(schedule)
        dt, tau = item.dt, item.tau
        settings = SquareJ1SimpleUpdateSettings(
            Dstar=Dstar,
            Dmax=Dmax,
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
