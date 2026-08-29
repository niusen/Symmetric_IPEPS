using TensorKit
import TensorKit: ×
using LinearAlgebra: I, diag, norm
using ChainRulesCore
using JLD2
using Random
using Dates

cd(@__DIR__)

include("../../../src/bosonic/Settings.jl")
include("../../../src/bosonic/AD_lib.jl")
include("../../../src/bosonic/square/square_spin_operator.jl")
include("../../../src/bosonic/square/simple_update_lib.jl")
include("../../../src/bosonic/square/simple_update_J1_cell.jl")

Random.seed!(parse(Int, get(ENV, "SU_SEED", "666")))

cell_Lx = parse(Int, get(ENV, "SU_LX", "2"))
cell_Ly = parse(Int, get(ENV, "SU_LY", "2"))
Dmax = parse(Int, get(ENV, "SU_DMAX", "12"))
Dstar = parse(Int, get(ENV, "SU_DSTAR", "4"))
tau = parse(Float64, get(ENV, "SU_TAU", "0.1"))
dt = parse(Float64, get(ENV, "SU_DT", "0.01"))
J1 = parse(Float64, get(ENV, "SU_J1", "1.0"))
multiplet_tol_su = parse(Float64, get(ENV, "SU_MULTIPLET_TOL", "1e-5"))
print_every = parse(Int, get(ENV, "SU_PRINT_EVERY", "1"))
save_every = parse(Int, get(ENV, "SU_SAVE_EVERY", "1"))
save_every > 0 || error("SU_SAVE_EVERY must be positive")
init_filename = get(ENV, "SU_INIT", "nothing")
save_filename = get(
    ENV,
    "SU_SAVE",
    "SimpleUpdate_J1_SU2_dynamic_$(cell_Lx)x$(cell_Ly)_Dmax_$(Dmax).jld2",
)

Vp = SU2Space(1 / 2 => 1)
Vmix = SU2Space(0 => 1, 1 / 2 => 1)

if init_filename == "nothing"
    T_set, lambda_x, lambda_y = square_J1_initial_mixed_cell(
        cell_Lx, cell_Ly; Vp=Vp, Vv=Vmix,
    )
else
    data = load(init_filename)
    all(haskey(data, name) for name in ("T_set", "lambda_x", "lambda_y")) ||
        error("SU_INIT must contain T_set, lambda_x and lambda_y")
    T_set = data["T_set"]
    lambda_x = data["lambda_x"]
    lambda_y = data["lambda_y"]
    size(T_set) == (cell_Lx, cell_Ly) || error(
        "SU_INIT cell size $(size(T_set)) does not match SU_LX×SU_LY=" *
        "$cell_Lx×$cell_Ly",
    )
end

settings = SquareJ1SimpleUpdateSettings(
    Dstar=Dstar,
    Dmax=Dmax,
    multiplet_tol=multiplet_tol_su,
    convergence_tol=parse(Float64, get(ENV, "SU_CONV_TOL", "0.0")),
    print_every=print_every,
    verbose=parse(Bool, get(ENV, "SU_VERBOSE", "true")),
)

println("Starting square-lattice bosonic J1 SU(2) Simple Update")
println(
    "cell=$cell_Lx×$cell_Ly, initial Vmix=$Vmix, Dstar=$Dstar, Dmax=$Dmax, " *
    "tau=$tau, dt=$dt, J1=$J1, multiplet_tol=$multiplet_tol_su",
)
println("Initial virtual bonds:")
square_J1_print_bond_spaces(lambda_x, lambda_y; prefix="  ")
starting_time = now()

function save_simple_update_step(T_now, lx_now, ly_now, step, report, error)
    (step % save_every == 0 || step == Int(round(tau / dt))) || return nothing
    elapsed = Dates.canonicalize(Dates.CompoundPeriod(now() - starting_time))
    jldsave(
        save_filename;
        T_set=T_now,
        lambda_x=lx_now,
        lambda_y=ly_now,
        Lx=cell_Lx,
        Ly=cell_Ly,
        Dmax=Dmax,
        Dstar=Dstar,
        J1=J1,
        tau_completed=step * dt,
        dt=dt,
        multiplet_tol=multiplet_tol_su,
        completed_steps=step,
        convergence_error=error,
        bond_spaces=report,
    )
    println("Saved step $step to $save_filename (elapsed $elapsed)")
    return nothing
end

T_set, lambda_x, lambda_y, history = square_J1_simple_update_cell(
    T_set,
    lambda_x,
    lambda_y,
    tau,
    dt;
    J1=J1,
    settings=settings,
    callback=save_simple_update_step,
)

final_report = square_J1_bond_space_report(lambda_x, lambda_y)
jldsave(
    save_filename;
    T_set,
    lambda_x,
    lambda_y,
    Lx=cell_Lx,
    Ly=cell_Ly,
    Dmax,
    Dstar,
    J1,
    tau,
    dt,
    multiplet_tol=multiplet_tol_su,
    history,
    bond_spaces=final_report,
)
println("Simple Update finished; final state saved to $save_filename")
