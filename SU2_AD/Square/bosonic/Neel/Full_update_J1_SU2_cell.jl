using TensorKit
import TensorKit: ×
using Zygote
using LinearAlgebra: I, diag, diagm, dot, norm
using KrylovKit
using ChainRulesCore
using JLD2
using Random
using Dates
using Zygote: @ignore_derivatives

cd(@__DIR__)

include("../../../src/bosonic/square/square_spin_operator.jl")
include("../../../src/bosonic/iPEPS_ansatz.jl")
include("../../../src/bosonic/Settings.jl")
include("../../../src/bosonic/Settings_cell.jl")
include("../../../src/bosonic/CTMRG.jl")
include("../../../src/bosonic/CTMRG_unitcell.jl")
include("../../../src/bosonic/square/square_model.jl")
include("../../../src/bosonic/square/simple_update_lib.jl")
include("../../../src/bosonic/square/full_update_J1.jl")
include("../../../src/bosonic/square/full_update_J1_cell.jl")
include("square_J1_initial_states.jl")

Random.seed!(parse(Int, get(ENV, "FU_SEED", "666")))

function full_update_cell_virtual_space(D::Int)
    D == 3 && return SU2Space(0 => 1, 1 / 2 => 1)
    D == 4 && return SU2Space(0 => 1, 1 => 1)
    D == 5 && return SU2Space(0 => 1, 1 / 2 => 2)
    D == 6 && return SU2Space(0 => 1, 1 / 2 => 1, 1 => 1)
    D == 7 && return SU2Space(0 => 1, 1 / 2 => 3)
    D == 8 && return SU2Space(0 => 1, 1 / 2 => 2, 1 => 1)
    D == 9 && return SU2Space(0 => 2, 1 / 2 => 2, 1 => 1)
    D == 11 && return SU2Space(0 => 1, 1 / 2 => 2, 1 => 2)
    D == 16 && return SU2Space(0 => 1, 1 / 2 => 3, 1 => 3)
    error(
        "No default SU(2) multiplet structure for D=$D. " *
        "Load a state with FU_INIT or add the desired virtual space explicitly.",
    )
end

function random_full_update_cell_tensor(Vv)
    Vp = SU2Space(1 / 2 => 1)
    # Directly use the oriented square-iPEPS Hom-space
    # (L,D,R,U) = (Vv,Vv,Vv',Vv').
    codom = Vv ⊗ Vv ⊗ Vv' ⊗ Vv'
    A = try
        TensorMap(randn, codom, Vp)
    catch error_old_constructor
        tensor_space = codom ← Vp
        try
            TensorMap(randn(ComplexF64, dim(tensor_space)), tensor_space)
        catch
            rethrow(error_old_constructor)
        end
    end
    A = permute(A, (1, 2, 3, 4, 5), ())
    norm_A = norm(A)
    isfinite(norm_A) && norm_A > 0 || error(
        "The homogeneous SU(2) intertwiner space is empty for virtual space " *
        "$Vv and physical spin 1/2. For total dimension D=4, a valid " *
        "homogeneous choice is SU2Space(0=>2, 1/2=>1). The paper's D*=4 " *
        "state instead has total D=12 and uses three integer-spin legs plus " *
        "one half-integer-spin leg; it requires a leg-dependent cell initializer.",
    )
    return A / norm_A
end

function _full_update_cell_entry_tensor(entry)
    entry isa TensorMap && return entry
    hasproperty(entry, :T) && return getproperty(entry, :T)
    throw(ArgumentError("cannot extract a TensorMap from cell entry $(typeof(entry))"))
end

function _full_update_cell_from_loaded(value)
    raw = value isa Tuple ? square_fu_cell_to_matrix(value) : value
    raw isa AbstractMatrix ||
        throw(ArgumentError("loaded cell must be a matrix or a tuple of tuples"))
    return [_square_fu_normalize(_full_update_cell_entry_tensor(raw[cx, cy]))
            for cx in axes(raw, 1), cy in axes(raw, 2)]
end

function load_full_update_cell(filename::String, requested_Lx::Int, requested_Ly::Int)
    data = load(filename)
    value = if haskey(data, "A_set")
        data["A_set"]
    elseif haskey(data, "T_set")
        data["T_set"]
    elseif haskey(data, "A_cell")
        data["A_cell"]
    elseif haskey(data, "x")
        data["x"]
    elseif haskey(data, "A")
        fill(data["A"], requested_Lx, requested_Ly)
    else
        error("FU_INIT must contain A_set, T_set, A_cell, x, or A")
    end
    A_set = _full_update_cell_from_loaded(value)
    size(A_set) == (requested_Lx, requested_Ly) || error(
        "FU_INIT contains a $(size(A_set, 1))×$(size(A_set, 2)) cell, " *
        "but FU_LX×FU_LY=$requested_Lx×$requested_Ly",
    )
    return A_set
end

cell_Lx = parse(Int, get(ENV, "FU_LX", "2"))
cell_Ly = parse(Int, get(ENV, "FU_LY", "2"))
requested_D = parse(Int, get(ENV, "FU_D", "4"))
chi = parse(Int, get(ENV, "FU_CHI", "40"))
tau = parse(Float64, get(ENV, "FU_TAU", "0.5"))
dt = parse(Float64, get(ENV, "FU_DT", "0.01"))
J1 = parse(Float64, get(ENV, "FU_J1", "1.0"))
init_filename = get(ENV, "FU_INIT", "nothing")
init_kind = Symbol(get(ENV, "FU_INIT_KIND", "homogeneous"))

if init_filename == "nothing"
    if init_kind === :homogeneous
        D = requested_D
        global Vv = full_update_cell_virtual_space(D)
        A_set = [random_full_update_cell_tensor(Vv) for _ in 1:cell_Lx, _ in 1:cell_Ly]
    else
        (cell_Lx, cell_Ly) == (2, 2) || error(
            "FU_INIT_KIND=$init_kind is a named 2×2 matching; set FU_LX=2 FU_LY=2",
        )
        A_set, _, _ = square_J1_named_initial_state(
            init_kind,
            parse(Int, get(ENV, "FU_SEED", "666")),
        )
        global Vv = space(A_set[1, 1], 1)
        D = maximum(
            dim(space(A_set[cx, cy], leg))
            for cx in 1:cell_Lx, cy in 1:cell_Ly, leg in 1:4
        )
    end
else
    A_set = load_full_update_cell(init_filename, cell_Lx, cell_Ly)
    global Vv = space(A_set[1, 1], 1)
    D = dim(Vv)
    if haskey(ENV, "FU_D") && requested_D != D
        error("FU_D=$requested_D, but the loaded state has D=$D and virtual space $Vv")
    end
end
_square_fu_validate_cell(A_set)
default_Dmax = init_filename == "nothing" && startswith(string(init_kind), "minimal_") ?
    12 : D

save_filename = get(
    ENV,
    "FU_SAVE",
    "FullUpdate_J1_$(cell_Lx)x$(cell_Ly)_D_$(D)_chi_$(chi).jld2",
)

ctm_setting = LS_CTMRG_settings()
ctm_setting.CTM_conv_tol = parse(Float64, get(ENV, "FU_CTM_TOL", "1e-6"))
ctm_setting.CTM_ite_nums = parse(Int, get(ENV, "FU_CTM_MAXITER", "150"))
ctm_setting.CTM_trun_tol = 1e-8
ctm_setting.svd_lanczos_tol = 1e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = true
ctm_setting.CTM_conv_info = true
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true

global Lx = cell_Lx
global Ly = cell_Ly
global multiplet_tol = parse(Float64, get(ENV, "FU_MULTIPLET_TOL", "1e-5"))
global projector_trun_tol = ctm_setting.CTM_trun_tol
global backward_settings = Backward_settings()
global algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = get(
    ENV, "FU_CTM_CELL_METHOD", "continuous_update",
)

fu_settings = SquareJ1FullUpdateSettings(
    Dmax=parse(Int, get(ENV, "FU_DMAX", string(default_Dmax))),
    multiplet_tol=parse(Float64, get(ENV, "FU_MULTIPLET_TOL", "1e-5")),
    maxiter=parse(Int, get(ENV, "FU_LOCAL_MAXITER", "20")),
    gradient_tolerance=parse(Float64, get(ENV, "FU_GRAD_TOL", "1e-8")),
    loss_tolerance=parse(Float64, get(ENV, "FU_LOSS_TOL", "1e-12")),
    initial_step=parse(Float64, get(ENV, "FU_INITIAL_STEP", "0.2")),
    refresh_environment=parse(Bool, get(ENV, "FU_REFRESH_CTM", "true")),
    verbose=parse(Bool, get(ENV, "FU_VERBOSE", "true")),
)

groups = square_J1_bond_groups(cell_Lx, cell_Ly)
println("Periodic bond-group sizes: $(map(length, groups))")
starting_time = now()

function save_and_measure_cell(A_set_now, environment, step, reports)
    energies = square_J1_energy_cell(A_set_now, environment; J1=J1)
    elapsed = Dates.canonicalize(Dates.CompoundPeriod(now() - starting_time))
    println(
        "FU sweep $step: E/site=$(energies.energy_per_site), " *
        "mean(Ex)=$(sum(energies.Ex) / length(energies.Ex)), " *
        "mean(Ey)=$(sum(energies.Ey) / length(energies.Ey)), " *
        "CTM error=$(environment.ite_err), elapsed=$elapsed",
    )
    jldsave(
        save_filename;
        A_set=A_set_now,
        A_cell=square_fu_cell_to_tuple(A_set_now),
        Lx=cell_Lx,
        Ly=cell_Ly,
        D=D,
        initial_D=D,
        Dmax=fu_settings.Dmax,
        init_kind=init_kind,
        init_filename=init_filename,
        multiplet_tol=fu_settings.multiplet_tol,
        chi=chi,
        J1=J1,
        tau=tau,
        dt=dt,
        completed_sweeps=step,
        reports=reports,
        Ex=energies.Ex,
        Ey=energies.Ey,
        energy_per_site=energies.energy_per_site,
    )
end

println("Starting bosonic square-lattice J1 cell Full Update")
println(
    "cell=$(cell_Lx)×$(cell_Ly), D=$D, Dmax=$(fu_settings.Dmax), " *
    "multiplet_tol=$(fu_settings.multiplet_tol), Vv=$Vv, chi=$chi, " *
    "tau=$tau, dt=$dt, J1=$J1, init=$init_filename, init_kind=$init_kind",
)
A_set, environment, history = square_J1_full_update_cell(
    A_set,
    chi,
    tau,
    dt,
    ctm_setting;
    J1=J1,
    settings=fu_settings,
    callback=save_and_measure_cell,
)

final_energies = square_J1_energy_cell(A_set, environment; J1=J1)
jldsave(
    save_filename;
    A_set=A_set,
    A_cell=square_fu_cell_to_tuple(A_set),
    Lx=cell_Lx,
    Ly=cell_Ly,
    D=D,
    initial_D=D,
    Dmax=fu_settings.Dmax,
    init_kind=init_kind,
    init_filename=init_filename,
    multiplet_tol=fu_settings.multiplet_tol,
    chi=chi,
    J1=J1,
    tau=tau,
    dt=dt,
    history=history,
    Ex=final_energies.Ex,
    Ey=final_energies.Ey,
    energy_per_site=final_energies.energy_per_site,
)
println("Full Update finished; state saved to $save_filename")
