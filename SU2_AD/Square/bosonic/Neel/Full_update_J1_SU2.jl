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
include("../../../src/bosonic/CTMRG.jl")
include("../../../src/bosonic/square/square_model.jl")
include("../../../src/bosonic/square/simple_update_lib.jl")
include("../../../src/bosonic/square/full_update_J1.jl")

Random.seed!(parse(Int, get(ENV, "FU_SEED", "666")))

function full_update_virtual_space(D::Int)
    D == 3 && return SU2Space(0 => 1, 1 / 2 => 1)
    D == 4 && return SU2Space(0 => 2, 1 / 2 => 1)
    D == 5 && return SU2Space(0 => 1, 1 / 2 => 2)
    D == 6 && return SU2Space(0 => 1, 1 / 2 => 1, 1 => 1)
    D == 7 && return SU2Space(0 => 1, 1 / 2 => 3)
    D == 8 && return SU2Space(0 => 1, 1 / 2 => 2, 1 => 1)
    D == 9 && return SU2Space(0 => 2, 1 / 2 => 2, 1 => 1)
    D == 11 && return SU2Space(0 => 1, 1 / 2 => 2, 1 => 2)
    D == 16 && return SU2Space(0 => 1, 1 / 2 => 3, 1 => 3)
    error("No default SU(2) multiplet structure for D=$D. Load a state with FU_INIT or add the desired Vv explicitly.")
end

function random_full_update_tensor(Vv)
    Vp = SU2Space(1 / 2 => 1)
    # iPEPS convention: (L,D,R,U) = (Vv,Vv,Vv',Vv').  Construct the
    # intertwiner in this oriented Hom-space directly, as in the older square
    # and triangular iPEPS initializers, without an auxiliary basis map.
    codom = Vv ⊗ Vv ⊗ Vv' ⊗ Vv'
    A = try
        # TensorKit <= 0.12
        TensorMap(randn, codom, Vp)
    catch error_old_constructor
        # New TensorKit constructor: provide the independent symmetry-block data.
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

function load_full_update_state(filename::String)
    filename == "nothing" && error("random initialization requires an explicit virtual space")
    data = load(filename)
    haskey(data, "A") || error("FU_INIT must contain an iPEPS tensor under key A")
    A = data["A"]
    return A / norm(A)
end

requested_D = parse(Int, get(ENV, "FU_D", "4"))
chi = parse(Int, get(ENV, "FU_CHI", "40"))
tau = parse(Float64, get(ENV, "FU_TAU", "0.1"))
dt = parse(Float64, get(ENV, "FU_DT", "0.01"))
J1 = parse(Float64, get(ENV, "FU_J1", "1.0"))
init_filename = get(ENV, "FU_INIT", "nothing")

if init_filename == "nothing"
    D = requested_D
    global Vv = full_update_virtual_space(D)
    A = random_full_update_tensor(Vv)
else
    A = load_full_update_state(init_filename)
    global Vv = space(A, 1)
    D = dim(Vv)
    if haskey(ENV, "FU_D") && requested_D != D
        error("FU_D=$requested_D, but the loaded state has D=$D and virtual space $Vv")
    end
end
@assert dim(Vv) == D
save_filename = get(ENV, "FU_SAVE", "FullUpdate_J1_D_$(D)_chi_$(chi).jld2")

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

global multiplet_tol = 1e-5
global projector_trun_tol = ctm_setting.CTM_trun_tol
global backward_settings = Backward_settings()

fu_settings = SquareJ1FullUpdateSettings(
    maxiter=parse(Int, get(ENV, "FU_LOCAL_MAXITER", "20")),
    gradient_tolerance=parse(Float64, get(ENV, "FU_GRAD_TOL", "1e-8")),
    loss_tolerance=parse(Float64, get(ENV, "FU_LOSS_TOL", "1e-12")),
    initial_step=parse(Float64, get(ENV, "FU_INITIAL_STEP", "0.2")),
    refresh_environment=true,
    verbose=true,
)

starting_time = now()

function save_and_measure(A, environment, step, reports)
    Ex, Ey = evaluate_NN(
        A,
        environment.AA,
        environment.U_L,
        environment.U_D,
        environment.U_R,
        environment.U_U,
        environment.CTM,
        ctm_setting,
    )
    elapsed = Dates.canonicalize(Dates.CompoundPeriod(now() - starting_time))
    println(
        "FU sweep $step: Ex=$(real(Ex)), Ey=$(real(Ey)), " *
        "E/site=$(real(Ex + Ey)), CTM error=$(environment.ite_err), elapsed=$elapsed",
    )
    jldsave(
        save_filename;
        A=A,
        D=D,
        chi=chi,
        J1=J1,
        tau=tau,
        dt=dt,
        completed_sweeps=step,
        reports=reports,
    )
end

println("Starting bosonic square-lattice J1 Full Update")
println("D=$D, Vv=$Vv, chi=$chi, tau=$tau, dt=$dt, J1=$J1")
A, environment, history = square_J1_full_update(
    A,
    chi,
    tau,
    dt,
    ctm_setting;
    J1=J1,
    settings=fu_settings,
    callback=save_and_measure,
)

jldsave(save_filename; A=A, D=D, chi=chi, J1=J1, tau=tau, dt=dt, history=history)
println("Full Update finished; state saved to $save_filename")
