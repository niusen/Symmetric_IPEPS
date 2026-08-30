using TensorKit
using LinearAlgebra
using JLD2
using KrylovKit
using ChainRulesCore
using Zygote
using Zygote: @ignore_derivatives

const J1_FU_DRIVER_IS_MAIN = abspath(PROGRAM_FILE) == abspath(@__FILE__)

# ======================== user settings ========================
# Edit these values directly; this driver does not use ARGS.
const J1_FU_INPUT_FILE = joinpath(
    @__DIR__,
    "triangle_spin_J1_1_Jchi_0p4_D6_chi80_Lx2_Ly1_-0.60632.jld2",
)
const J1_FU_OUTPUT_FILE = joinpath(
    @__DIR__,
    "triangle_spin_J1_1_Jchi_0p4_D6_chi80_Lx2_Ly1_D6_to_D8.jld2",
)
const J1_FU_J1 = 1.0
const J1_FU_JCHI_UP = 0.4  # Set to 0.4 to update the up-triangle chirality too.
const J1_FU_D_MAX = 8
const J1_FU_ENVIRONMENT_CHI = 80
const J1_FU_DT = 0.02
const J1_FU_TOTAL_TAU = 0.02
# ===============================================================

cd(@__DIR__)

include("../src/bosonic/Settings.jl")
include("../src/bosonic/Settings_cell.jl")
include("../src/bosonic/iPEPS_ansatz.jl")
include("../src/bosonic/AD_lib.jl")
include("../src/bosonic/CTMRG.jl")
include("../src/bosonic/CTMRG_unitcell.jl")
include("../src/bosonic/square/square_spin_operator.jl")
include("../src/bosonic/square/square_model.jl")
include("../src/bosonic/square/square_model_cell.jl")
include("../src/bosonic/triangle/triangle_iPESS_method.jl")
include("../src/bosonic/triangle/simple_update/triangle_SimpleUpdate_iPESS.jl")
include("../src/bosonic/triangle/full_update/triangle_FullUpdate_iPESS.jl")


"""Read either a `B_set,T_set` checkpoint or a variational `x` checkpoint."""
function load_bosonic_iPESS_sets(input_file::AbstractString)
    data = load(input_file)
    if haskey(data, "B_set") && haskey(data, "T_set")
        return data["B_set"], data["T_set"], data
    elseif haskey(data, "x")
        state = data["x"]
        Lx_state, Ly_state = size(state)
        B_set = Matrix{TensorMap}(undef, Lx_state, Ly_state)
        T_set = Matrix{TensorMap}(undef, Lx_state, Ly_state)
        for cx in 1:Lx_state, cy in 1:Ly_state
            hasproperty(state[cx, cy], :Tm) ||
                error("x[$cx,$cy] has no Tm field; this is not a triangular iPESS state")
            hasproperty(state[cx, cy], :Bm) ||
                error("x[$cx,$cy] has no Bm field; this is not a triangular iPESS state")
            # Repository convention: B_set stores the rank-3 simplex tensor Tm,
            # while T_set stores the rank-4 physical tensor Bm.
            B_set[cx, cy] = state[cx, cy].Tm
            T_set[cx, cy] = state[cx, cy].Bm
        end
        return B_set, T_set, data
    end
    error("Checkpoint must contain either B_set/T_set or x; keys=$(collect(keys(data)))")
end


function iPESS_sets_to_state(B_set, T_set)
    Lx_state, Ly_state = size(B_set)
    state = Matrix{Triangle_iPESS}(undef, Lx_state, Ly_state)
    for cx in 1:Lx_state, cy in 1:Ly_state
        state[cx, cy] = Triangle_iPESS(T_set[cx, cy], B_set[cx, cy])
    end
    return state
end


"""
    run_triangle_full_update_up(input_file, output_file; kwargs...)

Load a bosonic triangular-lattice iPESS checkpoint and evolve it for
`total_tau` using full-update steps of size `dt` over the simplex-centred
up triangles. `total_tau` must be an integer multiple of `dt`.
The input bond dimension is detected from the checkpoint, while `D_max` sets
the output truncation dimension.

This up-triangle J1 decomposition contains every nearest-neighbour bond once;
therefore a separate down-triangle update is not required for the J1-only
Hamiltonian. It would be required for a chirality term on both orientations.
"""
function run_triangle_full_update_up(
    input_file::AbstractString,
    output_file::AbstractString;
    D_max::Integer=8,
    environment_chi::Integer=80,
    dt=0.01,
    total_tau=dt,
    J1=1.0,
    Jchi_up=0.0,
    n_sweep::Integer=10,
    trun_tol=1e-8,
    ctm_conv_tol=1e-6,
    ctm_iterations::Integer=50,
    ctm_trun_tol=1e-8,
    projector_strategy="4x4",
    reuse_saved_CTM::Bool=false,
)
    B_set, T_set, data = load_bosonic_iPESS_sets(input_file)
    input_Lx, input_Ly = size(B_set)
    size(T_set) == (input_Lx, input_Ly) || error("B_set and T_set cell sizes differ")

    target_Lx = max(input_Lx, 2)
    target_Ly = max(input_Ly, 2)
    expanded_cell = (target_Lx, target_Ly) != (input_Lx, input_Ly)
    if expanded_cell
        println("periodically expanding unit cell: $(input_Lx)x$(input_Ly) -> $(target_Lx)x$(target_Ly)")
        B_set = [
            deepcopy(B_set[mod1(cx, input_Lx), mod1(cy, input_Ly)])
            for cx in 1:target_Lx, cy in 1:target_Ly
        ]
        T_set = [
            deepcopy(T_set[mod1(cx, input_Lx), mod1(cy, input_Ly)])
            for cx in 1:target_Lx, cy in 1:target_Ly
        ]
    end
    Lx_state, Ly_state = size(B_set)

    input_dimensions = [
        dim(space(B_set[cx, cy], leg))
        for cx in 1:Lx_state for cy in 1:Ly_state for leg in 1:3
    ]
    D_max >= maximum(input_dimensions) ||
        error("D_max=$D_max must not be smaller than the input dimensions $input_dimensions")
    n_time_steps = round(Int, total_tau / dt)
    @assert n_time_steps > 0 && n_time_steps * dt ≈ total_tau

    global Lx = Lx_state
    global Ly = Ly_state
    global chi = environment_chi
    global multiplet_tol = 1e-5
    global projector_trun_tol = ctm_trun_tol
    global algrithm_CTMRG_settings = Algrithm_CTMRG_settings(
        CTM_cell_ite_method="continuous_update"
    )

    ctm_setting = LS_CTMRG_settings(
        CTM_conv_tol=ctm_conv_tol,
        CTM_ite_nums=ctm_iterations,
        CTM_trun_tol=ctm_trun_tol,
        svd_lanczos_tol=ctm_trun_tol,
        projector_strategy=projector_strategy,
        conv_check="singular_value",
        CTM_ite_info=true,
        CTM_conv_info=true,
        CTM_trun_svd=false,
        construct_double_layer=true,
        grad_checkpoint=false,
    )

    can_reuse_saved_CTM = reuse_saved_CTM && !expanded_cell && haskey(data, "CTM_cell")
    reuse_saved_CTM && expanded_cell && println(
        "saved CTM is not reused because its cell size is incompatible with the expanded state"
    )
    CTM_work = can_reuse_saved_CTM ? data["CTM_cell"] : []
    B_work = B_set
    T_work = T_set
    result = nothing
    for time_step in 1:n_time_steps
        println(
            "imaginary-time step $time_step/$n_time_steps: " *
            "tau=$(time_step * dt)/$total_tau, dt=$dt"
        )
        result = bosonic_full_update_iPESS_J1_once(
            B_work,
            T_work,
            environment_chi,
            dt,
            D_max,
            ctm_setting;
            J1=J1,
            Jchi_up=Jchi_up,
            n_sweep=n_sweep,
            trun_order="simultaneous",
            trun_tol=trun_tol,
            init_CTM=CTM_work,
        )
        B_work = result.B_set
        T_work = result.T_set
        CTM_work = result.CTM_cell
    end

    x = iPESS_sets_to_state(result.B_set, result.T_set)
    metadata = (
        model="triangular J1-Jchi",
        update="repeated up-triangle full-update sweeps",
        input_file=abspath(input_file),
        input_cell_size=(input_Lx, input_Ly),
        output_cell_size=(Lx_state, Ly_state),
        periodically_expanded_cell=expanded_cell,
        input_bond_dimensions=input_dimensions,
        output_bond_dimensions=result.bond_dimensions,
        D_max=D_max,
        environment_chi=environment_chi,
        dt=dt,
        total_tau=total_tau,
        n_time_steps=n_time_steps,
        J1=J1,
        Jchi_up=Jchi_up,
        n_sweep=n_sweep,
        trun_tol=trun_tol,
        ctm_error=result.ctm_error,
        ctm_iterations=result.ctm_iterations,
        energy_J1=result.energy.energy_J1,
        energy_chi_up=result.energy.energy_chi_up,
        energy_chi_down=result.energy.energy_chi_down,
        energy_half_chirality=result.energy.energy_half_chirality,
        energy_full_J1_Jchi=result.energy.energy_full_J1_Jchi,
        reused_saved_CTM=can_reuse_saved_CTM,
    )
    jldsave(
        output_file;
        B_set=result.B_set,
        T_set=result.T_set,
        x=x,
        CTM_cell=result.CTM_cell,
        energy=result.energy,
        metadata=metadata,
    )
    println("saved D=$D_max state to $(abspath(output_file))")
    return result
end


if J1_FU_DRIVER_IS_MAIN
    isfile(J1_FU_INPUT_FILE) || error("Input checkpoint not found: $J1_FU_INPUT_FILE")
    run_triangle_full_update_up(
        J1_FU_INPUT_FILE,
        J1_FU_OUTPUT_FILE;
        D_max=J1_FU_D_MAX,
        environment_chi=J1_FU_ENVIRONMENT_CHI,
        dt=J1_FU_DT,
        total_tau=J1_FU_TOTAL_TAU,
        J1=J1_FU_J1,
        Jchi_up=J1_FU_JCHI_UP,
    )
end
