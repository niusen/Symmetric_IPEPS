using TensorKit
using LinearAlgebra
using JLD2
using MAT
using KrylovKit
using ChainRulesCore
using Zygote
using Zygote: @ignore_derivatives
using Random

const DENSE_ES_DRIVER_IS_MAIN = abspath(PROGRAM_FILE) == abspath(@__FILE__)

# ======================== user settings ========================
# Edit these values directly; this driver does not use ARGS.
const DENSE_ES_INPUT_FILE = joinpath(
    @__DIR__,
    "triangle_spin_J1_1_Jchi_0p4_D8_chi80_Lx2_Ly1_-0.60645.jld2",
)
const DENSE_ES_CHI = 60
const DENSE_ES_NV = 6                 # Supported: 4, 5, 6, 8
const DENSE_ES_EIGENVALUE_NUMBER = 30
const DENSE_ES_USE_K_PROJECTOR = false
const DENSE_ES_CUT_X_VALUES = [1, 2]  # The two inequivalent cuts of a 2x1 cell
const DENSE_ES_CUT_Y = 1
const DENSE_ES_T_SCALE = 10
const DENSE_ES_KRYLOVDIM = 2 * DENSE_ES_EIGENVALUE_NUMBER + 5
const DENSE_ES_OUTPUT_DIRECTORY = @__DIR__

const DENSE_ES_CTM_CONV_TOL = 1e-6
const DENSE_ES_CTM_ITERATIONS = 100
const DENSE_ES_CTM_TRUN_TOL = 1e-8
const DENSE_ES_PROJECTOR_STRATEGY = "4x4"
# ===============================================================

cd(@__DIR__)

include("../src/bosonic/Settings.jl")
include("../src/bosonic/Settings_cell.jl")
include("../src/bosonic/iPEPS_ansatz.jl")
include("../src/bosonic/AD_lib.jl")
include("../src/bosonic/CTMRG.jl")
include("../src/bosonic/CTMRG_unitcell.jl")
include("../src/bosonic/triangle/triangle_iPESS_method.jl")
include("../src/bosonic/triangle/simple_update/triangle_SimpleUpdate_iPESS.jl")
include("../src/mps_algorithms/ES_CTM_algorithms_dense_matrixfree.jl")


function _dense_es_load_ipess_sets(input_file::AbstractString)
    data = load(input_file)
    if haskey(data, "B_set") && haskey(data, "T_set")
        return data["B_set"], data["T_set"]
    elseif haskey(data, "x")
        state = data["x"]
        B_set = Matrix{TensorMap}(undef, size(state)...)
        T_set = Matrix{TensorMap}(undef, size(state)...)
        for index in eachindex(state)
            hasproperty(state[index], :Tm) ||
                error("x[$index] has no simplex tensor Tm.")
            hasproperty(state[index], :Bm) ||
                error("x[$index] has no physical tensor Bm.")
            B_set[index] = state[index].Tm
            T_set[index] = state[index].Bm
        end
        return B_set, T_set
    end
    error("Checkpoint must contain B_set/T_set or x; keys=$(collect(keys(data))).")
end


function _dense_es_unpack_ctm_result(result)
    if length(result) == 8
        return result
    elseif length(result) == 6
        CTM_cell, AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell = result
        return CTM_cell, AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell, missing, missing
    end
    error("Unexpected CTMRG_cell return length $(length(result)); expected 6 or 8.")
end


function _dense_es_boundary(CTM_cell, U_L_cell, U_R_cell, cx::Int, cy::Int)
    T2 = CTM_cell.Tset[mod1(cx + 1, Lx)][mod1(cy, Ly)].T2
    T4 = CTM_cell.Tset[mod1(cx, Lx)][mod1(cy, Ly)].T4
    CTM = (Tset=(T2=T2, T4=T4),)
    return CTM, U_L_cell[cx][cy], U_R_cell[cx][cy]
end


function _dense_es_output_filename(
    input_file::AbstractString,
    D::Int,
    chi_value::Int,
    Nv_value::Int,
    cut_x::Int,
    use_Kprojector_value::Bool,
    output_directory::AbstractString,
)
    state_tag = splitext(basename(input_file))[1]
    projector_tag = use_Kprojector_value ? "Kprojector" : "noKprojector"
    filename = "ES_CTM_dense_matrixfree_$(state_tag)_D$(D)_chi$(chi_value)" *
        "_Nv$(Nv_value)_cutx$(cut_x)_$(projector_tag).mat"
    return joinpath(output_directory, filename)
end


"""
    run_dense_iPESS_ES_CTM_matrixfree(input_file; kwargs...)

Load a dense bosonic triangular iPESS, converge one unit-cell CTM environment,
and calculate the entanglement spectrum for the requested vertical cuts.  The
Lanczos/Arnoldi map is matrix-free: it stores boundary vectors of length
`D^Nv`, but never forms a `D^Nv` by `D^Nv` transfer matrix.

The present driver requires `Ly == 1`, so one-site translation around the
cylinder is a symmetry.  A state with `Ly > 1` needs an `Ly`-site transfer
operator and unit-cell momentum convention instead of silently repeating one
row tensor.
"""
function run_dense_iPESS_ES_CTM_matrixfree(
    input_file::AbstractString=DENSE_ES_INPUT_FILE;
    chi_value::Int=DENSE_ES_CHI,
    Nv_value::Int=DENSE_ES_NV,
    EH_n_value::Int=DENSE_ES_EIGENVALUE_NUMBER,
    use_Kprojector_value::Bool=DENSE_ES_USE_K_PROJECTOR,
    cut_x_values=DENSE_ES_CUT_X_VALUES,
    cut_y::Int=DENSE_ES_CUT_Y,
    T_scale_value=DENSE_ES_T_SCALE,
    krylovdim_value::Int=DENSE_ES_KRYLOVDIM,
    output_directory::AbstractString=DENSE_ES_OUTPUT_DIRECTORY,
)
    Nv_value in (4, 5, 6, 8) || error("Nv must be one of 4, 5, 6, 8.")
    isfile(input_file) || error("Input state does not exist: $input_file")
    mkpath(output_directory)

    B_set, T_set = _dense_es_load_ipess_sets(input_file)
    size(B_set) == size(T_set) || error("B_set and T_set cell sizes differ.")
    global Lx, Ly, chi, multiplet_tol, projector_trun_tol
    Lx, Ly = size(B_set)
    Ly == 1 || error(
        "This one-site-translation dense ES driver requires Ly=1; loaded cell is $(Lx)x$(Ly).",
    )
    1 <= cut_y <= Ly || error("cut_y=$cut_y lies outside Ly=$Ly.")
    all(cx -> 1 <= cx <= Lx, cut_x_values) ||
        error("cut_x_values=$(cut_x_values) lies outside Lx=$Lx.")

    chi = chi_value
    multiplet_tol = 1e-5
    projector_trun_tol = DENSE_ES_CTM_TRUN_TOL
    global algrithm_CTMRG_settings = Algrithm_CTMRG_settings(
        CTM_cell_ite_method="continuous_update",
    )

    ctm_setting = LS_CTMRG_settings(
        CTM_conv_tol=DENSE_ES_CTM_CONV_TOL,
        CTM_ite_nums=DENSE_ES_CTM_ITERATIONS,
        CTM_trun_tol=DENSE_ES_CTM_TRUN_TOL,
        svd_lanczos_tol=DENSE_ES_CTM_TRUN_TOL,
        projector_strategy=DENSE_ES_PROJECTOR_STRATEGY,
        conv_check="singular_value",
        CTM_ite_info=true,
        CTM_conv_info=true,
        CTM_trun_svd=false,
        construct_double_layer=true,
        grad_checkpoint=false,
    )

    A_cell = convert_iPESS_to_iPEPS(B_set, T_set)
    D = dim(space(B_set[1, 1], 1))
    println("Dense triangular iPESS CTM entanglement spectrum")
    println("  input       = $(abspath(input_file))")
    println("  cell        = $(Lx)x$(Ly)")
    println("  D           = $D")
    println("  chi         = $chi_value")
    println("  Nv          = $Nv_value")
    println("  EH_n        = $EH_n_value")
    println("  cut_x       = $cut_x_values")
    println("  K projector = $use_Kprojector_value")
    println("  matrix-free = true")
    flush(stdout)

    init = initial_condition(
        init_type="PBC",
        reconstruct_CTM=true,
        reconstruct_AA=true,
    )
    ctm_result = _dense_es_unpack_ctm_result(
        CTMRG_cell(A_cell, chi_value, init, [], ctm_setting),
    )
    CTM_cell, AA_cell, U_L_cell, _, U_R_cell, _, ite_num, ite_err = ctm_result
    println("CTMRG finished: iterations=$ite_num, error=$ite_err")
    flush(stdout)

    results = Dict{Int,Any}()
    for cut_x in cut_x_values
        CTM, U_L, U_R = _dense_es_boundary(
            CTM_cell, U_L_cell, U_R_cell, cut_x, cut_y,
        )
        output_file = _dense_es_output_filename(
            input_file,
            D,
            chi_value,
            Nv_value,
            cut_x,
            use_Kprojector_value,
            output_directory,
        )
        if use_Kprojector_value
            es_result = ES_CTMRG_ED_Kprojector_dense_matrixfree(
                CTM, U_L, U_R, D, chi_value, Nv_value, EH_n_value;
                save_filenm=output_file,
                T_scale=T_scale_value,
                krylovdim=krylovdim_value,
            )
        else
            es_result = ES_CTMRG_ED_dense_matrixfree(
                CTM, U_L, U_R, D, chi_value, Nv_value, EH_n_value;
                save_filenm=output_file,
                T_scale=T_scale_value,
                krylovdim=krylovdim_value,
            )
        end
        results[cut_x] = (result=es_result, output_file=output_file)
    end

    return (
        results=results,
        CTM_cell=CTM_cell,
        AA_cell=AA_cell,
        ctm_iterations=ite_num,
        ctm_error=ite_err,
    )
end


if DENSE_ES_DRIVER_IS_MAIN
    Random.seed!(555)
    run_dense_iPESS_ES_CTM_matrixfree()
end

nothing
