using TensorKit
using LinearAlgebra
using JLD2
using KrylovKit
using ChainRulesCore
using Zygote
using Zygote: @ignore_derivatives

# Resolve command-line paths before the repository driver changes directory.
const VERIFY_INPUT_FILE = isempty(ARGS) ? "" : abspath(ARGS[1])
const VERIFY_OUTPUT_FILE = length(ARGS) >= 2 ? abspath(ARGS[2]) : ""
const VERIFY_IS_MAIN = abspath(PROGRAM_FILE) == abspath(@__FILE__)

include("FullUpdate_iPESS_J1_Jchi_up.jl")


"""
    verify_dense_bosonic_J1_Jchi_energy(input_file, output_file; kwargs...)

Load a dense bosonic triangular iPESS checkpoint, converge its bosonic CTM
environment, and measure both the full and one-orientation-chirality energies.
No imaginary-time or full-update step is performed.
"""
function verify_dense_bosonic_J1_Jchi_energy(
    input_file::AbstractString,
    output_file::AbstractString;
    J1=1.0,
    Jchi=0.4,
    environment_chi::Integer=80,
    ctm_conv_tol=1e-6,
    ctm_iterations::Integer=50,
    ctm_trun_tol=1e-8,
    projector_strategy="4x4",
)
    B_set, T_set, _ = load_bosonic_iPESS_sets(input_file)
    Lx_state, Ly_state = size(B_set)
    size(T_set) == (Lx_state, Ly_state) || error("B_set and T_set cell sizes differ")

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

    A_cell = convert_iPESS_to_iPEPS(B_set, T_set)
    init = initial_condition(
        init_type="PBC",
        reconstruct_CTM=true,
        reconstruct_AA=true,
    )
    ctm_result = CTMRG_cell(A_cell, environment_chi, init, [], ctm_setting)
    CTM_cell, AA_cell, _, _, _, _, ite_num, ite_err =
        bosonic_unpack_ctm_result(ctm_result)

    energy = bosonic_measure_J1_Jchi_energy(
        A_cell,
        AA_cell,
        CTM_cell;
        J1=J1,
        Jchi=Jchi,
    )
    println("CTMRG summary: chi=$environment_chi, iterations=$ite_num, error=$ite_err")
    bosonic_print_J1_Jchi_energy(energy)

    metadata = (
        model="triangular J1-Jchi",
        input_file=abspath(input_file),
        unit_cell=(Lx_state, Ly_state),
        bond_dimension=dim(space(B_set[1, 1], 1)),
        physical_dimension=dim(space(T_set[1, 1], 2)),
        J1=J1,
        Jchi=Jchi,
        environment_chi=environment_chi,
        ctm_conv_tol=ctm_conv_tol,
        ctm_iterations=ite_num,
        ctm_error=ite_err,
        ctm_trun_tol=ctm_trun_tol,
        projector_strategy=projector_strategy,
    )
    jldsave(output_file; energy=energy, metadata=metadata)
    println("Saved Julia energy verification to $(abspath(output_file))")
    return (energy=energy, metadata=metadata)
end


if VERIFY_IS_MAIN
    length(ARGS) in (2, 3, 4, 5) || error(
        "usage: julia verify_dense_bosonic_J1_Jchi_energy.jl " *
        "INPUT.jld2 OUTPUT.jld2 [J1=1.0] [Jchi=0.4] [chi=80]"
    )
    J1 = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1.0
    Jchi = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.4
    environment_chi = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 80
    verify_dense_bosonic_J1_Jchi_energy(
        VERIFY_INPUT_FILE,
        VERIFY_OUTPUT_FILE;
        J1=J1,
        Jchi=Jchi,
        environment_chi=environment_chi,
    )
end
