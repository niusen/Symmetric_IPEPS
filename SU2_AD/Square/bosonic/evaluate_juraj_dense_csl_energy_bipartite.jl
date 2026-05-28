using TensorKit
using Zygote
using Zygote: @ignore_derivatives
using ChainRulesCore
using LinearAlgebra: norm
using JLD2
using JSON
using KrylovKit

cd(@__DIR__)

include("..\\..\\src\\bosonic\\square\\square_spin_operator.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\bosonic\\square\\square_model.jl")
include("..\\..\\src\\bosonic\\square\\square_AD_SU2.jl")
include("..\\..\\src\\bosonic\\Settings.jl")
include("read_juraj_ipeps_tensor.jl")

juraj_state_json = raw"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev\data_c4_pt_csl\j1j2lambda_c4pt_D3_chi40_seed0_gpu_state_-0.9865"

chi = 40

J1 = 2 * cos(0.06 * pi) * cos(0.14 * pi)
J2 = 2 * cos(0.06 * pi) * sin(0.14 * pi)
Jchi = 2 * sin(0.06 * pi) * 2
parameters = Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)])

ctm_setting = LS_CTMRG_settings()
ctm_setting.CTM_conv_tol = 1e-6
ctm_setting.CTM_ite_nums = 50
ctm_setting.CTM_trun_tol = 1e-8
ctm_setting.svd_lanczos_tol = 1e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = true
ctm_setting.CTM_conv_info = true
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true
ctm_setting.grad_checkpoint = false

global multiplet_tol, projector_trun_tol, backward_settings
multiplet_tol = 1e-5
projector_trun_tol = ctm_setting.CTM_trun_tol
backward_settings = Backward_settings()

function bipartite_spin_rotation(space_phy)
    @assert dim(space_phy) == 2 "BP_rot is implemented here only for spin-1/2 dense physical space."
    rot = zeros(ComplexF64, 2, 2)
    rot[1, 2] = 1
    rot[2, 1] = -1
    return TensorMap(rot, space_phy, space_phy)
end

function rotate_physical_leg(A::TensorMap)
    R = bipartite_spin_rotation(space(A, 5))
    @tensor A_rot[:] := A[-1, -2, -3, -4, 1] * R[-5, 1]
    return A_rot
end

function evaluate_triangle_bipartite_rotation(H_triangle, A::TensorMap, AA, CTM, ctm_setting)
    A_even = A
    A_odd = rotate_physical_leg(A)

    AA_even, U_s_s = build_double_layer_open(A_even)
    AA_odd, _ = build_double_layer_open(A_odd)

    # Juraj's C4V_BIPARTITE convention applies BP_rot on the odd sublattice.
    # For a 2x2 plaquette ordered as
    #   LU  RU
    #   LD  RD
    # the odd sites are RU and LD, matching the rotations in j1j2lambda.py.
    AA_LU = AA_even
    AA_RU = AA_odd
    AA_LD = AA_odd
    AA_RD = AA_even

    rho_LU_RU_LD = ob_LU_RU_LD(CTM, AA, AA_LU, AA_RU, AA_LD)
    @tensor rho_LU_RU_LD[:] := rho_LU_RU_LD[1, 2, 3] *
        U_s_s[-1, -4, 1] * U_s_s[-2, -5, 2] * U_s_s[-3, -6, 3]
    norm_LU_RU_LD = @tensor rho_LU_RU_LD[1, 2, 3, 1, 2, 3]
    E_LU_RU_LD = @tensor rho_LU_RU_LD[1, 2, 3, 4, 5, 6] *
        H_triangle[1, 2, 3, 4, 5, 6]
    E_LU_RU_LD = E_LU_RU_LD / norm_LU_RU_LD

    rho_LD_RU_RD = ob_LD_RU_RD(CTM, AA, AA_LD, AA_RU, AA_RD)
    rho_LD_RU_RD = permute(rho_LD_RU_RD, (3, 1, 2,))
    @tensor rho_LD_RU_RD[:] := rho_LD_RU_RD[1, 2, 3] *
        U_s_s[-1, -4, 1] * U_s_s[-2, -5, 2] * U_s_s[-3, -6, 3]
    norm_LD_RU_RD = @tensor rho_LD_RU_RD[1, 2, 3, 1, 2, 3]
    E_LD_RU_RD = @tensor rho_LD_RU_RD[1, 2, 3, 4, 5, 6] *
        H_triangle[1, 2, 3, 4, 5, 6]
    E_LD_RU_RD = E_LD_RU_RD / norm_LD_RU_RD

    rho_LU_LD_RD = ob_LU_LD_RD(CTM, AA, AA_LU, AA_LD, AA_RD)
    rho_LU_LD_RD = permute(rho_LU_LD_RD, (2, 1, 3,))
    @tensor rho_LU_LD_RD[:] := rho_LU_LD_RD[1, 2, 3] *
        U_s_s[-1, -4, 1] * U_s_s[-2, -5, 2] * U_s_s[-3, -6, 3]
    norm_LU_LD_RD = @tensor rho_LU_LD_RD[1, 2, 3, 1, 2, 3]
    E_LU_LD_RD = @tensor rho_LU_LD_RD[1, 2, 3, 4, 5, 6] *
        H_triangle[1, 2, 3, 4, 5, 6]
    E_LU_LD_RD = E_LU_LD_RD / norm_LU_LD_RD

    rho_LU_RU_RD = ob_LU_RU_RD(CTM, AA, AA_LU, AA_RU, AA_RD)
    rho_LU_RU_RD = permute(rho_LU_RU_RD, (2, 3, 1,))
    @tensor rho_LU_RU_RD[:] := rho_LU_RU_RD[1, 2, 3] *
        U_s_s[-1, -4, 1] * U_s_s[-2, -5, 2] * U_s_s[-3, -6, 3]
    norm_LU_RU_RD = @tensor rho_LU_RU_RD[1, 2, 3, 1, 2, 3]
    E_LU_RU_RD = @tensor rho_LU_RU_RD[1, 2, 3, 4, 5, 6] *
        H_triangle[1, 2, 3, 4, 5, 6]
    E_LU_RU_RD = E_LU_RU_RD / norm_LU_RU_RD

    return E_LU_RU_LD, E_LD_RU_RD, E_LU_LD_RD, E_LU_RU_RD
end

function evaluate_juraj_dense_csl_energy_bipartite(jsonfile::AbstractString;
        chi_value::Int=chi,
        parameters_value::Dict=parameters,
        ctm_setting_value::LS_CTMRG_settings=ctm_setting)

    data = read_juraj_ipeps_tensor(jsonfile)
    A = juraj_ipeps_array_to_tensormap(data.A_julia)
    A = A / norm(A)

    println("Read Juraj dense iPEPS tensor with bipartite physical spin rotation")
    println("  jsonfile          = " * data.jsonfile)
    println("  python leg order  = " * string(data.python_leg_order))
    println("  julia leg order   = " * string(data.julia_leg_order))
    println("  TensorMap space   = " * string(space(A)))
    println("  BP_rot odd sites  = (:RU, :LD)")
    println("")

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err = CTMRG(A, chi_value, init, [], ctm_setting_value)

    H_Heisenberg, H123chiral, H12, H31, H23 = @ignore_derivatives Hamiltonians(space(A, 1))
    J1 = parameters_value["J1"]
    J2 = parameters_value["J2"]
    Jchi = parameters_value["Jchi"]
    H_triangle = J1 / 4 * (H12 + H31) + J2 / 2 * H23 + Jchi * H123chiral

    E_T1, E_T2, E_T3, E_T4 = evaluate_triangle_bipartite_rotation(
        H_triangle,
        A,
        AA,
        CTM,
        ctm_setting_value,
    )
    E = real(E_T1 + E_T2 + E_T3 + E_T4)

    println("dense CSL iPEPS energy, Juraj bipartite validation")
    println("  chi         = " * string(chi_value))
    println("  J1,J2,Jchi  = " * string((J1, J2, Jchi)))
    println("  E           = " * string(E))
    println("  E_triangle  = " * string(real.([E_T1, E_T2, E_T3, E_T4])))
    println("  ctm_ite_num = " * string(ite_num))
    println("  ctm_ite_err = " * string(ite_err))
    flush(stdout)

    return (
        E=E,
        E_triangle=[E_T1, E_T2, E_T3, E_T4],
        A=A,
        A_odd=rotate_physical_leg(A),
        CTM=CTM,
        AA=AA,
        ctm_ite_num=ite_num,
        ctm_ite_err=ite_err,
        parameters=parameters_value,
        chi=chi_value,
    )
end

result = evaluate_juraj_dense_csl_energy_bipartite(juraj_state_json; chi_value=chi);
