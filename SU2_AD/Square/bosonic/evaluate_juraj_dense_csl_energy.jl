using TensorKit
using Zygote
using Zygote: @ignore_derivatives
using ChainRulesCore
using LinearAlgebra: norm
using JLD2
using JSON
using KrylovKit
using Dates

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
ctm_setting.CTM_ite_nums = 150
ctm_setting.CTM_trun_tol = 1e-8
ctm_setting.svd_lanczos_tol = 1e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = true
ctm_setting.CTM_conv_info = true
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true
ctm_setting.grad_checkpoint = false

energy_setting = Square_Energy_settings()
energy_setting.model = "triangle_J1_J2_Jchi"

global multiplet_tol, projector_trun_tol, backward_settings
multiplet_tol = 1e-5
projector_trun_tol = ctm_setting.CTM_trun_tol
backward_settings = Backward_settings()

function evaluate_juraj_dense_csl_energy(jsonfile::AbstractString;
        chi_value::Int=chi,
        parameters_value::Dict=parameters,
        ctm_setting_value::LS_CTMRG_settings=ctm_setting)

    data = read_juraj_ipeps_tensor(jsonfile)
    A = juraj_ipeps_array_to_tensormap(data.A_julia)
    A = A / norm(A)

    println("Read Juraj dense iPEPS tensor")
    println("  jsonfile          = " * data.jsonfile)
    println("  site_id           = " * string(data.site_id))
    println("  python leg order  = " * string(data.python_leg_order))
    println("  python dims       = " * string(data.dims_python))
    println("  julia leg order   = " * string(data.julia_leg_order))
    println("  julia dims        = " * string(data.dims_julia))
    println("  TensorMap space   = " * string(space(A)))
    println("  norm(A) after normalization = " * string(norm(A)))
    println("")

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err = CTMRG(A, chi_value, init, [], ctm_setting_value)

    H_Heisenberg, H123chiral, H12, H31, H23 = @ignore_derivatives Hamiltonians(space(A, 1))
    J1 = parameters_value["J1"]
    J2 = parameters_value["J2"]
    Jchi = parameters_value["Jchi"]
    H_triangle = J1 / 4 * (H12 + H31) + J2 / 2 * H23 + Jchi * H123chiral

    E_T1, E_T2, E_T3, E_T4 = evaluate_triangle(
        H_triangle,
        A,
        AA,
        U_L,
        U_D,
        U_R,
        U_U,
        CTM,
        ctm_setting_value,
    )
    E = real(E_T1 + E_T2 + E_T3 + E_T4)

    println("dense CSL iPEPS energy")
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
        CTM=CTM,
        AA=AA,
        ctm_ite_num=ite_num,
        ctm_ite_err=ite_err,
        parameters=parameters_value,
        chi=chi_value,
    )
end

result = evaluate_juraj_dense_csl_energy(juraj_state_json; chi_value=chi);
