using TensorKit
using Zygote
using Zygote: @ignore_derivatives
using ChainRulesCore
using LinearAlgebra: norm, dot, diag, I, eigen
import LinearAlgebra.BLAS as BLAS
using JLD2
using JSON
using MAT
using KrylovKit
using Random

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_AD_SU2.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "read_juraj_ipeps_tensor.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mps_algorithms", "ES_algorithms_dense.jl"))

Random.seed!(555)

n_cpu = 10
BLAS.set_num_threads(n_cpu)
println("number of cpus: " * string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C" * string(n_cpu) * "_dense_ES")
println("pid=" * string(getpid()))
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm = gethostname()

juraj_state_json = raw"D:\My Documents\Code\python_codes\Juraj\tn-torch_dev\data_c4_pt_csl\j1j2lambda_c4pt_D3_chi40_seed0_gpu_state_-0.9865"

chi_set = [40]
Nv = 4
EH_n = 50
use_Kprojector = false
T_tensor_scale = 10

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
dump(ctm_setting)

global multiplet_tol, projector_trun_tol, backward_settings
multiplet_tol = 1e-5
projector_trun_tol = ctm_setting.CTM_trun_tol
backward_settings = Backward_settings()

function _dense_ES_filename(jsonfile::AbstractString, D::Int, chi::Int, Nv::Int, use_Kprojector::Bool)
    state_tag = splitext(basename(jsonfile))[1]
    projector_tag = use_Kprojector ? "Kprojector" : "noKprojector"
    return "ES_dense_juraj_$(state_tag)_D$(D)_chi$(chi)_N$(Nv)_$(projector_tag).mat"
end

function run_juraj_dense_ES(jsonfile::AbstractString;
        chi_values=chi_set,
        Nv_value::Int=Nv,
        EH_n_value::Int=EH_n,
        use_Kprojector_value::Bool=use_Kprojector,
        T_tensor_scale_value=T_tensor_scale,
        ctm_setting_value::LS_CTMRG_settings=ctm_setting)

    data = read_juraj_ipeps_tensor(jsonfile)
    A = juraj_ipeps_array_to_tensormap(data.A_julia)
    A = A / norm(A)
    D = dim(space(A, 1))

    println("Read Juraj dense iPEPS tensor for CTM entanglement spectrum")
    println("  jsonfile          = " * data.jsonfile)
    println("  python leg order  = " * string(data.python_leg_order))
    println("  julia leg order   = " * string(data.julia_leg_order))
    println("  TensorMap space   = " * string(space(A)))
    println("  virtual spaces    = " * string((space(A, 1), space(A, 2), space(A, 3), space(A, 4))))
    println("  physical space    = " * string(space(A, 5)))
    println("  Nv                = " * string(Nv_value))
    println("  EH_n              = " * string(EH_n_value))
    println("  Kprojector        = " * string(use_Kprojector_value))
    println("  T_tensor_scale    = " * string(T_tensor_scale_value))
    println("  note              = no odd-site spin rotation is applied for closed double layer ES")
    flush(stdout)

    results = Dict{Int,Any}()
    for chi_value in chi_values
        global chi
        chi = chi_value

        println("\nCompute dense CTM entanglement spectrum with chi = " * string(chi))
        flush(stdout)

        init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
        CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err = CTMRG(A, chi, init, [], ctm_setting_value)

        println("CTMRG finished: ctm_ite_num=" * string(ite_num) *
            ", ctm_ite_err=" * string(ite_err))
        flush(stdout)

        save_filenm = _dense_ES_filename(data.jsonfile, D, chi, Nv_value, use_Kprojector_value)
        if use_Kprojector_value
            result = ES_CTMRG_ED_Kprojector_dense(
                CTM, U_L, U_R, D, chi, Nv_value, EH_n_value;
                save_filenm=save_filenm,
                T_scale=T_tensor_scale_value,
            )
        else
            result = ES_CTMRG_ED_dense(
                CTM, U_L, U_R, D, chi, Nv_value, EH_n_value;
                save_filenm=save_filenm,
                T_scale=T_tensor_scale_value,
            )
        end

        results[chi] = (
            result=result,
            ctm_ite_num=ite_num,
            ctm_ite_err=ite_err,
            save_filenm=save_filenm,
        )
        println("Saved dense ES to " * save_filenm)
        flush(stdout)
    end
    return results
end

results = run_juraj_dense_ES(juraj_state_json)

nothing
