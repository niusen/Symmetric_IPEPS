using Revise, TensorKit, Zygote
using LinearAlgebra: norm, dot,I, norm
import LinearAlgebra.BLAS as BLAS
using JLD2, ChainRulesCore, MAT
using KrylovKit
using Random
using Zygote: @ignore_derivatives

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_chiral_pair_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mps_algorithms", "ES_algorithms.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mps_algorithms", "ES_algorithms_explicit.jl"))

Random.seed!(555)

###########################
n_cpu = 10
BLAS.set_num_threads(n_cpu)
println("number of cpus: " * string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C" * string(n_cpu) * "_chiral_pair_ES")
println("pid=" * string(getpid()))
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm = gethostname()
###########################

init_statenm = "Optim_RVB_coefficients_D3_chi_54_-0.9834.jld2"
T_tensor_scale = 10

chi_set = [40,]
Nv = 6
EH_n = 30
group_index = true
vison = false

# Set to true if you want momentum sectors projected before eigsolve.
use_Kprojector = false

ctm_setting = LS_CTMRG_settings()
ctm_setting.CTM_conv_tol = 1e-6
ctm_setting.CTM_ite_nums = 100
ctm_setting.CTM_trun_tol = 1e-8
ctm_setting.svd_lanczos_tol = 1e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = true
ctm_setting.CTM_conv_info = true
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true
ctm_setting.grad_checkpoint = true
dump(ctm_setting)

global multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = ctm_setting.CTM_trun_tol

data = load(init_statenm)
@assert haskey(data, "A") "State file must contain tensor `A`."
A = data["A"]


@show space(A, 5)  

D = dim(space(A, 1))
println("ES initial state virtual spaces: " *
        string((space(A, 1), space(A, 2), space(A, 3), space(A, 4))))
println("ES initial state physical space: " * string(space(A, 5)))
println("ES T_tensor_scale: " * string(T_tensor_scale))
println("ES settings: Nv=" * string(Nv) *
        ", EH_n=" * string(EH_n) *
        ", group_index=" * string(group_index) *
        ", vison=" * string(vison) *
        ", Kprojector=" * string(use_Kprojector))
flush(stdout)

function _default_ES_filename(D, chi, Nv, vison, use_Kprojector)
    if use_Kprojector
        if vison
            return "ES_Kprojector_vison_D$(D)_chi$(chi)_N$(Nv).mat"
        else
            return "ES_Kprojector_D$(D)_chi$(chi)_N$(Nv).mat"
        end
    else
        if vison
            return "ES_vison_D$(D)_chi$(chi)_N$(Nv).mat"
        else
            return "ES_D$(D)_chi$(chi)_N$(Nv).mat"
        end
    end
end

function _chiral_pair_ES_filename(init_statenm, D, chi, Nv, vison, use_Kprojector)
    state_tag = splitext(basename(init_statenm))[1]
    sector_tag = vison ? "vison" : "novison"
    projector_tag = use_Kprojector ? "Kprojector" : "noKprojector"
    return "ES_chiral_pair_$(state_tag)_D$(D)_chi$(chi)_N$(Nv)_$(sector_tag)_$(projector_tag).mat"
end

for chi_value in chi_set
    global chi
    chi = chi_value

    println("\nCompute chiral-pair CTM entanglement spectrum with chi = " * string(chi))
    flush(stdout)

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM, AA, U_L0, U_D0, U_R0, U_U0, ite_num, ite_err = CTMRG(A, chi, init, [], ctm_setting)

    println("CTMRG finished: ctm_ite_num=" * string(ite_num) *
            ", ctm_ite_err=" * string(ite_err))
    flush(stdout)

    if use_Kprojector
        ES_CTMRG_ED_Kprojector_explicit(
            CTM, U_L0, U_R0, D, chi, Nv, EH_n, group_index, vison;
            T_scale=T_tensor_scale,
        )
    else
        ES_CTMRG_ED_explicit(
            CTM, U_L0, U_R0, D, chi, Nv, EH_n, group_index, vison;
            T_scale=T_tensor_scale,
        )
    end

    default_filenm = _default_ES_filename(D, chi, Nv, vison, use_Kprojector)
    target_filenm = _chiral_pair_ES_filename(init_statenm, D, chi, Nv, vison, use_Kprojector)
    if isfile(default_filenm)
        mv(default_filenm, target_filenm; force=true)
        println("Saved chiral-pair ES to " * target_filenm)
    else
        println("WARNING: expected ES output file was not found: " * default_filenm)
    end
    flush(stdout)
end

nothing
