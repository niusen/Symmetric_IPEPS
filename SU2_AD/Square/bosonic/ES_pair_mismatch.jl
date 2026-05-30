using TensorKit, Zygote
using LinearAlgebra: I, diagm, norm
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

# ES_algorithms.jl provides shared helpers such as vison_op, k_projection,
# calculate_k, and CTM_T_action. The explicit version avoids global U_L/U_R.
include(joinpath(@__DIR__, "..", "..", "src", "mps_algorithms", "ES_algorithms.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mps_algorithms", "ES_algorithms_explicit.jl"))

Random.seed!(555)

###########################
n_cpu = 10
BLAS.set_num_threads(n_cpu)
println("number of cpus: " * string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C" * string(n_cpu) * "_ES_pair_mismatch")
println("pid=" * string(getpid()))
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm = gethostname()
###########################

init_statenm = "OptimKit_SU2_chiral_pair_D9_chi_54.jld2"
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

const CHIRAL_PAIR_COPY_SWAP_CACHE = Ref{Any}(nothing)

function chiral_pair_copy_swap_operator()
    if CHIRAL_PAIR_COPY_SWAP_CACHE[] === nothing
        U_pair = @ignore_derivatives chiral_pair_fuse_unitary()

        # In copy basis this is |a,b><b,a|. In the fused d=4 physical space it
        # implements the requested chiral/antichiral mismatch contraction.
        @tensor swap_pair[:] := U_pair[-1, 2, 1] * U_pair'[1, 2, -2]
        CHIRAL_PAIR_COPY_SWAP_CACHE[] = swap_pair
    end
    return CHIRAL_PAIR_COPY_SWAP_CACHE[]
end

function CTMRG_from_prebuilt_double_layer(A, AA_fused, U_L, U_D, U_R, U_U,
        chi, init, CTM0, ctm_setting)
    CTM_trun_tol = ctm_setting.CTM_trun_tol
    CTM_ite_info = ctm_setting.CTM_ite_info
    CTM_conv_info = ctm_setting.CTM_conv_info
    projector_strategy = ctm_setting.projector_strategy
    CTM_trun_svd = ctm_setting.CTM_trun_svd
    svd_lanczos_tol = ctm_setting.svd_lanczos_tol
    CTM_ite_nums = ctm_setting.CTM_ite_nums
    construct_double_layer = true

    AA_memory = @ignore_derivatives Base.summarysize(AA_fused) / 1024 / 1024
    @ignore_derivatives if CTM_ite_info
        println("Memory cost of prebuilt double layer tensor: " * string(AA_memory) * " Mb.")
        flush(stdout)
    end

    if init.reconstruct_CTM
        CTM = init_CTM(chi, A, init.init_type, CTM_ite_info)
    else
        CTM = deepcopy(CTM0)
    end

    Cset = CTM.Cset
    Tset = CTM.Tset
    ctm_setting.conv_check = "singular_value"

    ss_old1 = ones(chi) * 2
    ss_old2 = ones(chi) * 2
    ss_old3 = ones(chi) * 2
    ss_old4 = ones(chi) * 2

    AA_rotated = rotate_AA(AA_fused, construct_double_layer)

    @ignore_derivatives if CTM_ite_info
        println("start CTM iterations:")
    end
    ite_num = 0
    ite_err = 1
    for ci in 1:CTM_ite_nums
        ite_num = ci
        direction_order = [3, 4, 1, 2]
        for direction in direction_order
            if ctm_setting.grad_checkpoint
                Cset, Tset = Zygote.checkpointed(
                    CTM_ite, Cset, Tset, get_Tset(AA_rotated, direction), chi,
                    direction, CTM_trun_tol, CTM_ite_info, projector_strategy,
                    CTM_trun_svd, svd_lanczos_tol, construct_double_layer,
                )
            else
                Cset, Tset = CTM_ite(
                    Cset, Tset, get_Tset(AA_rotated, direction), chi,
                    direction, CTM_trun_tol, CTM_ite_info, projector_strategy,
                    CTM_trun_svd, svd_lanczos_tol, construct_double_layer,
                )
            end
        end

        if ctm_setting.conv_check == "singular_value"
            er1, ss_new1 = @ignore_derivatives spectrum_conv_check(ss_old1, Cset.C1)
            er2, ss_new2 = @ignore_derivatives spectrum_conv_check(ss_old2, Cset.C2)
            er3, ss_new3 = @ignore_derivatives spectrum_conv_check(ss_old3, Cset.C3)
            er4, ss_new4 = @ignore_derivatives spectrum_conv_check(ss_old4, Cset.C4)

            er = @ignore_derivatives max(er1, er2, er3, er4)
            ite_err = er
            @ignore_derivatives if CTM_ite_info
                println("CTMRG iteration: " * string(ci) * ", CTMRG err: " * string(er))
                flush(stdout)
            end
            if er < ctm_setting.CTM_conv_tol
                break
            end
            ss_old1 = ss_new1
            ss_old2 = ss_new2
            ss_old3 = ss_new3
            ss_old4 = ss_new4
        end
    end

    CTM = CTM_struc(Cset, Tset)
    if CTM_conv_info
        return CTM, AA_fused, U_L, U_D, U_R, U_U, ite_num, ite_err
    end
    return CTM, AA_fused, U_L, U_D, U_R, U_U
end

function CTMRG_chiral_pair_mismatch_left(A, chi, init, CTM0, ctm_setting)
    swap_pair = chiral_pair_copy_swap_operator()
    AA_mismatch, U_L, U_D, U_R, U_U = build_double_layer(A, swap_pair)
    return CTMRG_from_prebuilt_double_layer(
        A, AA_mismatch, U_L, U_D, U_R, U_U, chi, init, CTM0, ctm_setting,
    )
end

function pair_mismatch_ES_CTM(CTM_left, CTM_right)
    return (Tset=(T4=CTM_left.Tset.T4, T2=CTM_right.Tset.T2),)
end

function _default_ES_filename(D, chi, Nv, vison, use_Kprojector)
    if use_Kprojector
        return vison ? "ES_Kprojector_vison_D$(D)_chi$(chi)_N$(Nv).mat" :
            "ES_Kprojector_D$(D)_chi$(chi)_N$(Nv).mat"
    end
    return vison ? "ES_vison_D$(D)_chi$(chi)_N$(Nv).mat" :
        "ES_D$(D)_chi$(chi)_N$(Nv).mat"
end

function _pair_mismatch_ES_filename(init_statenm, D, chi, Nv, vison, use_Kprojector)
    state_tag = splitext(basename(init_statenm))[1]
    sector_tag = vison ? "vison" : "novison"
    projector_tag = use_Kprojector ? "Kprojector" : "noKprojector"
    return "ES_pair_mismatch_$(state_tag)_D$(D)_chi$(chi)_N$(Nv)_$(sector_tag)_$(projector_tag).mat"
end

data = load(init_statenm)
@assert haskey(data, "A") "State file must contain tensor `A`."
A = data["A"]
@assert space(A, 5) == chiral_pair_physical_space()' "Input tensor must be a d=4 chiral-pair tensor."

D = dim(space(A, 1))
println("pair-mismatch ES initial state virtual spaces: " *
        string((space(A, 1), space(A, 2), space(A, 3), space(A, 4))))
println("pair-mismatch ES initial state physical space: " * string(space(A, 5)))
println("pair-mismatch ES T_tensor_scale: " * string(T_tensor_scale))
println("pair-mismatch ES settings: Nv=" * string(Nv) *
        ", EH_n=" * string(EH_n) *
        ", group_index=" * string(group_index) *
        ", vison=" * string(vison) *
        ", Kprojector=" * string(use_Kprojector))
flush(stdout)

for chi_value in chi_set
    global chi
    chi = chi_value

    println("\nCompute right normal CTM with chi = " * string(chi))
    flush(stdout)
    init_right = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM_right, AA_right, U_L_right, U_D_right, U_R_right, U_U_right, ite_num_right, ite_err_right =
        CTMRG(A, chi, init_right, [], ctm_setting)

    println("right CTMRG finished: ctm_ite_num=" * string(ite_num_right) *
            ", ctm_ite_err=" * string(ite_err_right))
    flush(stdout)

    println("\nCompute left mismatch CTM with chi = " * string(chi))
    flush(stdout)
    init_left = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM_left, AA_left, U_L_left, U_D_left, U_R_left, U_U_left, ite_num_left, ite_err_left =
        CTMRG_chiral_pair_mismatch_left(A, chi, init_left, [], ctm_setting)

    println("left mismatch CTMRG finished: ctm_ite_num=" * string(ite_num_left) *
            ", ctm_ite_err=" * string(ite_err_left))
    flush(stdout)

    CTM_pair = pair_mismatch_ES_CTM(CTM_left, CTM_right)

    if use_Kprojector
        ES_CTMRG_ED_Kprojector_explicit(
            CTM_pair, U_L_left, U_R_right, D, chi, Nv, EH_n, group_index, vison;
            T_scale=T_tensor_scale,
        )
    else
        ES_CTMRG_ED_explicit(
            CTM_pair, U_L_left, U_R_right, D, chi, Nv, EH_n, group_index, vison;
            T_scale=T_tensor_scale,
        )
    end

    default_filenm = _default_ES_filename(D, chi, Nv, vison, use_Kprojector)
    target_filenm = _pair_mismatch_ES_filename(init_statenm, D, chi, Nv, vison, use_Kprojector)
    if isfile(default_filenm)
        mv(default_filenm, target_filenm; force=true)
        println("Saved pair-mismatch ES to " * target_filenm)
    else
        println("WARNING: expected ES output file was not found: " * default_filenm)
    end
    flush(stdout)
end

nothing
