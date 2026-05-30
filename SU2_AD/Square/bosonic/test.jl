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
Base.Sys.set_process_title("C" * string(n_cpu) * "_ES_pair_custom_left")
println("pid=" * string(getpid()))
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm = gethostname()
###########################

init_statenm = "OptimKit_SU2_chiral_pair_D6_chi_54_-1.9594.jld2"
T_tensor_scale = 10

chi_set = [40,]
Nv = 6
EH_n = 30
group_index = true
vison = false

# Set to true if you want momentum sectors projected before eigsolve.
use_Kprojector = false

# Turn this on only when debugging the custom left double layer.  It materializes
# the large six-leg unfused tensor, so it is intentionally off for production ES.
check_custom_left_double_layer = false

# Initial CTM auxiliary space for the custom left environment. This is only
# the starting boundary space; CTMRG projectors will generate/truncate the
# environment during iterations. Increase or customize these sectors if the
# random initial boundary is too restrictive for your tensor.
custom_left_CTM_virtual_space = SU2Space(0 => 2, 1 / 2 => 2, 1 => 2, 3 / 2 => 1, 2 => 1)

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

function build_custom_left_double_layer(A)
    @assert space(A, 5) == chiral_pair_physical_space()' "A must have the fused chiral-pair physical leg."

    A_ = permute(A, (1, 2, 3, 4, 5,))

    U_pair = @ignore_derivatives chiral_pair_fuse_unitary()
    Vcopy_B = @ignore_derivatives space(U_pair, 3)
    IdB = unitary(Vcopy_B, Vcopy_B)

    # Split the fused physical leg first.  The chiral copy is contracted
    # onsite later.  The antichiral identity IdB is an extra tensor-product
    # factor: d1_in contracts the bra antichiral leg, while the ket
    # antichiral leg becomes d1_out and is carried to the next site.

    A_=deepcopy(A);
    @tensor A_[:]:=A_[-1,-2,-3,-4,1]*U_pair[1, -5, -6];#L,D,R,U,d1a,d2b
    @tensor A_[:]:=A_[-1,-2,-4,-5,-3,-8]*IdB[-7,-6];#L, D,d1a, R, U,d1a, d1b, d2b
    @tensor A_[:]:=A_[-1,-2,-3,-4,-5,-6,1,2]*U_pair'[1,2,-7];#L, D,d1a, R, U,d1a, d

    U_combine=unitary(fuse(space(A_,2)*space(A_,3)), (space(A_,2)*space(A_,3)));
    @tensor A_[:]:=A_[-1,1,2,-3,3,4,-5]*U_combine[-2,1,2]*U_combine'[3,4,-4];#L,Dnew,R,Unew




    U_L=@ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A_, 1)), space(A, 1)' ⊗ space(A_, 1))*(1+0*im);
    U_D_ext=@ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A_, 2)), space(A, 2)' ⊗ space(A_, 2))*(1+0*im);
    # U_R=(U_L)';
    # U_U=(U_D)';
    U_R=@ignore_derivatives unitary(space(A, 3) ⊗ space(A_, 3)', fuse(space(A, 3)' ⊗ space(A_, 3)))*(1+0*im);
    U_U_ext=@ignore_derivatives unitary(space(A, 4) ⊗ space(A_, 4)', fuse(space(A, 4)' ⊗ space(A_, 4)))*(1+0*im);



    # println(space(permute(A',(1,2,3,4,5,))))
    # println(space(A_))
    # println(space(U_L))
    # println(space(U_D_ext))
    # println(space(U_R))
    # println(space(U_U_ext))
    @tensor AA_left[-1, -2, -3, -4] := A'[2, 4, 6, 8, 10] *
        A_[3, 5, 7, 9, 10] *
        U_L[-1, 2, 3] *
        U_D_ext[-2, 4, 5] *
        U_R[6, 7, -3] *
        U_U_ext[8, 9, -4]



    return AA_left, U_L, U_D_ext, U_R, U_U_ext
end

function init_CTM_from_double_layer(chi, AA_fused, type, CTM_ite_info)
    @ignore_derivatives if CTM_ite_info
        display("initialize random CTM from custom double layer")
    end
    @assert type == "PBC" "Only PBC CTM initialization is implemented for custom double layers."

    Vctm = custom_left_CTM_virtual_space
    function random_out_tensor(spaces...)
        @assert length(spaces) >= 2
        cod = spaces[1]
        for sp in spaces[2:(end - 1)]
            cod = cod * sp
        end
        T = permute(TensorMap(randn, cod, spaces[end]'), Tuple(1:length(spaces)), ())
        @assert norm(T) > 1e-12 "Random CTM tensor has zero norm; enlarge custom_left_CTM_virtual_space."
        return T / norm(T)
    end

    # Tensor order follows CTM_ite for construct_double_layer=true:
    #   T1 middle leg contracts rotated AA leg 4 for direction 1 -> original up.
    #   T2 middle leg contracts rotated AA leg 4 for direction 2 -> original right.
    #   T3 middle leg contracts rotated AA leg 4 for direction 3 -> original down.
    #   T4 middle leg contracts rotated AA leg 4 for direction 4 -> original left.
    C1 = random_out_tensor(Vctm, Vctm')
    C2 = random_out_tensor(Vctm, Vctm')
    C3 = random_out_tensor(Vctm, Vctm')
    C4 = random_out_tensor(Vctm, Vctm')

    T1 = random_out_tensor(Vctm, space(AA_fused, 4), Vctm')
    T2 = random_out_tensor(Vctm, space(AA_fused, 3), Vctm')
    T3 = random_out_tensor(Vctm, space(AA_fused, 2), Vctm')
    T4 = random_out_tensor(Vctm, space(AA_fused, 1), Vctm')

    @ignore_derivatives if CTM_ite_info
        println("custom random CTM virtual space: " * string(Vctm))
        println("custom random CTM T middle spaces:")
        println("  T1 middle/up    = " * string(space(T1, 2)))
        println("  T2 middle/right = " * string(space(T2, 2)))
        println("  T3 middle/down  = " * string(space(T3, 2)))
        println("  T4 middle/left  = " * string(space(T4, 2)))
        flush(stdout)
    end

    Cset = Cset_struc(C1, C2, C3, C4)
    Tset = Tset_struc(T1, T2, T3, T4)
    return CTM_struc(Cset, Tset)
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
        println("Memory cost of custom left double layer tensor: " * string(AA_memory) * " Mb.")
        flush(stdout)
    end

    if init.reconstruct_CTM
        CTM = init_CTM_from_double_layer(chi, AA_fused, init.init_type, CTM_ite_info)
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

function CTMRG_custom_left(A, chi, init, CTM0, ctm_setting)
    AA_left, U_L, U_D, U_R, U_U = build_custom_left_double_layer(A)
    return CTMRG_from_prebuilt_double_layer(
        A, AA_left, U_L, U_D, U_R, U_U, chi, init, CTM0, ctm_setting,
    )
end

function pair_custom_left_ES_CTM(CTM_left, CTM_right)
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

function _pair_custom_left_ES_filename(init_statenm, D, chi, Nv, vison, use_Kprojector)
    state_tag = splitext(basename(init_statenm))[1]
    sector_tag = vison ? "vison" : "novison"
    projector_tag = use_Kprojector ? "Kprojector" : "noKprojector"
    return "ES_pair_custom_left_$(state_tag)_D$(D)_chi$(chi)_N$(Nv)_$(sector_tag)_$(projector_tag).mat"
end

data = load(init_statenm)
@assert haskey(data, "A") "State file must contain tensor `A`."
A = data["A"]
@assert space(A, 5) == chiral_pair_physical_space()' "Input tensor must be a d=4 chiral-pair tensor."

D = dim(space(A, 1))
println("pair-custom-left ES initial state virtual spaces: " *
        string((space(A, 1), space(A, 2), space(A, 3), space(A, 4))))
println("pair-custom-left ES initial state physical space: " * string(space(A, 5)))
println("pair-custom-left ES T_tensor_scale: " * string(T_tensor_scale))
println("pair-custom-left ES settings: Nv=" * string(Nv) *
        ", EH_n=" * string(EH_n) *
        ", group_index=" * string(group_index) *
        ", vison=" * string(vison) *
        ", Kprojector=" * string(use_Kprojector))
flush(stdout)


    global chi
    chi = 40



    println("\nCompute left custom CTM with chi = " * string(chi))
    flush(stdout)
    init_left = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM_left, AA_left, U_L_left, U_D_left, U_R_left, U_U_left, ite_num_left, ite_err_left =
        CTMRG_custom_left(A, chi, init_left, [], ctm_setting)

    println("left custom CTMRG finished: ctm_ite_num=" * string(ite_num_left) *
            ", ctm_ite_err=" * string(ite_err_left))
    flush(stdout)

    CTM_pair = pair_custom_left_ES_CTM(CTM_left, CTM_right)

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
    target_filenm = _pair_custom_left_ES_filename(init_statenm, D, chi, Nv, vison, use_Kprojector)
    if isfile(default_filenm)
        mv(default_filenm, target_filenm; force=true)
        println("Saved pair-custom-left ES to " * target_filenm)
    else
        println("WARNING: expected ES output file was not found: " * default_filenm)
    end
    flush(stdout)


nothing
