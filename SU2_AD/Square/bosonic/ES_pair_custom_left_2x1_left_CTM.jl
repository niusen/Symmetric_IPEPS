using TensorKit, Zygote
using LinearAlgebra: I, diagm, norm
import LinearAlgebra.BLAS as BLAS
using JLD2, ChainRulesCore, MAT
using KrylovKit
using Random
using Statistics
using Zygote: @ignore_derivatives

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "Settings_cell.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_chiral_pair_model.jl"))

# ES_algorithms.jl provides shared helpers such as vison_op, k_projection,
# calculate_k, and CTM_T_action. The explicit version avoids global U_L/U_R.
include(joinpath(@__DIR__, "..", "..", "src", "mps_algorithms", "ES_algorithms.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "mps_algorithms", "ES_algorithms_explicit.jl"))

function include_single_function_definition(filename::AbstractString, fname::AbstractString)
    lines = readlines(filename)
    start = findfirst(line -> occursin("function " * fname * "(", line), lines)
    @assert start !== nothing "Cannot find function `$(fname)` in $(filename)."

    buffer = String[]
    depth = 0
    for line in lines[start:end]
        push!(buffer, line)
        code = strip(split(line, "#"; limit=2)[1])
        depth += length(collect(eachmatch(r"\b(function|if|for|while|let|try|begin|quote|macro)\b", code)))
        depth += occursin(r"^\s*mutable\s+struct\b", code) ? 1 : 0
        depth += occursin(r"^\s*struct\b", code) ? 1 : 0
        if occursin(r"^\s*end\s*$", code)
            depth -= 1
            if depth == 0
                break
            end
        end
    end
    @assert depth == 0 "Function `$(fname)` extraction from $(filename) did not close cleanly."
    include_string(@__MODULE__, join(buffer, "\n"), filename)
    return nothing
end

# Use exactly the custom left double-layer constructor from ES_pair_custom_left.jl.
# Do not include that file directly, because it has executable top-level code.
include_single_function_definition(joinpath(@__DIR__, "ES_pair_custom_left.jl"), "build_custom_left_double_layer")

Random.seed!(555)

###########################
n_cpu = 10
BLAS.set_num_threads(n_cpu)
println("number of cpus: " * string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C" * string(n_cpu) * "_ES_pair_custom_left_2x1_left_CTM")
println("pid=" * string(getpid()))
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm = gethostname()
###########################

init_statenm = "OptimKit_SU2_chiral_pair_D6_chi_54_-1.9594.jld2"
T_tensor_scale = 10

chi_set = [40,]
Nv = 4
EH_n = 30
group_index = true
vison = false

# Set to true if you want momentum sectors projected before eigsolve.
use_Kprojector = false

# The left custom CTMRG is run on a 2x1 unit cell.  A single-site saved state is
# copied to both x positions; saved cells are read periodically onto 2x1.
left_cell_Lx = 2
left_cell_Ly = 1
left_boundary_cell = (1, 1)

# Random CTM auxiliary space used only to initialize the custom-left 2x1 CTMRG.
# This is deliberately local to this file, so it is easy to tune without touching
# ES_pair_custom_left.jl.
custom_left_cell_CTM_virtual_space = SU2Space(
    0 => 4, 1 / 2 => 4, 1 => 4, 3 / 2 => 2, 2 => 2,
)

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

algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"

global multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = ctm_setting.CTM_trun_tol
global algrithm_CTMRG_settings

function _as_chiral_pair_tensor(x)
    if isa(x, Square_iPEPS) || isa(x, Square_iPEPS_immutable)
        return x.T
    end
    return x
end

function _tuple_cell_size(A_cell)
    return (length(A_cell), length(A_cell[1]))
end

function _loaded_chiral_pair_tensor_for_cell(data, cx::Int, cy::Int)
    if haskey(data, "x")
        x = data["x"]
        sx, sy = size(x)
        return _as_chiral_pair_tensor(x[mod1(cx, sx), mod1(cy, sy)])
    elseif haskey(data, "A_cell")
        A_cell = data["A_cell"]
        sx, sy = _tuple_cell_size(A_cell)
        return _as_chiral_pair_tensor(A_cell[mod1(cx, sx)][mod1(cy, sy)])
    elseif haskey(data, "A")
        A = data["A"]
        if isa(A, AbstractArray) && !(A isa TensorMap)
            sx, sy = size(A)
            return _as_chiral_pair_tensor(A[mod1(cx, sx), mod1(cy, sy)])
        end
        return _as_chiral_pair_tensor(A)
    end
    error("State file must contain `A`, `A_cell`, or `x`.")
end

function load_chiral_pair_A_cell_2x1(statenm::AbstractString)
    data = load(statenm)
    global Lx, Ly
    Lx = left_cell_Lx
    Ly = left_cell_Ly

    A_cell = initial_tuple_cell(Lx, Ly)
    for cx in 1:Lx, cy in 1:Ly
        A = _loaded_chiral_pair_tensor_for_cell(data, cx, cy)
        @assert space(A, 5) == chiral_pair_physical_space()' "Input tensor must be a d=4 chiral-pair tensor."
        A_cell = fill_tuple(A_cell, A / norm(A), cx, cy)
    end
    return A_cell, data
end

function build_custom_left_double_layer_cell(A_cell)
    AA_cell = initial_tuple_cell(Lx, Ly)
    U_L_cell = initial_tuple_cell(Lx, Ly)
    U_D_cell = initial_tuple_cell(Lx, Ly)
    U_R_cell = initial_tuple_cell(Lx, Ly)
    U_U_cell = initial_tuple_cell(Lx, Ly)

    for cx in 1:Lx, cy in 1:Ly
        AA, U_L, U_D, U_R, U_U = build_custom_left_double_layer(A_cell[cx][cy])
        AA_cell = fill_tuple(AA_cell, AA, cx, cy)
        U_L_cell = fill_tuple(U_L_cell, U_L, cx, cy)
        U_D_cell = fill_tuple(U_D_cell, U_D, cx, cy)
        U_R_cell = fill_tuple(U_R_cell, U_R, cx, cy)
        U_U_cell = fill_tuple(U_U_cell, U_U, cx, cy)
    end
    return AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell
end

function init_CTM_from_custom_double_layer(AA_fused, type, CTM_ite_info)
    @ignore_derivatives if CTM_ite_info
        display("initialize random CTM from custom double layer")
    end
    @assert type == "PBC" "Only PBC CTM initialization is implemented for custom double layers."

    Vctm = custom_left_cell_CTM_virtual_space
    function random_out_tensor(spaces...)
        @assert length(spaces) >= 2
        cod = spaces[1]
        for sp in spaces[2:(end - 1)]
            cod = cod * sp
        end
        T = permute(TensorMap(randn, cod, spaces[end]'), Tuple(1:length(spaces)), ())
        @assert norm(T) > 1e-12 "Random CTM tensor has zero norm; enlarge custom_left_cell_CTM_virtual_space."
        return T / norm(T)
    end

    C1 = random_out_tensor(Vctm, Vctm')
    C2 = random_out_tensor(Vctm, Vctm')
    C3 = random_out_tensor(Vctm, Vctm')
    C4 = random_out_tensor(Vctm, Vctm')

    T1 = random_out_tensor(Vctm, space(AA_fused, 4)', Vctm')
    T2 = random_out_tensor(Vctm, space(AA_fused, 3)', Vctm')
    T3 = random_out_tensor(Vctm, space(AA_fused, 2)', Vctm')
    T4 = random_out_tensor(Vctm, space(AA_fused, 1)', Vctm')

    @ignore_derivatives if CTM_ite_info
        println("custom-left 2x1 random CTM virtual space: " * string(Vctm))
        println("custom-left 2x1 random CTM T middle spaces:")
        println("  T1 middle/up    = " * string(space(T1, 2)))
        println("  T2 middle/right = " * string(space(T2, 2)))
        println("  T3 middle/down  = " * string(space(T3, 2)))
        println("  T4 middle/left  = " * string(space(T4, 2)))
        flush(stdout)
    end

    return CTM_struc(Cset_struc(C1, C2, C3, C4), Tset_struc(T1, T2, T3, T4))
end

function init_CTM_cell_from_custom_double_layer(AA_cell, type, CTM_ite_info)
    Cset_cell = initial_tuple_cell(Lx, Ly)
    Tset_cell = initial_tuple_cell(Lx, Ly)
    for cx in 1:Lx, cy in 1:Ly
        CTM = init_CTM_from_custom_double_layer(AA_cell[cx][cy], type, CTM_ite_info)
        Cset_cell = fill_tuple(Cset_cell, CTM.Cset, cx, cy)
        Tset_cell = fill_tuple(Tset_cell, CTM.Tset, cx, cy)
    end
    return (Cset=Cset_cell, Tset=Tset_cell)
end

function CTMRG_cell_from_prebuilt_double_layer(
        AA_fused_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell,
        chi_value, init, CTM0, ctm_setting)
    global Lx, Ly
    global algrithm_CTMRG_settings

    CTM_trun_tol = ctm_setting.CTM_trun_tol
    CTM_ite_info = ctm_setting.CTM_ite_info
    CTM_conv_info = ctm_setting.CTM_conv_info
    projector_strategy = ctm_setting.projector_strategy
    CTM_trun_svd = ctm_setting.CTM_trun_svd
    svd_lanczos_tol = ctm_setting.svd_lanczos_tol
    CTM_ite_nums = ctm_setting.CTM_ite_nums
    construct_double_layer = true

    if (CTM_trun_svd == true) && (projector_strategy == "4x4")
        println("Attention: truncated svd with 4x4 projector could give large error")
    end

    AA_memory = @ignore_derivatives Base.summarysize(AA_fused_cell) / 1024 / 1024
    @ignore_derivatives if CTM_ite_info
        println("Memory cost of custom-left 2x1 double layer cell: " * string(AA_memory) * " Mb.")
        flush(stdout)
    end

    if init.reconstruct_CTM
        CTM_cell = init_CTM_cell_from_custom_double_layer(
            AA_fused_cell, init.init_type, CTM_ite_info,
        )
    else
        CTM_cell = deepcopy(CTM0)
    end

    ss_old1_cell = Matrix(undef, Lx, Ly)
    ss_old2_cell = Matrix(undef, Lx, Ly)
    ss_old3_cell = Matrix(undef, Lx, Ly)
    ss_old4_cell = Matrix(undef, Lx, Ly)
    ss_new1_cell = Matrix(undef, Lx, Ly)
    ss_new2_cell = Matrix(undef, Lx, Ly)
    ss_new3_cell = Matrix(undef, Lx, Ly)
    ss_new4_cell = Matrix(undef, Lx, Ly)
    er1_cell = Matrix(undef, Lx, Ly)
    er2_cell = Matrix(undef, Lx, Ly)
    er3_cell = Matrix(undef, Lx, Ly)
    er4_cell = Matrix(undef, Lx, Ly)

    Cset_cell = CTM_cell.Cset
    Tset_cell = CTM_cell.Tset
    conv_check = "singular_value"

    @ignore_derivatives for cx in 1:Lx, cy in 1:Ly
        ss_old1_cell[cx, cy] = ones(chi_value) * 2
        ss_old2_cell[cx, cy] = ones(chi_value) * 2
        ss_old3_cell[cx, cy] = ones(chi_value) * 2
        ss_old4_cell[cx, cy] = ones(chi_value) * 2
    end

    AA_rotated_cell = rotate_AA_cell(AA_fused_cell, construct_double_layer)

    @ignore_derivatives if CTM_ite_info
        println("start custom-left 2x1 CTM iterations:")
    end

    if algrithm_CTMRG_settings.CTM_cell_ite_method == "together_update"
        CTM_ite_cell = CTM_ite_cell_together_update
    elseif algrithm_CTMRG_settings.CTM_cell_ite_method == "continuous_update"
        CTM_ite_cell = CTM_ite_cell_continuous_update
    else
        error("Unknown CTM_cell_ite_method: " * string(algrithm_CTMRG_settings.CTM_cell_ite_method))
    end

    ite_num = 0
    ite_err = 1
    err_set = [1.0]

    for ci in 1:CTM_ite_nums
        ite_num = ci
        direction_order = [3, 4, 1, 2]
        for direction in direction_order
            if ctm_setting.grad_checkpoint
                Cset_cell, Tset_cell = Zygote.checkpointed(
                    CTM_ite_cell, Cset_cell, Tset_cell, get_Tset(AA_rotated_cell, direction),
                    chi_value, direction, CTM_trun_tol, CTM_ite_info, projector_strategy,
                    CTM_trun_svd, svd_lanczos_tol, construct_double_layer,
                )
            else
                Cset_cell, Tset_cell = CTM_ite_cell(
                    Cset_cell, Tset_cell, get_Tset(AA_rotated_cell, direction),
                    chi_value, direction, CTM_trun_tol, CTM_ite_info, projector_strategy,
                    CTM_trun_svd, svd_lanczos_tol, construct_double_layer,
                )
            end
        end

        if conv_check == "singular_value"
            for cx in 1:Lx, cy in 1:Ly
                er1, ss_new1 = @ignore_derivatives spectrum_conv_check(ss_old1_cell[cx, cy], Cset_cell[cx][cy].C1)
                er2, ss_new2 = @ignore_derivatives spectrum_conv_check(ss_old2_cell[cx, cy], Cset_cell[cx][cy].C2)
                er3, ss_new3 = @ignore_derivatives spectrum_conv_check(ss_old3_cell[cx, cy], Cset_cell[cx][cy].C3)
                er4, ss_new4 = @ignore_derivatives spectrum_conv_check(ss_old4_cell[cx, cy], Cset_cell[cx][cy].C4)

                @ignore_derivatives er1_cell[cx, cy] = er1
                @ignore_derivatives er2_cell[cx, cy] = er2
                @ignore_derivatives er3_cell[cx, cy] = er3
                @ignore_derivatives er4_cell[cx, cy] = er4
                @ignore_derivatives ss_new1_cell[cx, cy] = ss_new1
                @ignore_derivatives ss_new2_cell[cx, cy] = ss_new2
                @ignore_derivatives ss_new3_cell[cx, cy] = ss_new3
                @ignore_derivatives ss_new4_cell[cx, cy] = ss_new4
            end

            er = @ignore_derivatives maximum([
                maximum(er1_cell[:]), maximum(er2_cell[:]),
                maximum(er3_cell[:]), maximum(er4_cell[:]),
            ])
            ite_err = er
            push!(err_set, er)

            @ignore_derivatives if CTM_ite_info
                println("custom-left 2x1 CTMRG iteration: " * string(ci) *
                    ", CTMRG err: " * string(er))
                flush(stdout)
            end
            if er < ctm_setting.CTM_conv_tol
                break
            end
            if ci > 30
                err_recent = err_set[(end - 10):end]
                if (std(err_recent) / mean(err_recent) < 0.001) && (er > 1e-4)
                    break
                end
            end

            ss_old1_cell = ss_new1_cell
            ss_old2_cell = ss_new2_cell
            ss_old3_cell = ss_new3_cell
            ss_old4_cell = ss_new4_cell
        end
    end

    CTM_cell = (Cset=Cset_cell, Tset=Tset_cell)
    if CTM_conv_info
        return CTM_cell, AA_fused_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell, ite_num, ite_err
    end
    return CTM_cell, AA_fused_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell
end

function CTMRG_custom_left_2x1_cell(A_cell, chi_value, init, CTM0, ctm_setting)
    AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell = build_custom_left_double_layer_cell(A_cell)
    return CTMRG_cell_from_prebuilt_double_layer(
        AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell,
        chi_value, init, CTM0, ctm_setting,
    )
end

function _default_ES_filename(D, chi_value, Nv_value, vison_value, use_Kprojector_value)
    if use_Kprojector_value
        return vison_value ? "ES_Kprojector_vison_D$(D)_chi$(chi_value)_N$(Nv_value).mat" :
            "ES_Kprojector_D$(D)_chi$(chi_value)_N$(Nv_value).mat"
    end
    return vison_value ? "ES_vison_D$(D)_chi$(chi_value)_N$(Nv_value).mat" :
        "ES_D$(D)_chi$(chi_value)_N$(Nv_value).mat"
end

function _pair_custom_left_2x1_ES_filename(init_statenm, D, chi_value, Nv_value,
        vison_value, use_Kprojector_value, boundary_cell)
    state_tag = splitext(basename(init_statenm))[1]
    sector_tag = vison_value ? "vison" : "novison"
    projector_tag = use_Kprojector_value ? "Kprojector" : "noKprojector"
    cx, cy = boundary_cell
    return "ES_pair_custom_left_2x1left_$(state_tag)_D$(D)_chi$(chi_value)_N$(Nv_value)_cell$(cx)x$(cy)_$(sector_tag)_$(projector_tag).mat"
end

function run_pair_custom_left_2x1_left_CTM(statenm::AbstractString=init_statenm;
        chi_values=chi_set,
        Nv_value::Int=Nv,
        EH_n_value::Int=EH_n,
        group_index_value::Bool=group_index,
        use_Kprojector_value::Bool=use_Kprojector,
        vison_value::Bool=vison,
        T_tensor_scale_value=T_tensor_scale,
        boundary_cell::Tuple{Int,Int}=left_boundary_cell,
        ctm_setting_value=ctm_setting)

    A_cell, data = load_chiral_pair_A_cell_2x1(statenm)
    A0 = A_cell[1][1]
    D = dim(space(A0, 1))

    println("pair-custom-left ES with 2x1 left CTMRG")
    println("  state file       = " * statenm)
    println("  left CTM cell    = " * string((Lx, Ly)))
    println("  boundary cell    = " * string(boundary_cell))
    println("  D                = " * string(D))
    println("  Nv               = " * string(Nv_value))
    println("  EH_n             = " * string(EH_n_value))
    println("  group_index      = " * string(group_index_value))
    println("  use_Kprojector   = " * string(use_Kprojector_value))
    println("  vison            = " * string(vison_value))
    println("  T_tensor_scale   = " * string(T_tensor_scale_value))
    println("  initial virtual spaces at (1,1): " *
        string((space(A0, 1), space(A0, 2), space(A0, 3), space(A0, 4))))
    println("  initial physical space at (1,1): " * string(space(A0, 5)))
    flush(stdout)

    cx, cy = boundary_cell
    @assert 1 <= cx <= Lx && 1 <= cy <= Ly "boundary_cell must be inside the 2x1 left cell."

    saved_files = String[]
    for chi_value in chi_values
        global chi
        chi = chi_value

        println("\nCompute right normal 1x1 CTM with chi = " * string(chi_value))
        flush(stdout)
        init_right = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
        CTM_right, AA_right, U_L_right, U_D_right, U_R_right, U_U_right, ite_num_right, ite_err_right =
            CTMRG(A0, chi_value, init_right, [], ctm_setting_value)

        println("right normal CTMRG finished: ctm_ite_num=" * string(ite_num_right) *
            ", ctm_ite_err=" * string(ite_err_right))
        flush(stdout)

        println("\nCompute left custom 2x1 CTM with chi = " * string(chi_value))
        flush(stdout)
        init_left = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
        CTM_left_cell, AA_left_cell, U_L_left_cell, U_D_left_cell, U_R_left_cell, U_U_left_cell,
            ite_num_left, ite_err_left =
            CTMRG_custom_left_2x1_cell(A_cell, chi_value, init_left, [], ctm_setting_value)

        println("left custom 2x1 CTMRG finished: ctm_ite_num=" * string(ite_num_left) *
            ", ctm_ite_err=" * string(ite_err_left))
        flush(stdout)

        CTM_pair = (Tset=(T4=CTM_left_cell.Tset[cx][cy].T4, T2=CTM_right.Tset.T2),)
        U_L = U_L_left_cell[cx][cy]
        U_R = U_R_right
        if use_Kprojector_value
            ES_CTMRG_ED_Kprojector_explicit(
                CTM_pair, U_L, U_R, D, chi_value, Nv_value, EH_n_value,
                group_index_value, vison_value; T_scale=T_tensor_scale_value,
            )
        else
            ES_CTMRG_ED_explicit(
                CTM_pair, U_L, U_R, D, chi_value, Nv_value, EH_n_value,
                group_index_value, vison_value; T_scale=T_tensor_scale_value,
            )
        end

        default_filenm = _default_ES_filename(D, chi_value, Nv_value, vison_value, use_Kprojector_value)
        target_filenm = _pair_custom_left_2x1_ES_filename(
            statenm, D, chi_value, Nv_value, vison_value, use_Kprojector_value, boundary_cell,
        )
        if isfile(default_filenm)
            mv(default_filenm, target_filenm; force=true)
            println("Saved pair-custom-left 2x1-left ES to " * target_filenm)
            push!(saved_files, target_filenm)
        else
            println("WARNING: expected ES output file was not found: " * default_filenm)
        end
        flush(stdout)
    end

    return saved_files
end

results = run_pair_custom_left_2x1_left_CTM()
nothing
