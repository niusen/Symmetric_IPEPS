using TensorKit, Zygote
using LinearAlgebra: I, diagm, norm
using JLD2, ChainRulesCore
using KrylovKit
using MAT
using Random
using Zygote: @ignore_derivatives

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "CTMRG_unitcell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_model.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_model_cell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "Settings_cell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "optimkit_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "line_search_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "line_search_lib_cell.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_large_cell_optimization.jl"))

# ES_algorithms.jl provides shared helpers such as vison_op, k_projection,
# calculate_k, and CTM_T_action. The explicit version avoids global U_L/U_R.
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "mps_algorithms", "ES_algorithms.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "mps_algorithms", "ES_algorithms_explicit.jl"))

Random.seed!(555)

default_state = joinpath(@__DIR__, "OptimKit_SU2_cell_2x1_D3_chi_40_-0.9865.jld2")
init_statenm = length(ARGS) >= 1 ? ARGS[1] : default_state
chi = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 40
Nv = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8

EH_n = 30
group_index = true
use_Kprojector = false
vison_values = [false, true]
T_tensor_scale = 10

cell_x = 2
cell_y = 1

LS_ctm_setting = LS_CTMRG_settings()
LS_ctm_setting.CTM_conv_tol = 1e-6
LS_ctm_setting.CTM_ite_nums = 50
LS_ctm_setting.CTM_trun_tol = 1e-8
LS_ctm_setting.svd_lanczos_tol = 1e-8
LS_ctm_setting.projector_strategy = "4x4"
LS_ctm_setting.conv_check = "singular_value"
LS_ctm_setting.CTM_ite_info = false
LS_ctm_setting.CTM_conv_info = true
LS_ctm_setting.CTM_trun_svd = false
LS_ctm_setting.construct_double_layer = true
LS_ctm_setting.grad_checkpoint = true

algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"

global chi, Lx, Ly, multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = LS_ctm_setting.CTM_trun_tol
global algrithm_CTMRG_settings

function _load_square_large_cell_state(statenm::AbstractString)
    data = load(statenm)
    local state

    if haskey(data, "x")
        x = data["x"]
        state = Matrix{Square_iPEPS}(undef, size(x, 1), size(x, 2))
        for cc in eachindex(x)
            state[cc] = Square_iPEPS(_as_square_tensor(x[cc]))
        end
    elseif haskey(data, "A_cell")
        A_cell = data["A_cell"]
        state = Matrix{Square_iPEPS}(undef, length(A_cell), length(A_cell[1]))
        for cx in 1:size(state, 1), cy in 1:size(state, 2)
            state[cx, cy] = Square_iPEPS(_as_square_tensor(A_cell[cx][cy]))
        end
    elseif haskey(data, "A")
        A = data["A"]
        state = Matrix{Square_iPEPS}(undef, 1, 1)
        state[1, 1] = Square_iPEPS(_as_square_tensor(A))
    else
        error("State file must contain `x`, `A_cell`, or `A`.")
    end

    loaded_Lx = haskey(data, "Lx") ? Int(data["Lx"]) : size(state, 1)
    loaded_Ly = haskey(data, "Ly") ? Int(data["Ly"]) : size(state, 2)
    @assert loaded_Lx == size(state, 1)
    @assert loaded_Ly == size(state, 2)
    return state, data
end

function _es_ctm_from_cell_boundary(CTM_cell, cx::Int, cy::Int)
    T2 = CTM_cell.Tset[mod1(cx + 1, Lx)][mod1(cy, Ly)].T2
    T4 = CTM_cell.Tset[mod1(cx, Lx)][mod1(cy, Ly)].T4
    return (Tset=(T2=T2, T4=T4),)
end

function _generic_es_filename(D::Int, chi::Int, Nv::Int, vison::Bool, use_Kprojector::Bool)
    if use_Kprojector
        return vison ? "ES_Kprojector_vison_D$(D)_chi$(chi)_N$(Nv).mat" :
            "ES_Kprojector_D$(D)_chi$(chi)_N$(Nv).mat"
    end
    return vison ? "ES_vison_D$(D)_chi$(chi)_N$(Nv).mat" :
        "ES_D$(D)_chi$(chi)_N$(Nv).mat"
end

function _large_cell_es_filename(statenm::AbstractString, D::Int, chi::Int, Nv::Int,
        vison::Bool, use_Kprojector::Bool)
    tag = splitext(basename(statenm))[1]
    sector = vison ? "vison" : "novison"
    projector = use_Kprojector ? "Kprojector" : "noKprojector"
    return "ES_CTM_$(tag)_cell$(Lx)x$(Ly)_D$(D)_chi$(chi)_Nv$(Nv)_$(sector)_$(projector).mat"
end

function run_SU2_large_cell_ES_CTM(statenm::AbstractString=init_statenm;
        chi_value::Int=chi,
        Nv_value::Int=Nv,
        EH_n_value::Int=EH_n,
        group_index_value::Bool=group_index,
        use_Kprojector_value::Bool=use_Kprojector,
        vison_values_value=vison_values,
        T_tensor_scale_value=T_tensor_scale,
        boundary_cell::Tuple{Int,Int}=(cell_x, cell_y),
        ctm_setting=LS_ctm_setting)

    state, data = _load_square_large_cell_state(statenm)
    global Lx, Ly, chi
    Lx, Ly = size(state)
    chi = chi_value

    A0 = state[1, 1].T
    D = dim(space(A0, 1))
    println("Run SU2 large-cell CTM entanglement spectrum")
    println("  state file     = " * statenm)
    println("  cell           = " * string((Lx, Ly)))
    println("  boundary cell  = " * string(boundary_cell))
    println("  D              = " * string(D))
    println("  chi            = " * string(chi_value))
    println("  Nv             = " * string(Nv_value))
    println("  EH_n           = " * string(EH_n_value))
    println("  group_index    = " * string(group_index_value))
    println("  use_Kprojector = " * string(use_Kprojector_value))
    println("  vison values   = " * string(vison_values_value))
    flush(stdout)

    A_cell = square_cell_to_A_cell(state)
    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM_cell, AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell, ite_num, ite_err =
        square_large_cell_ctmrg(A_cell, chi_value, init, [], ctm_setting)

    converged = _large_cell_ctm_converged(ite_err, ctm_setting)
    println("CTMRG finished: ctm_ite_num= " * string(ite_num) *
            ", ctm_ite_err= " * string(ite_err) *
            ", ctm_converged= " * string(converged))
    flush(stdout)

    cx, cy = boundary_cell
    CTM = _es_ctm_from_cell_boundary(CTM_cell, cx, cy)
    U_L = U_L_cell[cx][cy]
    U_R = U_R_cell[cx][cy]

    saved_files = String[]
    for vison in vison_values_value
        if use_Kprojector_value
            ES_CTMRG_ED_Kprojector_explicit(
                CTM,
                U_L,
                U_R,
                D,
                chi_value,
                Nv_value,
                EH_n_value,
                group_index_value,
                vison;
                T_scale=T_tensor_scale_value,
            )
        else
            ES_CTMRG_ED_explicit(
                CTM,
                U_L,
                U_R,
                D,
                chi_value,
                Nv_value,
                EH_n_value,
                group_index_value,
                vison;
                T_scale=T_tensor_scale_value,
            )
        end

        src = _generic_es_filename(D, chi_value, Nv_value, vison, use_Kprojector_value)
        dst = _large_cell_es_filename(statenm, D, chi_value, Nv_value, vison, use_Kprojector_value)
        if isfile(src)
            mv(src, dst; force=true)
            println("Saved ES to " * dst)
            push!(saved_files, dst)
        else
            println("Expected ES output file was not found: " * src)
        end
        flush(stdout)
    end

    return (
        CTM_cell=CTM_cell,
        AA_cell=AA_cell,
        U_L_cell=U_L_cell,
        U_R_cell=U_R_cell,
        ctm_ite_num=ite_num,
        ctm_ite_err=ite_err,
        saved_files=saved_files,
    )
end

results = run_SU2_large_cell_ES_CTM()
nothing
