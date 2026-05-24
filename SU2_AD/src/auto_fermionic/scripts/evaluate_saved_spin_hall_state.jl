using LinearAlgebra
using TensorKit
using JLD2
using JSON
using ChainRulesCore
using Zygote
using HDF5
using MAT
using Random
using Dates

try
    using LineSearches
    using OptimKit
catch err
    @warn "LineSearches/OptimKit failed to load. Energy evaluation may still work if the included files do not touch optimizer code." err
end

using Zygote: @ignore_derivatives

@eval Main const $(:×) = TensorKit.$(:×)

@eval Main begin
    function truncdim(howmany::Integer; multiplet_tol=nothing, by=abs, rev::Bool=true)
        return TensorKit.truncrank(howmany; by=by, rev=rev)
    end

    function truncerr(tol::Real; kwargs...)
        return TensorKit.truncerror(tol; kwargs...)
    end

    function tsvd(t; trunc=TensorKit.notrunc(), kwargs...)
        return TensorKit.svd_trunc(t; trunc=trunc, kwargs...)
    end
end

function TensorKit.permute(
    t::TensorKit.AbstractTensorMap,
    codomain_order::Tuple,
    domain_order::Tuple;
    copy::Bool=false,
)
    return TensorKit.permute(t, (codomain_order, domain_order); copy=copy)
end

@eval Main begin
    function eigh(t::TensorKit.AbstractTensorMap)
        return eigen(t)
    end
end

const AUTO_DIR = normpath(joinpath(@__DIR__, ".."))
const SU2_AD_SRC = normpath(joinpath(AUTO_DIR, ".."))
const SU2_AD_ROOT = normpath(joinpath(SU2_AD_SRC, ".."))
const SPINHALL_VAR_DIR = joinpath(SU2_AD_ROOT, "Triangle", "iPESS", "spinHall", "variational")

const DEFAULT_STATE_FILE = joinpath(
    SPINHALL_VAR_DIR,
    "var_iPESS_Z2_Triangle_Hofstadter_Hubbard_spinHall_D4_chi_40_-2.3854.jld2",
)

function include_legacy_spin_hall_code()
    for rel in (
        "bosonic/Settings.jl",
        "bosonic/Settings_cell.jl",
        "bosonic/iPEPS_ansatz.jl",
        "bosonic/AD_lib.jl",
        "bosonic/line_search_lib.jl",
        "bosonic/line_search_lib_cell.jl",
        "bosonic/optimkit_lib.jl",
        "bosonic/CTMRG.jl",
        "fermionic/Fermionic_CTMRG.jl",
        "fermionic/Fermionic_CTMRG_unitcell.jl",
        "fermionic/square_Hubbard_model_cell.jl",
        "fermionic/swap_funs.jl",
        "fermionic/mpo_mps_funs.jl",
        "fermionic/double_layer_funs.jl",
        "fermionic/square_Hubbard_AD_cell.jl",
        "fermionic/triangle_fiPESS_method.jl",
    )
        path = joinpath(SU2_AD_SRC, rel)
        if rel == "bosonic/CTMRG.jl"
            lines = readlines(path)
            filtered = String[]
            for line in lines
                stripped = strip(line)
                if stripped == "verify_truncate_svd()"
                    push!(filtered, "# verify_truncate_svd() disabled by auto_fermionic energy script.")
                else
                    push!(filtered, line)
                end
            end
            text = join(filtered, "\n")
            include_string(Main, text, path)
        else
            include(path)
        end
    end
end

function setup_spin_hall_globals(; chi_in::Int=40, ctm_iters::Int=50, ctm_tol::Float64=1e-6)
    Random.seed!(888)

    global D = 4
    global chi = chi_in

    t = 1
    theta = pi / 2
    mu = 0
    U = 0
    mx = 0
    B = 0
    mx_type = "uniform"

    theta_key = String([Char(0x03B8)])
    mu_key = String([Char(0x03BC)])
    global parameters = Dict(
        "t1" => t,
        "t2" => t,
        theta_key => theta,
        mu_key => mu,
        "U" => U,
        "B" => B,
        "mx" => mx,
        "mx_type" => mx_type,
    )

    global grad_ctm_setting = grad_CTMRG_settings()
    grad_ctm_setting.CTM_conv_tol = ctm_tol
    grad_ctm_setting.CTM_ite_nums = ctm_iters
    grad_ctm_setting.CTM_trun_tol = 1e-8
    grad_ctm_setting.svd_lanczos_tol = 1e-8
    grad_ctm_setting.projector_strategy = "4x4"
    grad_ctm_setting.conv_check = "singular_value"
    grad_ctm_setting.CTM_ite_info = true
    grad_ctm_setting.CTM_conv_info = true
    grad_ctm_setting.CTM_trun_svd = false
    grad_ctm_setting.construct_double_layer = true
    grad_ctm_setting.grad_checkpoint = true

    global LS_ctm_setting = LS_CTMRG_settings()
    LS_ctm_setting.CTM_conv_tol = ctm_tol
    LS_ctm_setting.CTM_ite_nums = ctm_iters
    LS_ctm_setting.CTM_trun_tol = 1e-8
    LS_ctm_setting.svd_lanczos_tol = 1e-8
    LS_ctm_setting.projector_strategy = "4x4"
    LS_ctm_setting.conv_check = "singular_value"
    LS_ctm_setting.CTM_ite_info = false
    LS_ctm_setting.CTM_conv_info = true
    LS_ctm_setting.CTM_trun_svd = false
    LS_ctm_setting.construct_double_layer = true
    LS_ctm_setting.grad_checkpoint = true

    global backward_settings = Backward_settings()
    backward_settings.grad_inverse_tol = 1e-8
    backward_settings.grad_regulation_epsilon = 1e-12
    backward_settings.show_ite_grad_norm = false

    global energy_setting = Triangle_Hofstadter_Hubbard_settings()
    energy_setting.model = "Triangle_Hofstadter_Hubbard_spinHall"
    energy_setting.Lx = 2
    energy_setting.Ly = 2
    energy_setting.Magnetic_cell = 2

    global algrithm_CTMRG_settings = Algrithm_CTMRG_settings()
    algrithm_CTMRG_settings.CTM_cell_ite_method = "continuous_update"

    global multiplet_tol = 1e-5
    global projector_trun_tol = grad_ctm_setting.CTM_trun_tol
    global Lx = energy_setting.Lx
    global Ly = energy_setting.Ly

    return nothing
end

function normalize_loaded_state(x)
    x_norm = deepcopy(x)
    for cc in eachindex(x_norm)
        if isa(x_norm[cc], Triangle_iPESS_immutable)
            tm = x_norm[cc].Tm
            bm = x_norm[cc].Bm
            x_norm[cc] = Triangle_iPESS_immutable(bm / norm(bm), tm / norm(tm))
        elseif isa(x_norm[cc], Triangle_iPESS)
            tm = x_norm[cc].Tm
            bm = x_norm[cc].Bm
            x_norm[cc] = Triangle_iPESS(bm / norm(bm), tm / norm(tm))
        end
    end
    return x_norm
end

function to_mutable_triangle_pess_matrix(x)
    y = Matrix{Triangle_iPESS}(undef, size(x)...)
    for cc in eachindex(x)
        if isa(x[cc], Triangle_iPESS_immutable)
            y[cc] = Triangle_iPESS(x[cc].Bm, x[cc].Tm)
        elseif isa(x[cc], Triangle_iPESS)
            y[cc] = x[cc]
        else
            error("Unsupported saved ansatz type: $(typeof(x[cc]))")
        end
    end
    return y
end

function evaluate_saved_spin_hall_state(;
    state_file::AbstractString=DEFAULT_STATE_FILE,
    chi_in::Int=40,
    ctm_iters::Int=50,
    ctm_tol::Float64=1e-6,
    normalize_state::Bool=false,
)
    println("state_file = ", state_file)
    data = load(state_file)
    println("JLD2 keys = ", collect(keys(data)))

    haskey(data, "x") || error("Expected key `x` in saved variational state.")
    x = data["x"]
    println("loaded x type = ", typeof(x), ", size = ", size(x))

    if normalize_state
        println("normalizing each B/T tensor before evaluation")
        x = normalize_loaded_state(x)
    end

    x_mut = to_mutable_triangle_pess_matrix(x)
    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    result = energy_CTM(x_mut, chi, parameters, LS_ctm_setting, energy_setting, init, [])

    E = result[1]
    println("computed energy = ", E)
    println("ex_up_set = ", result[2])
    println("ey_up_set = ", result[3])
    println("e_diagonala_up_set = ", result[4])
    println("ex_dn_set = ", result[5])
    println("ey_dn_set = ", result[6])
    println("e_diagonala_dn_set = ", result[7])
    println("e0_set = ", result[8])
    println("eU_set = ", result[9])
    println("sx_set = ", result[10])
    println("sy_set = ", result[11])
    println("sz_set = ", result[12])
    println("ite_num = ", result[13])
    println("ite_err = ", result[14])
    println("CTM_cell returned but not printed.")
    return result
end

include_legacy_spin_hall_code()
setup_spin_hall_globals()
const SAVED_SPIN_HALL_RESULT = evaluate_saved_spin_hall_state()
nothing
