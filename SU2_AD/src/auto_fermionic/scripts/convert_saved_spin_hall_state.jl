using LinearAlgebra
using TensorKit
using JLD2
using AutoFermionicPESS

const AUTO_DIR = normpath(joinpath(@__DIR__, ".."))
const SU2_AD_SRC = normpath(joinpath(AUTO_DIR, ".."))
const SU2_AD_ROOT = normpath(joinpath(SU2_AD_SRC, ".."))
const SPINHALL_VAR_DIR = joinpath(SU2_AD_ROOT, "Triangle", "iPESS", "spinHall", "variational")

const DEFAULT_STATE_FILE = joinpath(
    SPINHALL_VAR_DIR,
    "var_iPESS_Z2_Triangle_Hofstadter_Hubbard_spinHall_D4_chi_40_-2.3854.jld2",
)

function TensorKit.permute(
    t::TensorKit.AbstractTensorMap,
    codomain_order::Tuple,
    domain_order::Tuple;
    copy::Bool=false,
)
    return TensorKit.permute(t, (codomain_order, domain_order); copy=copy)
end

function include_legacy_ansatz_code()
    include(joinpath(SU2_AD_SRC, "bosonic", "iPEPS_ansatz.jl"))
end

function convert_saved_spin_hall_state(; state_file::AbstractString=DEFAULT_STATE_FILE)
    include_legacy_ansatz_code()
    data = load(state_file)
    haskey(data, "x") || error("Expected key `x` in saved variational state.")
    x = data["x"]
    fx = z2_to_fermion_state(x)

    println("state_file = ", state_file)
    println("loaded x type = ", typeof(x), ", size = ", size(x))
    println("converted state type = ", typeof(fx), ", size = ", size(fx))

    max_B_error = 0.0
    max_T_error = 0.0
    for i in eachindex(x)
        max_B_error = max(max_B_error, conversion_error(x[i].Bm, fx[i].B))
        max_T_error = max(max_T_error, conversion_error(x[i].Tm, fx[i].T))
    end
    println("max dense B conversion error = ", max_B_error)
    println("max dense T conversion error = ", max_T_error)

    A = pess_to_ipeps_tensor(fx[1, 1])
    println("sample converted iPEPS tensor space = ", space(A))
    println("sample converted iPEPS tensor norm = ", norm(A))

    return nothing
end

convert_saved_spin_hall_state()

