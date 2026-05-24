using JLD2
using LinearAlgebra
using TensorKit
using AutoFermionicPESS

const AUTO_DIR = normpath(joinpath(@__DIR__, ".."))
const SU2_AD_SRC = normpath(joinpath(AUTO_DIR, ".."))
const SU2_AD_ROOT = normpath(joinpath(SU2_AD_SRC, ".."))
const STATE_FILE = joinpath(
    SU2_AD_ROOT,
    "Triangle",
    "iPESS",
    "spinHall",
    "variational",
    "var_iPESS_Z2_Triangle_Hofstadter_Hubbard_spinHall_D4_chi_40_-2.3854.jld2",
)

include(joinpath(SU2_AD_SRC, "bosonic", "iPEPS_ansatz.jl"))

function graded_skeleton_saved_state(; state_file::AbstractString=STATE_FILE)
    data = load(state_file)
    haskey(data, "x") || error("Expected key `x` in saved variational state.")
    x = data["x"]

    A_cell = legacy_z2_state_to_fermion_ipeps_cell(x)
    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)

    println("state_file = ", state_file)
    println("loaded x size = ", size(x))
    println("graded A_cell size = ", size(A_cell))
    println("sample A space = ", space(A_cell[1, 1]))
    println("sample AA space = ", space(setup.double_layer.AA[1, 1]))
    println("sample AA norm = ", norm(setup.double_layer.AA[1, 1]))
    println("sample C1 norm = ", norm(setup.CTM[1, 1].Cset.C1))
    println("sample T1 norm = ", norm(setup.CTM[1, 1].Tset.T1))
    println("graded CTM skeleton initialized; no CTMRG iterations or energy yet.")

    return nothing
end

graded_skeleton_saved_state()
