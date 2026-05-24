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

function graded_left_update_saved_state(; state_file::AbstractString=STATE_FILE, site::Tuple{Int,Int}=(1, 1), chi=nothing)
    data = load(state_file)
    haskey(data, "x") || error("Expected key `x` in saved variational state.")
    x = data["x"]

    A_cell = legacy_z2_state_to_fermion_ipeps_cell(x)
    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)

    cx, cy = site
    AA = setup.double_layer.AA[cx, cy]
    CTM = setup.CTM[cx, cy]
    result = graded_ctm_left_update(CTM, AA; chi)
    CTM2 = result.CTM

    println("state_file = ", state_file)
    println("loaded x size = ", size(x))
    println("updated site = ", site)
    println("chi = ", chi === nothing ? "notrunc" : chi)
    println("AA norm = ", norm(AA))
    println("P norm = ", norm(result.projectors.P))
    println("Pinv norm = ", norm(result.projectors.Pinv))
    println("C1 norm before -> after = ", norm(CTM.Cset.C1), " -> ", norm(CTM2.Cset.C1))
    println("C4 norm before -> after = ", norm(CTM.Cset.C4), " -> ", norm(CTM2.Cset.C4))
    println("T4 norm before -> after = ", norm(CTM.Tset.T4), " -> ", norm(CTM2.Tset.T4))
    println("C1 dim after = ", dim(space(CTM2.Cset.C1, 1)), " x ", dim(space(CTM2.Cset.C1, 2)))
    println("T4 dims after = ", map(i -> dim(space(CTM2.Tset.T4, i)), 1:numind(CTM2.Tset.T4)))
    println("graded left update completed; this is one branch-A CTMRG absorption step, not an energy calculation.")

    return nothing
end

graded_left_update_saved_state()
