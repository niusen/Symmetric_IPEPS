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

function graded_cell_update_saved_state(; state_file::AbstractString=STATE_FILE, direction::Int=1, chi=40)
    data = load(state_file)
    haskey(data, "x") || error("Expected key `x` in saved variational state.")
    x = data["x"]

    A_cell = legacy_z2_state_to_fermion_ipeps_cell(x)
    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)
    result = graded_ctm_cell_directional_update(setup.CTM, setup.double_layer.AA; direction, chi)

    println("state_file = ", state_file)
    println("loaded x size = ", size(x))
    println("direction = ", direction)
    println("chi = ", chi)
    println("AA cell size = ", size(setup.double_layer.AA))
    for cx in axes(result.CTM, 1), cy in axes(result.CTM, 2)
        before = setup.CTM[cx, cy]
        after = result.CTM[cx, cy]
        println(
            "site ", (cx, cy),
            ": C norms ",
            round.(Float64[norm(after.Cset.C1), norm(after.Cset.C2), norm(after.Cset.C3), norm(after.Cset.C4)]; sigdigits=8),
            ", T norms ",
            round.(Float64[norm(after.Tset.T1), norm(after.Tset.T2), norm(after.Tset.T3), norm(after.Tset.T4)]; sigdigits=8),
        )
        println(
            "site ", (cx, cy),
            ": before C norms ",
            round.(Float64[norm(before.Cset.C1), norm(before.Cset.C2), norm(before.Cset.C3), norm(before.Cset.C4)]; sigdigits=8),
            ", before T norms ",
            round.(Float64[norm(before.Tset.T1), norm(before.Tset.T2), norm(before.Tset.T3), norm(before.Tset.T4)]; sigdigits=8),
        )
    end
    println("graded unit-cell directional update completed; this is not an energy calculation.")

    return nothing
end

graded_cell_update_saved_state()
