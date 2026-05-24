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

function graded_iterate_saved_state(;
    state_file::AbstractString=STATE_FILE,
    maxiter::Int=1,
    chi=40,
    direction_order=(3, 4, 1, 2),
    conv_check::Symbol=:spectrum,
)
    data = load(state_file)
    haskey(data, "x") || error("Expected key `x` in saved variational state.")
    x = data["x"]

    A_cell = legacy_z2_state_to_fermion_ipeps_cell(x)
    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)
    result = graded_ctm_cell_iterate(
        setup.CTM,
        setup.double_layer.AA;
        maxiter,
        chi,
        direction_order,
        conv_check,
    )

    println("state_file = ", state_file)
    println("loaded x size = ", size(x))
    println("maxiter = ", maxiter)
    println("chi = ", chi)
    println("direction_order = ", direction_order)
    println("conv_check = ", conv_check)
    println("errors = ", result.errors)
    println("initial signature length = ", length(result.signatures[1]))
    println("final signature length = ", length(result.signatures[end]))
    for cx in axes(result.CTM, 1), cy in axes(result.CTM, 2)
        CTM = result.CTM[cx, cy]
        println(
            "site ", (cx, cy),
            ": final C norms ",
            round.(Float64[norm(CTM.Cset.C1), norm(CTM.Cset.C2), norm(CTM.Cset.C3), norm(CTM.Cset.C4)]; sigdigits=8),
            ", final T norms ",
            round.(Float64[norm(CTM.Tset.T1), norm(CTM.Tset.T2), norm(CTM.Tset.T3), norm(CTM.Tset.T4)]; sigdigits=8),
        )
    end
    for cx in axes(result.CTM, 1), cy in axes(result.CTM, 2)
        patch_norm = graded_ob_2x2_norm(result.CTM, setup.double_layer.AA, cx, cy)
        println("site ", (cx, cy), ": 2x2 norm patch = ", patch_norm)
    end
    onsite = graded_spinful_onsite_observables(result.CTM, A_cell, setup.double_layer.AA)
    println("onsite id = ", onsite.id)
    println("onsite n_total = ", onsite.n_total)
    println("onsite n_double = ", onsite.n_double)
    println("onsite sx = ", onsite.sx)
    println("onsite sy = ", onsite.sy)
    println("onsite sz = ", onsite.sz)
    println("graded CTMRG iteration prototype completed; this is still not an energy calculation.")

    return nothing
end

graded_iterate_saved_state()
