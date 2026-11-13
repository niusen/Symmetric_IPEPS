module TriangleSU4Six
using LinearAlgebra, Random, Serialization, Printf, Dates, MAT
using TensorKit, TensorKitSectors, SUNRepresentations, ChainRulesCore, KrylovKit, Zygote
using Zygote: @ignore_derivatives

# Reuse the original iPESS update, CTMRG and spin-model contraction kernels.
const SRC = normpath(joinpath(@__DIR__, "../.."))
for file in ("Settings.jl", "Settings_cell.jl", "iPEPS_ansatz.jl", "AD_lib.jl",
    "SUN_spin_op.jl", "CTMRG.jl", "CTMRG_unitcell.jl",
    "triangle/triangle_spin_model_cell.jl", "triangle/triangle_iPESS_method.jl",
    "triangle/simple_update/triangle_SimpleUpdate.jl",
    "triangle/simple_update/triangle_SimpleUpdate_iPESS.jl")
    include(joinpath(SRC, file))
end

export six_model, initial_state, initial_dimer_state, triangle_gates, simple_update!,
    measure_state, nearest_neighbor_bonds, observable_mat_data, save_observables,
    SixRunConfig, run_simulation, evaluate_checkpoint

include("Model.jl")
include("CTMRGCompat.jl")
ensure_repartition_compatibility!()
include("Evolution.jl")
include("Measurements.jl")
include("Run.jl")
end
