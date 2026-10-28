"""Pure TensorKit kagome conversion, factorized CTMRG and observables.

The SU kernels and B(down,up,physical)/Tu/Td convention remain unchanged.
An auxiliary projected charge is kept as a one-dimensional spectator: dropping
it would make the individual polarized U1 tensors non-invariant intertwiners.
"""
module KagomeNativeMeasurement
using TensorKit, LinearAlgebra, Statistics, ChainRulesCore, JLD2, TOML, SHA
if !isdefined(parentmodule(@__MODULE__), :KagomeChargedSU2)
    Base.include(parentmodule(@__MODULE__),joinpath(@__DIR__,"..","simple_update_charged_cell.jl"))
end
if !isdefined(parentmodule(@__MODULE__), :KagomeU1Cells)
    Base.include(parentmodule(@__MODULE__),joinpath(@__DIR__,"..","simple_update_u1_cells.jl"))
end
import ..KagomeChargedSU2
import ..KagomeU1Cells
const BS=KagomeChargedSU2.BaseSU

# Reuse the repository's settings, CTM structures, coordinate maps, four-corner
# projector construction, final environment update, and GPU storage helpers.
# New overloads below only replace the parts that would build a full rank-4 AA.
module CTM
using TensorKit, LinearAlgebra, Statistics, ChainRulesCore
include(joinpath(@__DIR__,"..","..","Settings.jl"))
include(joinpath(@__DIR__,"..","..","Settings_cell.jl"))
include(joinpath(@__DIR__,"..","..","..","fermionic","Fermionic_CTMRG_unitcell_iPESS.jl"))
include(joinpath(@__DIR__,"..","..","square","gpu","square_J1_gpu_utils.jl"))
end

include("conversion_layers.jl")
include("factorized_ctm.jl")
include("observables.jl")
include("measurement_io.jl")
include("conversion_workflow.jl")
export to_dense, conversion_state, native_ctmrg, native_observables,
       measure_checkpoint, measure_state, save_converted_state, prepare_device!, run_native_conversion
end
