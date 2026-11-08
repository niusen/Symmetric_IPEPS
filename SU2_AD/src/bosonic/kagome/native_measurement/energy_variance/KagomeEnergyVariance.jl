"""Large-cell energy variance, using the existing kagome Q/R and CTM kernels.

Only measurement is performed. The saved state and the SU code are untouched.
Exchange Hamiltonian only; it has the same variance as Hex+h*Sz_total for a
state in an exact fixed-Sz sector. A dense state need not have this property.
"""
module KagomeEnergyVariance
using TensorKit, LinearAlgebra, Statistics, TOML, SHA, Dates
native=get(ENV,"KAGOME_VARIANCE_NATIVE_ROOT",normpath(joinpath(@__DIR__,"..")))
if !isdefined(parentmodule(@__MODULE__),:KagomeNativeMeasurement)
    Base.include(parentmodule(@__MODULE__),joinpath(native,"KagomeNativeMeasurement.jl"))
end
import ..KagomeNativeMeasurement
const N=KagomeNativeMeasurement
const C=N.CTM
const PAIRS=((1,2),(2,3),(3,1))
include("variance_operators.jl")
include("variance_contraction.jl")
include("variance_array_sweeps.jl")
include("variance_workflow.jl")
if !isdefined(parentmodule(@__MODULE__),:KagomeVariancePlan)
    Base.include(parentmodule(@__MODULE__),joinpath(@__DIR__,"variance_batch_plan.jl"))
end
import ..KagomeVariancePlan
const Plan=KagomeVariancePlan
include("variance_batch.jl")
export exchange_variance, measure_variance_checkpoint, bond_catalogue, run_variance_batch, run_variance_benchmark
end
