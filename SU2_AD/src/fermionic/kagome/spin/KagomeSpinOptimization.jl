module KagomeSpinOptimization

using TensorKit, LinearAlgebra, TOML, JLD2, SHA
using ..KagomeFermionicZ2
using ..KagomeGFPEPSValidation
using ..KagomeFermionicOptimization
const KF = KagomeFermionicZ2
const GF = KagomeGFPEPSValidation
const Opt = KagomeFermionicOptimization
const CTM = Opt.CTM

export HeisenbergField, run_spin_optimization

include("model.jl")
include("workflow.jl")
end
