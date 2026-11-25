module KagomeFermionicOptimization

using TensorKit, LinearAlgebra, Random, Zygote, ChainRulesCore, JLD2
using ..KagomeFermionicZ2
using ..KagomeFermionicCTM
const KF = KagomeFermionicZ2
const KC = KagomeFermionicCTM
const CTM = KC.Legacy

include("differentiable.jl")
include("variational.jl")
include("triangle_environment.jl")
include("full_update.jl")

export VariationalProblem, variational_energy, value_and_gradient, check_gradient,
       optimize_variational, triangle_environment, fit_triangle,
       full_update_triangle!, full_update!, reverse_triangles,
       pack_state, unpack_state
end
