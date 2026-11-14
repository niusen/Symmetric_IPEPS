"""Out-of-plane DMI, reusing the existing Julia kagome SU and factorized CTMRG.

All elementary triangles have CCW arrows in Cartesian (a1,a2) coordinates.
The user's B1/B2/B3 simplex order is CLOCKWISE for both up and down triangles.
H = J*sum(SS) + Dz*sum_CCW(kappa_z) + h*sum(Sz), retaining the legacy +h sign.
Only Dz=0 is SU2 invariant; Dz≠0 supports U1 or dense tensors, not SU2 SU.
"""
module KagomeDMI
using TensorKit, LinearAlgebra, Random, JLD2, TOML
if !isdefined(parentmodule(@__MODULE__),:KagomeNativeMeasurement)
    Base.include(parentmodule(@__MODULE__),joinpath(@__DIR__,"..","native_measurement","KagomeNativeMeasurement.jl"))
end
import ..KagomeNativeMeasurement
const N=KagomeNativeMeasurement
const BS=N.BS
const SU=N.KagomeU1Cells
const CYCLE=((1,2),(2,3),(3,1))
# Reuse the version-neutral full U1 seed reader/writer, including lambdas and
# per-axis charge ordering. Do not reconstruct a seed without its SU weights.
const ServerU1=SU
const ServerBase=BS
server_atomic_toml(path,data)=N.atomic_toml(path,data)
include(joinpath(@__DIR__,"..","native_measurement","u1_seed_io.jl"))
export triangle_hamiltonian, random_dense_state, product_120, evolve!, measure,
       save_checkpoint, load_checkpoint, run
include("model.jl")
include("observables.jl")
include("workflow.jl")
end
