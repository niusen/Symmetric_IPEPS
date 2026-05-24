using JLD2
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

x = load(STATE_FILE)["x"]
A = legacy_z2_pess_to_fermion_ipeps_tensor(x[1, 1])
Ap = A'

println("A dims = ", [dim(space(A, i)) for i in 1:5])
println("Ap dims = ", [dim(space(Ap, i)) for i in 1:5])
println("A spaces = ")
foreach(i -> println("  A[$i]  = ", space(A, i)), 1:5)
println("Ap spaces = ")
foreach(i -> println("  Ap[$i] = ", space(Ap, i)), 1:5)
