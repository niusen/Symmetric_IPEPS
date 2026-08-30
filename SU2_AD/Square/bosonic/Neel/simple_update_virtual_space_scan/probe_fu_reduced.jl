using TensorKit
import TensorKit: ×
using Zygote
using LinearAlgebra: dot, norm
using ChainRulesCore
using JLD2
using Zygote: @ignore_derivatives

repo = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
include(joinpath(repo, "src", "bosonic", "Settings.jl"))
include(joinpath(repo, "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(repo, "src", "bosonic", "square", "simple_update_lib.jl"))
include(joinpath(repo, "src", "bosonic", "square", "full_update_J1.jl"))
include(joinpath(repo, "src", "bosonic", "square", "full_update_J1_cell.jl"))

use_D12 = parse(Bool, get(ENV, "FU_PROBE_D12", "false"))
if use_D12
    state_file = joinpath(@__DIR__, "results", "paper_y_staggered_seed666_fine_long", "final.jld2")
    A_set = load(state_file, "T_set")
    probe_Dmax = 12
else
    Vp = SU2Space(1 / 2 => 1)
    Vv = SU2Space(0 => 1, 1 / 2 => 1)
    A_set, _, _ = initial_iPEPS(2, 2, Vp, Vv)
    A_set = [A_set[cx, cy] / norm(A_set[cx, cy]) for cx in 1:2, cy in 1:2]
    probe_Dmax = dim(Vv)
end
_square_fu_validate_cell(A_set)
gate = prepare_gate_Heisenberg(0.01, "SU2")

probe_direction = Symbol(get(ENV, "FU_PROBE_DIRECTION", "both"))
y_start = parse(Int, get(ENV, "FU_PROBE_Y_START", "1"))
all_bonds = (
    SquareJ1CellBond(:x, CartesianIndex(1, 1), CartesianIndex(2, 1)),
    SquareJ1CellBond(
        :y,
        CartesianIndex(1, y_start),
        CartesianIndex(1, mod1(y_start + 1, size(A_set, 2))),
    ),
)
probe_bonds = probe_direction === :both ? all_bonds :
    filter(bond -> bond.direction === probe_direction, all_bonds)
for bond in probe_bonds
    A1, A2 = A_set[bond.site1], A_set[bond.site2]
    r1, k1, k2, r2 = _square_fu_split_reduced(A1, A2, bond.direction)
    println(bond.direction, " reduced spaces:")
    for (name, tensor) in (("r1", r1), ("k1", k1), ("k2", k2), ("r2", r2))
        println("  ", name, " = ", space(tensor))
    end
    reconstructed1, reconstructed2 =
        _square_fu_reassemble_reduced(r1, k1, k2, r2, bond.direction)
    println(bond.direction, " reconstruction errors = ",
            (norm(reconstructed1 - A1) / norm(A1), norm(reconstructed2 - A2) / norm(A2)))

    gated = _square_fu_gated_bond(k1, k2, gate)
    candidate1, candidate2, singular_values = _square_fu_factor_bond(
        gated; truncation=truncdim(probe_Dmax; multiplet_tol=1e-5),
    )
    candidate_A1, candidate_A2 =
        _square_fu_reassemble_reduced(r1, candidate1, candidate2, r2, bond.direction)
    AA1, _ = build_square_cross_double_layer_open(candidate_A1, A1)
    AA2, _ = build_square_cross_double_layer_open(candidate_A2, A2)
    println(bond.direction, " selected space = ", space(singular_values, 1))
    println(bond.direction, " cross internal spaces = ",
            bond.direction === :x ? (space(AA1, 3), space(AA2, 1)) :
                                    (space(AA1, 4), space(AA2, 2)))
end
