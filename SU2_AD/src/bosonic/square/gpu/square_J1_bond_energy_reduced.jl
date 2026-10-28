"""
Square-J1 bond energy without constructing a full open double layer.

Reuse the FU SVD split and the residual-to-CTM half contractions, not its
optimization or positive-metric machinery. Include after full_update_J1_cell.jl,
square_J1_gpu_utils.jl and square_J1_full_update_als.jl. Works on the selected
CPU/GPU device, with CTM fusion maps belonging to the measured A tensors.
"""

function _square_J1_validate_energy_method(method::Symbol)
    method in (:legacy, :reduced) || throw(ArgumentError(
        "energy_method must be :legacy (original AA_open) or :reduced (FU-style contraction); got $method",
    ))
    return method
end

# Keep only tensors needed by later stages. Non-inlined stages return nothing;
# collect AFTER their temporary references have left the allocating frame.
Base.@kwdef mutable struct _SquareJ1BondEnergyWorkspace
    residual1::Any = nothing
    residual2::Any = nothing
    pair_map::Any = nothing
    half1::Any = nothing
    half2::Any = nothing
    raw_metric::Any = nothing
    weighted_pair::Any = nothing
    rho::Any = nothing
end

Base.@noinline function _square_J1_bond_energy_stage_done(
    label, work::_SquareJ1BondEnergyWorkspace, reference, debug_memory::Bool,
)
    on_cpu = square_J1_storage_family(TensorKit.storagetype(reference)) === Array
    if on_cpu
        GC.gc(true)
    else
        square_J1_reclaim_device_memory!(aggressive=true)
    end
    debug_memory || return nothing
    bytes = 0
    for field in fieldnames(_SquareJ1BondEnergyWorkspace)
        tensor = getfield(work, field)
        isnothing(tensor) && continue
        bytes += sum(length(b) for (_, b) in blocks(tensor)) *
                 sizeof(TensorKit.scalartype(tensor))
    end
    # Payload sum, not a peak or unique allocation count; excludes A/CTM/maps.
    println("  reduced energy $label: workspace data=$(bytes / 2.0^30) GiB")
    on_cpu || square_J1_print_device_memory("  reduced energy $label after stage/reclaim:")
    flush(stdout)
    return nothing
end

Base.@noinline function _square_J1_bond_energy_split!(work, A1, A2, direction)
    # Exact SVD split: no evolution gate and no Dmax/multiplet truncation.
    residual1, keep1, keep2, residual2 = _square_fu_split_reduced(A1, A2, direction)
    # FU pair order is (a,p1,b,p2). For measurement use B:(p1,p2)->(a,b).
    pair = _square_fu_als_reduced_pair(keep1, keep2)
    work.pair_map = permute(pair, (1, 3), (2, 4))
    work.residual1, work.residual2 = residual1, residual2
    return nothing
end

Base.@noinline function _square_J1_bond_energy_half1!(work, environment, bond, cell_Lx, cell_Ly)
    x1, y1 = Tuple(bond.site1)
    position = bond.direction === :x ? :x_left : :y_upper
    # Reuse the existing explicit binary contraction order, including the CTM
    # fusion basis and y-direction orientation. Do not construct residual AA.
    work.half1 = _square_fu_als_residual_half(
        environment, work.residual1, position, x1 - 1, y1 - 1,
        cell_Lx, cell_Ly, bond.site1,
    )
    work.residual1 = nothing
    return nothing
end

Base.@noinline function _square_J1_bond_energy_half2!(work, environment, bond, cell_Lx, cell_Ly)
    x1, y1 = Tuple(bond.site1)
    position = bond.direction === :x ? :x_right : :y_lower
    work.half2 = _square_fu_als_residual_half(
        environment, work.residual2, position, x1 - 1, y1 - 1,
        cell_Lx, cell_Ly, bond.site2,
    )
    work.residual2 = nothing
    return nothing
end

Base.@noinline function _square_J1_bond_energy_metric!(work)
    half1, half2 = work.half1, work.half2
    # The first two legs of each half are the transverse CTM boundaries;
    # its last two are (bra reduced index, ket reduced index).
    @tensor metric[:] := half1[1, 2, -1, -3] * half2[1, 2, -2, -4]
    work.raw_metric = permute(metric, (1, 2), (3, 4))
    work.half1 = work.half2 = nothing
    # Do NOT call _square_fu_als_metric: it Hermitian-symmetrizes the result.
    # Do NOT call _square_fu_als_environment: it clips eigenvalues/factorizes it.
    return nothing
end

Base.@noinline function _square_J1_bond_energy_project!(work)
    # E[(bra a,bra b),(ket a,ket b)] * B[(ket a,ket b),(ket p1,ket p2)].
    work.weighted_pair = work.raw_metric * work.pair_map
    work.raw_metric = nothing
    return nothing
end

Base.@noinline function _square_J1_bond_energy_density!(work)
    # rho = B' E B, ordered (bra p1,bra p2,ket p1,ket p2), exactly the order
    # used by the legacy energy trace/contraction. No metric correction.
    work.rho = work.pair_map' * work.weighted_pair
    work.pair_map = work.weighted_pair = nothing
    return nothing
end

"""
    square_J1_bond_energy_reduced(environment, A1, A2, bond, Lx, Ly, H;
                                  J1=1, debug_memory=false)

Measure one bond with the raw 2x1/1x2 CTM network and exact reduced site tensors.
H must have the same scalar/storage type as A1, and physical order matching
the legacy Heisenberg operator. No AA_open, optimizer, gate, bond truncation,
metric Hermitianization, or positivity projection is used. Return a scalar
using the legacy convention real(numerator)/real(norm).
"""
Base.@noinline function square_J1_bond_energy_reduced(
    environment, A1::TensorMap, A2::TensorMap, bond::SquareJ1CellBond,
    cell_Lx::Int, cell_Ly::Int, H::TensorMap;
    J1::Real=1, debug_memory::Bool=false,
)
    bond.direction in (:x, :y) || throw(ArgumentError("bond direction must be :x or :y"))
    all(name -> hasproperty(environment, name), (:CTM, :U_L, :U_D, :U_R, :U_U)) ||
        throw(ArgumentError("reduced energy needs CTM and its U_L/U_D/U_R/U_U fusion maps"))
    label = "$(bond.direction) bond $(Tuple(bond.site1))"
    work = _SquareJ1BondEnergyWorkspace()

    _square_J1_bond_energy_split!(work, A1, A2, bond.direction)
    _square_J1_bond_energy_stage_done("$label split", work, A1, debug_memory)
    _square_J1_bond_energy_half1!(work, environment, bond, cell_Lx, cell_Ly)
    _square_J1_bond_energy_stage_done("$label first half", work, A1, debug_memory)
    _square_J1_bond_energy_half2!(work, environment, bond, cell_Lx, cell_Ly)
    _square_J1_bond_energy_stage_done("$label second half", work, A1, debug_memory)
    _square_J1_bond_energy_metric!(work)
    _square_J1_bond_energy_stage_done("$label raw metric", work, A1, debug_memory)
    _square_J1_bond_energy_project!(work)
    _square_J1_bond_energy_stage_done("$label metric times pair", work, A1, debug_memory)
    _square_J1_bond_energy_density!(work)
    _square_J1_bond_energy_stage_done("$label physical density", work, A1, debug_memory)

    rho = work.rho
    norm_rho = real(@tensor rho[1, 2, 1, 2])
    isfinite(norm_rho) && norm_rho != 0 || throw(ArgumentError(
        "zero or nonfinite reduced CTM norm for $label: $norm_rho",
    ))
    energy_numerator = @tensor rho[1, 2, 3, 4] * H[1, 2, 3, 4]
    energy = J1 * real(energy_numerator) / norm_rho
    isfinite(energy) || throw(ArgumentError("nonfinite reduced CTM energy for $label"))
    return energy
end

"""Compare all bond energies using the same state and same precomputed CTM.

Only call on small D unless deliberately testing memory limits: the reference
method constructs AA_open. Requires square_J1_full_update_gpu.jl to be included.
This does not rerun CTMRG, perform an update, or modify the input state.
"""
function square_J1_compare_energy_methods(
    A_set::AbstractMatrix, environment;
    J1::Real=1, device::AbstractString="cpu",
    atol::Real=1.0e-10, rtol::Real=1.0e-9, debug_memory::Bool=false,
)
    atol >= 0 && rtol >= 0 || throw(ArgumentError("comparison tolerances must be nonnegative"))
    legacy = square_J1_gpu_energy_cell(
        A_set, environment; J1, device, low_memory=true, debug_memory,
        energy_method=:legacy,
    )
    reduced = square_J1_gpu_energy_cell(
        A_set, environment; J1, device, low_memory=true, debug_memory,
        energy_method=:reduced,
    )
    rows = NamedTuple[]
    for direction in (:x, :y)
        old_values = direction === :x ? legacy.Ex : legacy.Ey
        new_values = direction === :x ? reduced.Ex : reduced.Ey
        for site in CartesianIndices(old_values)
            old_energy, new_energy = old_values[site], new_values[site]
            difference = abs(old_energy - new_energy)
            passed = isapprox(old_energy, new_energy; atol, rtol)
            println("$direction$(Tuple(site)): legacy=$old_energy reduced=$new_energy " *
                    "abs_diff=$difference passed=$passed")
            push!(rows, (; direction, site=Tuple(site), legacy=old_energy,
                         reduced=new_energy, abs_difference=difference, passed))
        end
    end
    total_difference = abs(legacy.energy_per_site - reduced.energy_per_site)
    total_passed = isapprox(legacy.energy_per_site, reduced.energy_per_site; atol, rtol)
    passed = all(row -> row.passed, rows) && total_passed
    max_abs_bond_error = maximum(row.abs_difference for row in rows)
    exact_match = isequal(legacy.Ex, reduced.Ex) && isequal(legacy.Ey, reduced.Ey) &&
                  isequal(legacy.energy_per_site, reduced.energy_per_site)
    println("energy/site: legacy=$(legacy.energy_per_site) reduced=$(reduced.energy_per_site) " *
            "abs_diff=$total_difference passed=$total_passed")
    println("Comparison: passed=$passed, max_abs_bond_error=$max_abs_bond_error, " *
            "bitwise_equal=$exact_match, atol=$atol, rtol=$rtol")
    flush(stdout)
    return (; passed, exact_match, max_abs_bond_error, total_difference,
            atol, rtol, legacy, reduced, bonds=rows)
end
