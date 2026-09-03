"""
Full Update for a bosonic square-lattice nearest-neighbour J1 model on an
arbitrary periodic `Lx × Ly` iPEPS unit cell.

Each updated bond is reduced by two local SVDs to a pair of rank-3 tensors;
the gate, multiplet-aware truncation, and alternating environment optimization
act only on that reduced two-site object.  The periodic-cell/CTM organization
follows the triangular-lattice Full Update.  Include, in order,
`Settings_cell.jl`, `CTMRG_unitcell.jl`, and `full_update_J1.jl` before this
file.
"""

struct SquareJ1CellBond
    direction::Symbol
    site1::CartesianIndex{2}
    site2::CartesianIndex{2}
end

function square_fu_cell_to_tuple(A_set::AbstractMatrix)
    cell_Lx, cell_Ly = size(A_set)
    return ntuple(cx -> ntuple(cy -> A_set[cx, cy], cell_Ly), cell_Lx)
end

function square_fu_cell_to_matrix(A_cell::Tuple)
    cell_Lx = length(A_cell)
    cell_Lx > 0 || throw(ArgumentError("the iPEPS cell cannot be empty"))
    cell_Ly = length(A_cell[1])
    all(length(column) == cell_Ly for column in A_cell) ||
        throw(ArgumentError("A_cell must be a rectangular tuple of tuples"))
    return [A_cell[cx][cy] for cx in 1:cell_Lx, cy in 1:cell_Ly]
end

function _square_fu_validate_cell(A_set::AbstractMatrix)
    cell_Lx, cell_Ly = size(A_set)
    cell_Lx > 0 || throw(ArgumentError("Lx must be positive"))
    cell_Ly > 0 || throw(ArgumentError("Ly must be positive"))
    A_reference = A_set[1, 1]
    A_reference isa TensorMap || throw(ArgumentError("every cell entry must be a TensorMap"))
    for position in CartesianIndices(A_set)
        A = A_set[position]
        A isa TensorMap || throw(ArgumentError("cell entry $position is not a TensorMap"))
        numind(A) == 5 || throw(ArgumentError(
            "cell entry $position must have five legs (L,D,R,U,physical)",
        ))
        space(A, 5) == space(A_reference, 5) || throw(SpaceMismatch(
            "physical space at $position differs from site (1,1)",
        ))
    end
    for cx in 1:cell_Lx, cy in 1:cell_Ly
        A = A_set[cx, cy]
        A_right = A_set[mod1(cx + 1, cell_Lx), cy]
        A_below = A_set[cx, mod1(cy + 1, cell_Ly)]
        space(A, 3)' == space(A_right, 1) || throw(SpaceMismatch(
            "horizontal bond spaces do not match between ($cx,$cy) and its right neighbour",
        ))
        space(A, 2)' == space(A_below, 4) || throw(SpaceMismatch(
            "vertical bond spaces do not match between ($cx,$cy) and its lower neighbour",
        ))
    end
    return cell_Lx, cell_Ly
end

_square_fu_cycle_color(index::Int, length::Int) =
    isodd(length) && index == length ? 3 : (isodd(index) ? 1 : 2)

"""
    square_J1_bond_groups(Lx, Ly)

Partition every directed positive-x and positive-y nearest-neighbour bond of a
periodic cell into non-overlapping groups.  Even cycles need two colors and odd
cycles need a third color for their closing bond, following the grouping idea
used by `get_triangles_PBC` in the fermionic triangular-lattice update.
"""
function square_J1_bond_groups(cell_Lx::Int, cell_Ly::Int)
    cell_Lx > 0 || throw(ArgumentError("Lx must be positive"))
    cell_Ly > 0 || throw(ArgumentError("Ly must be positive"))
    groups = Vector{Vector{SquareJ1CellBond}}()

    number_x_colors = 2 + Int(isodd(cell_Lx))
    horizontal = [SquareJ1CellBond[] for _ in 1:number_x_colors]
    for cy in 1:cell_Ly, cx in 1:cell_Lx
        color = _square_fu_cycle_color(cx, cell_Lx)
        push!(horizontal[color], SquareJ1CellBond(
            :x,
            CartesianIndex(cx, cy),
            CartesianIndex(mod1(cx + 1, cell_Lx), cy),
        ))
    end
    append!(groups, filter(group -> !isempty(group), horizontal))

    number_y_colors = 2 + Int(isodd(cell_Ly))
    vertical = [SquareJ1CellBond[] for _ in 1:number_y_colors]
    for cx in 1:cell_Lx, cy in 1:cell_Ly
        color = _square_fu_cycle_color(cy, cell_Ly)
        push!(vertical[color], SquareJ1CellBond(
            :y,
            CartesianIndex(cx, cy),
            CartesianIndex(cx, mod1(cy + 1, cell_Ly)),
        ))
    end
    append!(groups, filter(group -> !isempty(group), vertical))

    all_bonds = reduce(vcat, groups; init=SquareJ1CellBond[])
    count(bond -> bond.direction === :x, all_bonds) == cell_Lx * cell_Ly ||
        error("horizontal bond grouping lost or duplicated a bond")
    count(bond -> bond.direction === :y, all_bonds) == cell_Lx * cell_Ly ||
        error("vertical bond grouping lost or duplicated a bond")
    for group in groups
        occupied = Set{CartesianIndex{2}}()
        for bond in group
            for site in unique((bond.site1, bond.site2))
                site in occupied && error("bond group contains overlapping sites")
                push!(occupied, site)
            end
        end
    end
    return groups
end

# CTM environment of a horizontal 2×1 cluster; there are no spectator sites.
function _square_fu_ob_2sites_x_cell(
    cx::Int,
    cy::Int,
    CTM,
    AA_1,
    AA_2,
    cell_Lx::Int,
    cell_Ly::Int,
)
    Cset = CTM.Cset
    Tset = CTM.Tset

    @tensor envL[:] := Cset[mod1(cx, cell_Lx)][mod1(cy, cell_Ly)].C1[1, -1] *
        Tset[mod1(cx, cell_Lx)][mod1(cy + 1, cell_Ly)].T4[2, -2, 1] *
        Cset[mod1(cx, cell_Lx)][mod1(cy + 2, cell_Ly)].C4[-3, 2]
    @tensor envR[:] := Cset[mod1(cx + 3, cell_Lx)][mod1(cy, cell_Ly)].C2[-1, 1] *
        Tset[mod1(cx + 3, cell_Lx)][mod1(cy + 1, cell_Ly)].T2[1, -2, 2] *
        Cset[mod1(cx + 3, cell_Lx)][mod1(cy + 2, cell_Ly)].C3[2, -3]
    @tensor envL[:] := envL[1, 2, 4] *
        Tset[mod1(cx + 1, cell_Lx)][mod1(cy, cell_Ly)].T1[1, 3, -1] *
        AA_1[2, 5, -2, 3, -4] *
        Tset[mod1(cx + 1, cell_Lx)][mod1(cy + 2, cell_Ly)].T3[-3, 5, 4]
    @tensor envR[:] := Tset[mod1(cx + 2, cell_Lx)][mod1(cy, cell_Ly)].T1[-1, 3, 1] *
        AA_2[-2, 5, 2, 3, -4] *
        Tset[mod1(cx + 2, cell_Lx)][mod1(cy + 2, cell_Ly)].T3[4, 5, -3] *
        envR[1, 2, 4]
    @tensor rho[:] := envL[1, 2, 3, -1] * envR[1, 2, 3, -2]
    return rho
end

# CTM environment of a vertical 1×2 cluster; there are no spectator sites.
function _square_fu_ob_2sites_y_cell(
    cx::Int,
    cy::Int,
    CTM,
    AA_1,
    AA_2,
    cell_Lx::Int,
    cell_Ly::Int,
)
    Cset = CTM.Cset
    Tset = CTM.Tset

    @tensor envU[:] := Cset[mod1(cx + 2, cell_Lx)][mod1(cy, cell_Ly)].C2[1, -1] *
        Tset[mod1(cx + 1, cell_Lx)][mod1(cy, cell_Ly)].T1[2, -2, 1] *
        Cset[mod1(cx, cell_Lx)][mod1(cy, cell_Ly)].C1[-3, 2]
    @tensor envD[:] := Cset[mod1(cx + 2, cell_Lx)][mod1(cy + 3, cell_Ly)].C3[-1, 1] *
        Tset[mod1(cx + 1, cell_Lx)][mod1(cy + 3, cell_Ly)].T3[1, -2, 2] *
        Cset[mod1(cx, cell_Lx)][mod1(cy + 3, cell_Ly)].C4[2, -3]
    @tensor envU[:] := envU[1, 2, 4] *
        Tset[mod1(cx + 2, cell_Lx)][mod1(cy + 1, cell_Ly)].T2[1, 3, -1] *
        AA_1[5, -2, 3, 2, -4] *
        Tset[mod1(cx, cell_Lx)][mod1(cy + 1, cell_Ly)].T4[-3, 5, 4]
    @tensor envD[:] := Tset[mod1(cx + 2, cell_Lx)][mod1(cy + 2, cell_Ly)].T2[-1, 3, 1] *
        AA_2[5, 2, 3, -2, -4] *
        Tset[mod1(cx, cell_Lx)][mod1(cy + 2, cell_Ly)].T4[4, 5, -3] *
        envD[1, 2, 4]
    @tensor rho[:] := envU[1, 2, 3, -1] * envD[1, 2, 3, -2]
    return rho
end

function _square_fu_two_site_density_cell(
    CTM,
    A_bra_1::TensorMap,
    A_ket_1::TensorMap,
    A_bra_2::TensorMap,
    A_ket_2::TensorMap,
    bond::SquareJ1CellBond,
    cell_Lx::Int,
    cell_Ly::Int,
)
    x1, y1 = Tuple(bond.site1)
    anchor_x, anchor_y = x1 - 1, y1 - 1

    if bond.direction === :x
        AA_1, U_physical_1 = build_square_cross_double_layer_open(A_bra_1, A_ket_1)
        AA_2, U_physical_2 = build_square_cross_double_layer_open(A_bra_2, A_ket_2)
        rho_fused = _square_fu_ob_2sites_x_cell(
            anchor_x,
            anchor_y,
            CTM,
            AA_1,
            AA_2,
            cell_Lx,
            cell_Ly,
        )
        @tensor rho[:] := rho_fused[1, 2] *
            U_physical_1[-1, -3, 1] *
            U_physical_2[-2, -4, 2]
        return rho
    elseif bond.direction === :y
        # Site 1 is above site 2 in the cell convention.  The cluster contracts
        # site1.D to site2.U and preserves physical order (site1, site2).
        AA_1, U_physical_1 = build_square_cross_double_layer_open(A_bra_1, A_ket_1)
        AA_2, U_physical_2 = build_square_cross_double_layer_open(A_bra_2, A_ket_2)
        rho_fused = _square_fu_ob_2sites_y_cell(
            anchor_x,
            anchor_y,
            CTM,
            AA_1,
            AA_2,
            cell_Lx,
            cell_Ly,
        )
        @tensor rho[:] := rho_fused[1, 2] *
            U_physical_1[-1, -3, 1] *
            U_physical_2[-2, -4, 2]
        return rho
    else
        throw(ArgumentError("bond direction must be :x or :y"))
    end
end

"""Split two square-iPEPS tensors down to the two rank-3 tensors touching a bond."""
function _square_fu_split_reduced(A1, A2, direction::Symbol)
    if direction === :x
        residual1, singular1, right1 = tsvd(permute(A1, (1, 2, 4), (3, 5)))
        left2, singular2, residual2 = tsvd(permute(A2, (1, 5), (2, 3, 4)))
    elseif direction === :y
        # A1 is above A2.  Split off A1.D and A2.U, matching the 1×2 CTM
        # cluster contraction and the square-lattice coordinate convention.
        residual1, singular1, right1 = tsvd(permute(A1, (1, 3, 4), (2, 5)))
        left2, singular2, residual2 = tsvd(permute(A2, (4, 5), (1, 2, 3)))
    else
        throw(ArgumentError("bond direction must be :x or :y"))
    end
    keep1 = singular1 * right1
    keep2 = left2 * singular2
    return residual1, keep1, keep2, residual2
end

"""Reassemble a pair of rank-5 tensors from fixed rank-4 and variable rank-3 parts."""
function _square_fu_reassemble_reduced(residual1, keep1, keep2, residual2, direction::Symbol)
    if direction === :x
        @tensor A1[:] := residual1[-1, -2, -4, 1] * keep1[1, -3, -5]
        @tensor A2[:] := keep2[-1, -5, 1] * residual2[1, -2, -3, -4]
    elseif direction === :y
        @tensor A1[:] := residual1[-1, -3, -4, 1] * keep1[1, -2, -5]
        @tensor A2[:] := keep2[-4, -5, 1] * residual2[1, -1, -2, -3]
    else
        throw(ArgumentError("bond direction must be :x or :y"))
    end
    return A1, A2
end

"""Apply the two-site gate only to the reduced bond tensor."""
function _square_fu_gated_bond(keep1, keep2, gate)
    @tensor gated[:] := keep1[-1, 1, 2] * keep2[1, 3, -3] * gate[-2, -4, 2, 3]
    return permute(gated, (1, 2), (3, 4))
end

function _square_fu_factor_bond(bond_tensor; truncation=nothing)
    u, s, v = isnothing(truncation) ? tsvd(bond_tensor) :
        tsvd(bond_tensor; trunc=truncation)
    # Canonical reduced layouts match the tensors returned by
    # `_square_fu_split_reduced`: keep1=(environment,bond,physical) and
    # keep2=(bond,physical,environment).
    keep1 = permute(u * sqrt(s), (1,), (3, 2))
    keep2 = permute(sqrt(s) * v, (1, 3), (2,))
    return keep1, keep2, s
end

_square_fu_rho_trace(rho) = @tensor rho[1, 2, 1, 2]

function _square_fu_reduced_fidelity(
    keep1,
    keep2,
    residual1,
    residual2,
    old_A1,
    old_A2,
    gate,
    environment,
    bond,
    cell_Lx,
    cell_Ly;
    target_norm,
    metric_floor,
)
    candidate1, candidate2 = _square_fu_reassemble_reduced(
        residual1, keep1, keep2, residual2, bond.direction,
    )
    rho_candidate = _square_fu_two_site_density_cell(
        environment.CTM,
        candidate1,
        candidate1,
        candidate2,
        candidate2,
        bond,
        cell_Lx,
        cell_Ly,
    )
    norm_candidate = real(_square_fu_rho_trace(rho_candidate))
    rho_overlap = _square_fu_two_site_density_cell(
        environment.CTM,
        candidate1,
        old_A1,
        candidate2,
        old_A2,
        bond,
        cell_Lx,
        cell_Ly,
    )
    overlap = @tensor rho_overlap[1, 2, 3, 4] * gate[1, 2, 3, 4]
    denominator = abs(norm_candidate * target_norm)
    denominator > metric_floor || throw(ArgumentError(
        "the reduced two-site CTM fidelity denominator is too small: $denominator",
    ))
    return real(abs2(overlap) / denominator)
end

function _square_fu_cell_report(bond, loss_initial, loss_current, iteration, accepted_steps,
                                gradient_norm, target_norm, loss_history)
    return (
        direction=bond.direction,
        site1=Tuple(bond.site1),
        site2=Tuple(bond.site2),
        loss_initial=loss_initial,
        loss_final=loss_current,
        fidelity=1 - loss_current,
        iterations=iteration,
        accepted_steps=accepted_steps,
        gradient_norm=gradient_norm,
        target_norm=target_norm,
        loss_history=loss_history,
    )
end

function _square_fu_cell_take_step(objective, A_current, settings)
    loss_current = objective(A_current)
    gradient_tensor = Zygote.gradient(objective, A_current)[1]
    gradient_tensor === nothing && error("Zygote returned no gradient for a cell local subproblem")
    gradient_norm = norm(gradient_tensor)
    gradient_norm <= settings.gradient_tolerance &&
        return A_current, loss_current, gradient_norm, false

    direction_tensor = -gradient_tensor
    slope = real(dot(gradient_tensor, direction_tensor))
    step = settings.initial_step
    while step >= settings.minimum_step
        A_trial = _square_fu_normalize(A_current + step * direction_tensor)
        loss_trial = objective(A_trial)
        if isfinite(loss_trial) &&
           loss_trial <= loss_current + settings.armijo * step * slope
            return A_trial, loss_trial, gradient_norm, true
        end
        step *= settings.backtracking_factor
    end
    return A_current, loss_current, gradient_norm, false
end

function square_J1_full_update_cell_bond(
    A_set::AbstractMatrix,
    environment,
    gate::TensorMap,
    bond::SquareJ1CellBond;
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
)
    cell_Lx, cell_Ly = size(A_set)
    A1_old = A_set[bond.site1]
    A2_old = A_set[bond.site2]
    bond.site1 != bond.site2 || throw(ArgumentError(
        "bond-expanding Full Update requires two distinct tensor entries",
    ))
    settings.Dmax > 0 || throw(ArgumentError("Dmax must be positive"))
    settings.multiplet_tol >= 0 || throw(ArgumentError("multiplet_tol must be non-negative"))

    residual1, old_keep1, old_keep2, residual2 =
        _square_fu_split_reduced(A1_old, A2_old, bond.direction)
    gated_bond = _square_fu_gated_bond(old_keep1, old_keep2, gate)

    # The fork-specific multiplet-aware truncation chooses a new SU(2) bond
    # space before the CTM-environment variational sweeps begin.  The exact target is
    # kept as `gate * |old>`; it is never factorized or truncated.
    truncation = truncdim(settings.Dmax; multiplet_tol=settings.multiplet_tol)
    keep1_current, keep2_current, singular_values =
        _square_fu_factor_bond(gated_bond; truncation=truncation)

    rho_old = _square_fu_two_site_density_cell(
        environment.CTM,
        A1_old,
        A1_old,
        A2_old,
        A2_old,
        bond,
        cell_Lx,
        cell_Ly,
    )
    gate_norm = gate' * gate
    target_norm = real(@tensor rho_old[1, 2, 3, 4] * gate_norm[1, 2, 3, 4])
    target_norm > settings.metric_floor || throw(ArgumentError(
        "the reduced two-site CTM target norm is non-positive: $target_norm",
    ))

    objective(keep1, keep2) = 1 - _square_fu_reduced_fidelity(
        keep1,
        keep2,
        residual1,
        residual2,
        A1_old,
        A2_old,
        gate,
        environment,
        bond,
        cell_Lx,
        cell_Ly;
        target_norm=target_norm,
        metric_floor=settings.metric_floor,
    )

    loss_current = objective(keep1_current, keep2_current)
    loss_initial = loss_current
    loss_history = Float64[loss_current]
    gradient_norm = Inf
    accepted_steps = 0
    last_iteration = 0

    # As in the triangular iPESS FU, optimize one local tensor while holding all
    # other local tensors fixed, then sweep to the next tensor.  The bosonic
    # square bond has two rank-5 tensors instead of the four reduced tensors of
    # a triangular iPESS block, and needs no swap/parity gates.
    for iteration in 1:settings.maxiter
        last_iteration = iteration
        loss_before_sweep = loss_current
        objective_1(keep1) = objective(keep1, keep2_current)
        keep1_current, _, gradient_norm_1, accepted_1 =
            _square_fu_cell_take_step(objective_1, keep1_current, settings)

        objective_2(keep2) = objective(keep1_current, keep2)
        keep2_current, loss_current, gradient_norm_2, accepted_2 =
            _square_fu_cell_take_step(objective_2, keep2_current, settings)
        gradient_norm = sqrt(gradient_norm_1^2 + gradient_norm_2^2)
        accepted = accepted_1 || accepted_2
        accepted_steps += Int(accepted_1) + Int(accepted_2)
        push!(loss_history, loss_current)
        gradient_norm <= settings.gradient_tolerance && break
        accepted || break
        loss_before_sweep - loss_current <= settings.loss_tolerance && break
    end

    A1_current, A2_current = _square_fu_reassemble_reduced(
        residual1, keep1_current, keep2_current, residual2, bond.direction,
    )
    A1_current = _square_fu_normalize(A1_current)
    A2_current = _square_fu_normalize(A2_current)
    old_bond_space = bond.direction === :x ? space(A1_old, 3) : space(A1_old, 2)
    new_bond_space = bond.direction === :x ? space(A1_current, 3) : space(A1_current, 2)
    report = merge(_square_fu_cell_report(
        bond, loss_initial, loss_current, last_iteration, accepted_steps,
        gradient_norm, target_norm, loss_history,
    ), (
        old_bond_space=old_bond_space,
        new_bond_space=new_bond_space,
        singular_space=space(singular_values, 1),
        bond_space_changed=old_bond_space != new_bond_space,
    ))
    if settings.verbose
        overlap_initial = sqrt(max(0.0, 1 - report.loss_initial))
        overlap_final = sqrt(max(0.0, report.fidelity))
        println("direct truncation:" * string(report.singular_space))
        println("overlap without optimization:" * string(overlap_initial))
        println("overlap with environmen after optimization:" * string(overlap_final))
        flush(stdout)
    end
    return A1_current, A2_current, report
end

function _square_fu_environment_cell(
    A_set::AbstractMatrix,
    environment_chi::Int,
    ctm_setting;
    initial_CTM=nothing,
)
    cell_Lx, cell_Ly = _square_fu_validate_cell(A_set)
    global Lx = cell_Lx
    global Ly = cell_Ly
    # Legacy projector helpers use `chi` as a module-global value.
    global chi = environment_chi
    A_cell = square_fu_cell_to_tuple(A_set)
    if isnothing(initial_CTM)
        init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
        CTM0 = []
    else
        init = initial_condition(init_type="PBC", reconstruct_CTM=false, reconstruct_AA=true)
        CTM0 = initial_CTM
    end
    result = CTMRG_cell(A_cell, environment_chi, init, CTM0, ctm_setting)
    if length(result) == 8
        CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err = result
    else
        CTM, AA, U_L, U_D, U_R, U_U = result
        ite_num, ite_err = missing, missing
    end
    return (
        CTM=CTM,
        AA=AA,
        U_L=U_L,
        U_D=U_D,
        U_R=U_R,
        U_U=U_U,
        ite_num=ite_num,
        ite_err=ite_err,
        Lx=cell_Lx,
        Ly=cell_Ly,
    )
end

function square_J1_full_update_cell_sweep(
    A_set::AbstractMatrix,
    chi::Int,
    gate::TensorMap,
    ctm_setting;
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
    bond_groups=nothing,
    initial_environment=nothing,
)
    cell_Lx, cell_Ly = _square_fu_validate_cell(A_set)
    settings.refresh_environment || throw(ArgumentError(
        "square cell Full Update requires refresh_environment=true: " *
        "CTMRG is reconstructed from scratch after every bond, as in the old triangular FU",
    ))
    groups = isnothing(bond_groups) ? square_J1_bond_groups(cell_Lx, cell_Ly) : bond_groups
    A_current = copy(A_set)
    environment = isnothing(initial_environment) ?
        _square_fu_environment_cell(A_current, chi, ctm_setting) : initial_environment
    if settings.verbose && isnothing(initial_environment)
        println(
            "ctm_ite_num= " * string(environment.ite_num) *
            ", ctm_ite_err= " * string(environment.ite_err),
        )
        flush(stdout)
    end
    reports = NamedTuple[]

    for (group_index, group) in pairs(groups)
        for bond in group
            A1_new, A2_new, report = square_J1_full_update_cell_bond(
                A_current, environment, gate, bond; settings=settings,
            )
            A_current[bond.site1] = A1_new
            A_current[bond.site2] = A2_new
            push!(reports, merge(report, (group=group_index,)))
            # Follow the original triangular-lattice FU literally: after
            # every local update, reconstruct the CTM from scratch.  Do not
            # reuse the old CTM even when the virtual spaces happen to be
            # unchanged, since the local tensors themselves have changed.
            environment = _square_fu_environment_cell(
                A_current, chi, ctm_setting; initial_CTM=nothing,
            )
            if settings.verbose
                println(
                    "ctm_ite_num= " * string(environment.ite_num) *
                    ", ctm_ite_err= " * string(environment.ite_err),
                )
                flush(stdout)
            end
        end
    end
    return A_current, environment, reports
end

"""
    square_J1_energy_cell(A_set, environment; J1=1)

Measure all positive-x and positive-y J1 bonds.  The returned total is energy
per site, while `Ex` and `Ey` retain one value for every cell anchor.
"""
function square_J1_energy_cell(A_set::AbstractMatrix, environment; J1::Real=1)
    cell_Lx, cell_Ly = _square_fu_validate_cell(A_set)
    H_Heisenberg, _, _, _, _ = Hamiltonians(space(A_set[1, 1], 1))
    H = permute(H_Heisenberg, (1, 2), (3, 4))
    Ex = zeros(Float64, cell_Lx, cell_Ly)
    Ey = zeros(Float64, cell_Lx, cell_Ly)
    for cx in 1:cell_Lx, cy in 1:cell_Ly
        for direction in (:x, :y)
            site1 = CartesianIndex(cx, cy)
            site2 = direction === :x ?
                CartesianIndex(mod1(cx + 1, cell_Lx), cy) :
                CartesianIndex(cx, mod1(cy + 1, cell_Ly))
            bond = SquareJ1CellBond(direction, site1, site2)
            A1, A2 = A_set[site1], A_set[site2]
            rho = _square_fu_two_site_density_cell(
                environment.CTM,
                A1,
                A1,
                A2,
                A2,
                bond,
                cell_Lx,
                cell_Ly,
            )
            norm_rho = real(@tensor rho[1, 2, 1, 2])
            norm_rho != 0 || throw(ArgumentError("zero norm for bond $direction at ($cx,$cy)"))
            energy = J1 * real(@tensor rho[1, 2, 3, 4] * H[1, 2, 3, 4]) / norm_rho
            direction === :x ? (Ex[cx, cy] = energy) : (Ey[cx, cy] = energy)
        end
    end
    return (energy_per_site=(sum(Ex) + sum(Ey)) / (cell_Lx * cell_Ly), Ex=Ex, Ey=Ey)
end

"""
    square_J1_full_update_cell(A_set, chi, tau, dt, ctm_setting; ...)

Run first-order imaginary-time Full Update on every periodic bond of an
arbitrary `Lx × Ly` bosonic square-lattice iPEPS cell.
"""
function square_J1_full_update_cell(
    A_set::AbstractMatrix,
    chi::Int,
    tau::Real,
    dt::Real,
    ctm_setting;
    J1::Real=1,
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
    callback=nothing,
)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    tau >= 0 || throw(ArgumentError("tau must be non-negative"))
    steps_float = tau / dt
    nsteps = round(Int, steps_float)
    isapprox(steps_float, nsteps; atol=1.0e-12, rtol=1.0e-12) ||
        throw(ArgumentError("tau/dt must be an integer, got $steps_float"))
    cell_Lx, cell_Ly = _square_fu_validate_cell(A_set)
    gate = prepare_gate_Heisenberg(J1 * dt, space(A_set[1, 1], 1))
    groups = square_J1_bond_groups(cell_Lx, cell_Ly)
    A_current = copy(A_set)
    environment = _square_fu_environment_cell(A_current, chi, ctm_setting)
    if settings.verbose
        println(
            "ctm_ite_num= " * string(environment.ite_num) *
            ", ctm_ite_err= " * string(environment.ite_err),
        )
        println("Periodic bond-group sizes: $(map(length, groups))")
        flush(stdout)
    end
    history = Vector{Vector{NamedTuple}}()
    for step in 1:nsteps
        settings.verbose && println("iteration " * string(step))
        A_current, environment, reports = square_J1_full_update_cell_sweep(
            A_current,
            chi,
            gate,
            ctm_setting;
            settings=settings,
            bond_groups=groups,
            initial_environment=environment,
        )
        push!(history, reports)
        isnothing(callback) || callback(A_current, environment, step, reports)
    end
    return A_current, environment, history
end
