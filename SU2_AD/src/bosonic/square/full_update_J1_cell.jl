"""
Full Update for a bosonic square-lattice nearest-neighbour J1 model on an
arbitrary periodic `Lx × Ly` iPEPS unit cell.

This file reuses the one-site local fidelity machinery from
`full_update_J1.jl` and the periodic-cell/CTM organization used by the
triangular-lattice fermionic Full Update.  Include, in order,
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
        _square_fu_check_tensor_pair(A_reference, A)
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
    AA_1, U_physical_1 = build_square_cross_double_layer_open(A_bra_1, A_ket_1)
    AA_2, U_physical_2 = build_square_cross_double_layer_open(A_bra_2, A_ket_2)
    x1, y1 = Tuple(bond.site1)
    anchor_x, anchor_y = x1 - 1, y1 - 1

    if bond.direction === :x
        rho_fused = _square_fu_ob_2sites_x_cell(
            anchor_x,
            anchor_y,
            CTM,
            AA_1,
            AA_2,
            cell_Lx,
            cell_Ly,
        )
    elseif bond.direction === :y
        rho_fused = _square_fu_ob_2sites_y_cell(
            anchor_x,
            anchor_y,
            CTM,
            AA_1,
            AA_2,
            cell_Lx,
            cell_Ly,
        )
    else
        throw(ArgumentError("bond direction must be :x or :y"))
    end

    @tensor rho[:] := rho_fused[1, 2] *
        U_physical_1[-1, -3, 1] *
        U_physical_2[-2, -4, 2]
    return rho
end

function _square_fu_cell_target_norm(A1_old, A2_old, environment, gate, bond, cell_Lx, cell_Ly)
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
    return real(@tensor rho_old[1, 2, 3, 4] * gate_norm[1, 2, 3, 4])
end

function square_J1_cell_fidelity(
    A1_new,
    A2_new,
    A1_old,
    A2_old,
    environment,
    gate,
    bond::SquareJ1CellBond,
    cell_Lx::Int,
    cell_Ly::Int;
    target_norm=nothing,
    metric_floor::Real=0.0,
)
    rho_new = _square_fu_two_site_density_cell(
        environment.CTM,
        A1_new,
        A1_new,
        A2_new,
        A2_new,
        bond,
        cell_Lx,
        cell_Ly,
    )
    norm_new = real(@tensor rho_new[1, 2, 1, 2])
    rho_overlap = _square_fu_two_site_density_cell(
        environment.CTM,
        A1_new,
        A1_old,
        A2_new,
        A2_old,
        bond,
        cell_Lx,
        cell_Ly,
    )
    overlap = @tensor rho_overlap[1, 2, 3, 4] * gate[1, 2, 3, 4]
    norm_target = isnothing(target_norm) ?
        _square_fu_cell_target_norm(A1_old, A2_old, environment, gate, bond, cell_Lx, cell_Ly) :
        target_norm
    denominator = abs(norm_new * norm_target)
    denominator > metric_floor ||
        throw(ArgumentError("the cell CTM fidelity denominator is non-positive or too small: $denominator"))
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
    target_norm = _square_fu_cell_target_norm(
        A1_old, A2_old, environment, gate, bond, cell_Lx, cell_Ly,
    )
    target_norm > 0 || throw(ArgumentError("the cell CTM target norm is non-positive: $target_norm"))

    objective(A1, A2) = 1 - square_J1_cell_fidelity(
        A1,
        A2,
        A1_old,
        A2_old,
        environment,
        gate,
        bond,
        cell_Lx,
        cell_Ly;
        target_norm=target_norm,
        metric_floor=settings.metric_floor,
    )

    same_tensor = bond.site1 == bond.site2
    A1_current = _square_fu_normalize(A1_old)
    A2_current = same_tensor ? A1_current : _square_fu_normalize(A2_old)
    shared_objective(A) = objective(A, A)
    loss_current = same_tensor ? shared_objective(A1_current) : objective(A1_current, A2_current)
    loss_initial = loss_current
    loss_history = Float64[loss_current]
    gradient_norm = Inf
    accepted_steps = 0
    last_iteration = 0

    settings.verbose && println(
        "FU cell $(bond.direction) $(Tuple(bond.site1))→$(Tuple(bond.site2)): initial loss=$loss_current",
    )
    # As in the triangular iPESS FU, optimize one local tensor while holding all
    # other local tensors fixed, then sweep to the next tensor.  The bosonic
    # square bond has two rank-5 tensors instead of the four reduced tensors of
    # a triangular iPESS block, and needs no swap/parity gates.
    for iteration in 1:settings.maxiter
        last_iteration = iteration
        loss_before_sweep = loss_current
        if same_tensor
            A1_current, loss_current, gradient_norm, accepted =
                _square_fu_cell_take_step(shared_objective, A1_current, settings)
            A2_current = A1_current
            accepted_steps += Int(accepted)
        else
            objective_1(A1) = objective(A1, A2_current)
            A1_current, _, gradient_norm_1, accepted_1 =
                _square_fu_cell_take_step(objective_1, A1_current, settings)

            objective_2(A2) = objective(A1_current, A2)
            A2_current, loss_current, gradient_norm_2, accepted_2 =
                _square_fu_cell_take_step(objective_2, A2_current, settings)
            gradient_norm = sqrt(gradient_norm_1^2 + gradient_norm_2^2)
            accepted = accepted_1 || accepted_2
            accepted_steps += Int(accepted_1) + Int(accepted_2)
        end
        push!(loss_history, loss_current)
        settings.verbose && println(
            "FU cell $(bond.direction) $(Tuple(bond.site1)) alternating sweep $iteration: " *
            "loss=$loss_current, |grad|=$gradient_norm",
        )
        gradient_norm <= settings.gradient_tolerance && break
        accepted || break
        loss_before_sweep - loss_current <= settings.loss_tolerance && break
    end

    report = _square_fu_cell_report(
        bond, loss_initial, loss_current, last_iteration, accepted_steps,
        gradient_norm, target_norm, loss_history,
    )
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
    groups = isnothing(bond_groups) ? square_J1_bond_groups(cell_Lx, cell_Ly) : bond_groups
    A_current = copy(A_set)
    environment = isnothing(initial_environment) ?
        _square_fu_environment_cell(A_current, chi, ctm_setting) : initial_environment
    reports = NamedTuple[]

    for (group_index, group) in pairs(groups)
        settings.verbose && println("FU cell bond group $group_index/$(length(groups)), $(length(group)) bonds")
        for bond in group
            A1_new, A2_new, report = square_J1_full_update_cell_bond(
                A_current, environment, gate, bond; settings=settings,
            )
            A_current[bond.site1] = A1_new
            A_current[bond.site2] = A2_new
            push!(reports, merge(report, (group=group_index,)))
            if settings.refresh_environment
                environment = _square_fu_environment_cell(
                    A_current, chi, ctm_setting; initial_CTM=environment.CTM,
                )
            end
        end
    end
    if !settings.refresh_environment
        environment = _square_fu_environment_cell(
            A_current, chi, ctm_setting; initial_CTM=environment.CTM,
        )
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
    environment = nothing
    A_current = copy(A_set)
    history = Vector{Vector{NamedTuple}}()
    for step in 1:nsteps
        settings.verbose && println(
            "square J1 $(cell_Lx)×$(cell_Ly) FU sweep $step/$nsteps, dt=$dt",
        )
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
    if nsteps == 0
        environment = _square_fu_environment_cell(A_current, chi, ctm_setting)
    end
    return A_current, environment, history
end
