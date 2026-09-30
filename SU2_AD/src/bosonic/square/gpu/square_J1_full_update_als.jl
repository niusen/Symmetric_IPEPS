"""
Bosonic square-lattice J1 Full Update using the same local optimization idea as
the triangular Hofstadter-Hubbard Full Update.

The two site tensors are first split into fixed rank-4 residual tensors and
variable rank-3 bond tensors.  The CTM network and the fixed residual tensors
are contracted once into a positive local metric.  The two rank-3 tensors are
then updated by alternating linear solves (ALS).  No automatic differentiation
is used in this file.
"""

_square_fu_als_reclaim(; aggressive=false) =
    isdefined(@__MODULE__, :square_J1_reclaim_device_memory!) ?
        square_J1_reclaim_device_memory!(; aggressive) : GC.gc()

function _square_fu_als_positive(S::DiagonalTensorMap; warning_tol=1.0e-2)
    S_positive = deepcopy(S)
    if sectortype(S) == Trivial
        threshold = zero(real(zero(eltype(S.data))))
        S_positive.data .= ifelse.(
            real.(S.data) .> threshold,
            real.(S.data),
            threshold,
        )
    else
        for (sector, values) in blocks(S)
            threshold = zero(real(zero(eltype(values))))
            block(S_positive, sector) .= ifelse.(
                real.(values) .> threshold,
                real.(values),
                threshold,
            )
        end
    end
    denominator = norm(S_positive)
    relative_change = denominator == 0 ? norm(S - S_positive) :
        norm(S - S_positive) / denominator
    if relative_change > warning_tol
        @warn "Square-J1 FU environment has a sizable negative component" relative_change
    end
    return S_positive
end

# As in the old triangular FU, absorb every fixed residual into its adjacent
# CTM corner/edge block before joining the blocks.  In particular, never form
# a boundary with all three fused PEPS legs open: it scales as chi^2*D^6.
# Reuse the fusion maps returned by CTMRG so the residual legs are expanded in
# exactly the basis used by the CTM tensors.
function _square_fu_als_residual_half(
    environment,
    residual,
    position::Symbol,
    cx::Int,
    cy::Int,
    cell_Lx::Int,
    cell_Ly::Int,
    site::CartesianIndex{2},
)
    CTM = environment.CTM
    sx, sy = Tuple(site)
    U_L = environment.U_L[sx][sy]
    U_D = environment.U_D[sx][sy]
    U_R = environment.U_R[sx][sy]
    U_U = environment.U_U[sx][sy]
    Cset, Tset = CTM.Cset, CTM.Tset

    half = if position === :x_left
        # residual=(L,D,U,a)
        @tensor outer[:] := Cset[mod1(cx, cell_Lx)][mod1(cy, cell_Ly)].C1[1, -1] *
            Tset[mod1(cx, cell_Lx)][mod1(cy + 1, cell_Ly)].T4[2, -2, 1] *
            Cset[mod1(cx, cell_Lx)][mod1(cy + 2, cell_Ly)].C4[-3, 2]
        @tensor side[:] := outer[-1, 1, -2] * U_L[1, -3, -4]
        outer = nothing
        @tensor top[:] :=
            Tset[mod1(cx + 1, cell_Lx)][mod1(cy, cell_Ly)].T1[-1, 1, -2] *
            U_U[-3, -4, 1]
        @tensor side_top[:] := side[1, -1, -3, -4] * top[1, -2, -5, -6]
        side = nothing
        top = nothing
        @tensor ket_block[:] := side_top[-1, -2, -3, 1, -4, 2] *
            residual[1, -5, 2, -6]
        side_top = nothing
        _square_fu_als_reclaim()
        @tensor bottom[:] :=
            Tset[mod1(cx + 1, cell_Lx)][mod1(cy + 2, cell_Ly)].T3[-1, 1, -2] *
            U_D[1, -3, -4]
        @tensor bra_boundary[:] := ket_block[1, -1, -3, -4, 2, -6] *
            bottom[-2, 1, -5, 2]
        ket_block = nothing
        bottom = nothing
        _square_fu_als_reclaim()
        # residual has tensor-index order (L,D,U,a); its adjoint has
        # tensor-index order (a,L,D,U).
        @tensor result[:] := bra_boundary[-1, -2, 1, 3, 2, -4] *
            residual'[-3, 1, 2, 3]
        result
    elseif position === :x_right
        # residual=(b,D,R,U)
        @tensor outer[:] := Cset[mod1(cx + 3, cell_Lx)][mod1(cy, cell_Ly)].C2[-1, 1] *
            Tset[mod1(cx + 3, cell_Lx)][mod1(cy + 1, cell_Ly)].T2[1, -2, 2] *
            Cset[mod1(cx + 3, cell_Lx)][mod1(cy + 2, cell_Ly)].C3[2, -3]
        @tensor side[:] := outer[-1, 1, -2] * U_R[-3, -4, 1]
        outer = nothing
        @tensor top[:] :=
            Tset[mod1(cx + 2, cell_Lx)][mod1(cy, cell_Ly)].T1[-2, 1, -1] *
            U_U[-3, -4, 1]
        @tensor side_top[:] := side[1, -1, -3, -4] * top[1, -2, -5, -6]
        side = nothing
        top = nothing
        @tensor ket_block[:] := side_top[-1, -2, -3, 1, -4, 2] *
            residual[-6, -5, 1, 2]
        side_top = nothing
        _square_fu_als_reclaim()
        @tensor bottom[:] :=
            Tset[mod1(cx + 2, cell_Lx)][mod1(cy + 2, cell_Ly)].T3[-1, 1, -2] *
            U_D[1, -3, -4]
        @tensor bra_boundary[:] := ket_block[1, -1, -3, -4, 2, -6] *
            bottom[1, -2, -5, 2]
        ket_block = nothing
        bottom = nothing
        _square_fu_als_reclaim()
        # residual has order (b,D,R,U); its adjoint has order (D,R,U,b).
        @tensor result[:] := bra_boundary[-1, -2, 3, 4, 2, -4] *
            residual'[2, 3, 4, -3]
        result
    elseif position === :y_upper
        # residual=(L,R,U,a)
        @tensor outer[:] := Cset[mod1(cx + 2, cell_Lx)][mod1(cy, cell_Ly)].C2[1, -1] *
            Tset[mod1(cx + 1, cell_Lx)][mod1(cy, cell_Ly)].T1[2, -2, 1] *
            Cset[mod1(cx, cell_Lx)][mod1(cy, cell_Ly)].C1[-3, 2]
        @tensor side[:] := outer[-1, 1, -2] * U_U[-3, -4, 1]
        outer = nothing
        @tensor right[:] :=
            Tset[mod1(cx + 2, cell_Lx)][mod1(cy + 1, cell_Ly)].T2[-1, 1, -2] *
            U_R[-3, -4, 1]
        @tensor side_right[:] := side[1, -1, -3, -4] * right[1, -2, -5, -6]
        side = nothing
        right = nothing
        @tensor ket_block[:] := side_right[-1, -2, -3, 1, -4, 2] *
            residual[-5, 2, 1, -6]
        side_right = nothing
        _square_fu_als_reclaim()
        @tensor left[:] :=
            Tset[mod1(cx, cell_Lx)][mod1(cy + 1, cell_Ly)].T4[-1, 1, -2] *
            U_L[1, -3, -4]
        @tensor bra_boundary[:] := ket_block[1, -1, -3, -4, 2, -6] *
            left[-2, 1, -5, 2]
        ket_block = nothing
        left = nothing
        _square_fu_als_reclaim()
        # residual has order (L,R,U,a); its adjoint has order (a,L,R,U).
        @tensor result[:] := bra_boundary[-1, -2, 3, 2, 1, -4] *
            residual'[-3, 1, 2, 3]
        result
    elseif position === :y_lower
        # residual=(b,L,D,R)
        @tensor outer[:] := Cset[mod1(cx + 2, cell_Lx)][mod1(cy + 3, cell_Ly)].C3[-1, 1] *
            Tset[mod1(cx + 1, cell_Lx)][mod1(cy + 3, cell_Ly)].T3[1, -2, 2] *
            Cset[mod1(cx, cell_Lx)][mod1(cy + 3, cell_Ly)].C4[2, -3]
        @tensor side[:] := outer[-1, 1, -2] * U_D[1, -3, -4]
        outer = nothing
        @tensor right[:] :=
            Tset[mod1(cx + 2, cell_Lx)][mod1(cy + 2, cell_Ly)].T2[-2, 1, -1] *
            U_R[-3, -4, 1]
        @tensor side_right[:] := side[1, -1, -3, -4] * right[1, -2, -5, -6]
        side = nothing
        right = nothing
        @tensor ket_block[:] := side_right[-1, -2, -3, 1, -4, 2] *
            residual[-6, -5, 1, 2]
        side_right = nothing
        _square_fu_als_reclaim()
        @tensor left[:] :=
            Tset[mod1(cx, cell_Lx)][mod1(cy + 2, cell_Ly)].T4[-1, 1, -2] *
            U_L[1, -3, -4]
        @tensor bra_boundary[:] := ket_block[1, -1, -3, -4, 2, -6] *
            left[1, -2, -5, 2]
        ket_block = nothing
        left = nothing
        _square_fu_als_reclaim()
        # residual has order (b,L,D,R); its adjoint has order (L,D,R,b).
        @tensor result[:] := bra_boundary[-1, -2, 3, 4, 2, -4] *
            residual'[2, 3, 4, -3]
        result
    else
        throw(ArgumentError("unknown residual position $position"))
    end
    bra_boundary = nothing
    _square_fu_als_reclaim()
    return half
end

function _square_fu_als_metric(
    environment,
    residual1,
    residual2,
    bond::SquareJ1CellBond,
    cell_Lx::Int,
    cell_Ly::Int,
)
    x1, y1 = Tuple(bond.site1)
    cx, cy = x1 - 1, y1 - 1
    if bond.direction === :x
        half1 = _square_fu_als_residual_half(
            environment, residual1, :x_left, cx, cy, cell_Lx, cell_Ly, bond.site1,
        )
        half2 = _square_fu_als_residual_half(
            environment, residual2, :x_right, cx, cy, cell_Lx, cell_Ly, bond.site2,
        )
    elseif bond.direction === :y
        half1 = _square_fu_als_residual_half(
            environment, residual1, :y_upper, cx, cy, cell_Lx, cell_Ly, bond.site1,
        )
        half2 = _square_fu_als_residual_half(
            environment, residual2, :y_lower, cx, cy, cell_Lx, cell_Ly, bond.site2,
        )
    else
        throw(ArgumentError("bond direction must be :x or :y"))
    end

    # A 2x1 (or 1x2) CTM cluster has only the two transverse boundary legs
    # shared by its two halves.  The other two legs of each half are the open
    # bra/ket reduced-SVD indices optimized below.
    @tensor metric[:] := half1[1, 2, -1, -3] *
        half2[1, 2, -2, -4]
    half1 = nothing
    half2 = nothing
    _square_fu_als_reclaim()
    metric = permute(metric, (1, 2), (3, 4))
    metric = (metric + metric') / 2
    return metric
end

function _square_fu_als_environment(metric)
    eigenvalues, eigenvectors = eigh(metric)
    eigenvalues = _square_fu_als_positive(eigenvalues)
    env_bot = sqrt(eigenvalues) * eigenvectors'
    metric = nothing
    eigenvalues = nothing
    eigenvectors = nothing
    _square_fu_als_reclaim(; aggressive=true)
    return env_bot
end

function _square_fu_als_reduced_pair(keep1, keep2)
    @tensor pair[:] := keep1[-1, 1, -2] * keep2[1, -4, -3]
    return permute(pair, (1, 2), (3, 4))
end

function _square_fu_als_project(env_bot, pair)
    @tensor projected[:] := env_bot[-1, 1, 2] * pair[1, -2, 2, -3]
    return projected
end

function _square_fu_als_fidelity(env_bot, target_projected, target_norm, keep1, keep2)
    candidate = _square_fu_als_reduced_pair(keep1, keep2)
    candidate_projected = _square_fu_als_project(env_bot, candidate)
    candidate_norm = real(dot(candidate_projected, candidate_projected))
    overlap = dot(candidate_projected, target_projected)
    fidelity = candidate_norm == 0 || target_norm == 0 ? 0.0 :
        real(abs2(overlap) / abs(candidate_norm * target_norm))
    candidate = nothing
    candidate_projected = nothing
    return fidelity
end

function _square_fu_als_solve(rho, rightside, output_partition)
    rho = permute(rho, (1, 2, 3), (4, 5, 6))
    rho = (rho + rho') / 2
    eigenvalues, eigenvectors = eigh(rho)
    eigenvalues = _square_fu_als_positive(eigenvalues)
    rho_inverse = eigenvectors * my_pinv(eigenvalues) * eigenvectors'
    @tensor updated[:] := rho_inverse[-1, -2, -3, 1, 2, 3] * rightside[1, 2, 3]
    updated = permute(updated, output_partition[1], output_partition[2])
    rho = nothing
    rho_inverse = nothing
    eigenvalues = nothing
    eigenvectors = nothing
    _square_fu_als_reclaim()
    return updated
end

function _square_fu_als_update_keep1(env_bot, target_projected, keep1, keep2)
    @tensor partial[:] := env_bot[-1, -2, 1] * keep2[-3, -4, 1]
    identity_physical = _square_J1_unitary_like(
        keep1, space(keep1, 3), space(keep1, 3),
    )
    @tensor rho[:] := partial'[1, -1, -2, 2] * partial[1, -4, -5, 2] *
        identity_physical[-3, -6]
    @tensor rightside[:] := partial'[1, -1, -2, 2] * target_projected[1, -3, 2]
    partial = nothing
    identity_physical = nothing
    return _square_fu_als_solve(rho, rightside, ((1,), (2, 3)))
end

function _square_fu_als_update_keep2(env_bot, target_projected, keep1, keep2)
    @tensor partial[:] := env_bot[-1, 1, -4] * keep1[1, -2, -3]
    identity_physical = _square_J1_unitary_like(
        keep2, space(keep2, 2), space(keep2, 2),
    )
    @tensor rho[:] := partial'[1, -1, 2, -3] * partial[1, -4, 2, -6] *
        identity_physical[-2, -5]
    @tensor rightside[:] := partial'[1, -1, 2, -3] * target_projected[1, 2, -2]
    partial = nothing
    identity_physical = nothing
    return _square_fu_als_solve(rho, rightside, ((1, 2), (3,)))
end

function _square_fu_als_balance(keep1, keep2, kept_space)
    pair = _square_fu_als_reduced_pair(keep1, keep2)
    balanced1, balanced2, _ = _square_fu_factor_bond(
        pair; truncation=truncspace(kept_space),
    )
    pair = nothing
    return balanced1, balanced2
end

"""One square-lattice bond Full Update by reduced-tensor ALS, without AD."""
function square_J1_full_update_cell_bond_als(
    A_set::AbstractMatrix,
    environment,
    gate::TensorMap,
    bond::SquareJ1CellBond;
    settings::SquareJ1FullUpdateSettings=SquareJ1FullUpdateSettings(),
)
    cell_Lx, cell_Ly = size(A_set)
    A1_old, A2_old = A_set[bond.site1], A_set[bond.site2]
    bond.site1 != bond.site2 || throw(ArgumentError(
        "bond-expanding Full Update requires two distinct tensor entries",
    ))

    residual1, old_keep1, old_keep2, residual2 =
        _square_fu_split_reduced(A1_old, A2_old, bond.direction)
    target = _square_fu_gated_bond(old_keep1, old_keep2, gate)
    truncation = truncdim(settings.Dmax; multiplet_tol=settings.multiplet_tol)
    keep1, keep2, singular_values = _square_fu_factor_bond(
        target; truncation=truncation,
    )
    kept_space = space(singular_values, 1)

    metric = _square_fu_als_metric(
        environment, residual1, residual2, bond, cell_Lx, cell_Ly,
    )
    env_bot = _square_fu_als_environment(metric)
    target_projected = _square_fu_als_project(env_bot, target)
    target_norm = real(dot(target_projected, target_projected))
    target_norm > settings.metric_floor || throw(ArgumentError(
        "the reduced two-site CTM target norm is too small: $target_norm",
    ))

    fidelity = _square_fu_als_fidelity(
        env_bot, target_projected, target_norm, keep1, keep2,
    )
    fidelity_initial = fidelity
    loss_history = Float64[1 - fidelity]
    last_iteration = 0
    accepted_sweeps = 0

    for iteration in 1:settings.maxiter
        keep1_before, keep2_before = keep1, keep2
        fidelity_before = fidelity
        keep1 = _square_fu_als_update_keep1(env_bot, target_projected, keep1, keep2)
        keep2 = _square_fu_als_update_keep2(env_bot, target_projected, keep1, keep2)
        keep1, keep2 = _square_fu_als_balance(keep1, keep2, kept_space)
        fidelity = _square_fu_als_fidelity(
            env_bot, target_projected, target_norm, keep1, keep2,
        )
        last_iteration = iteration

        # An exact ALS solve cannot lower the fidelity.  Revert a sweep if
        # roundoff or an ill-conditioned normal equation violates this.
        if fidelity + max(settings.loss_tolerance, 1.0e-12) < fidelity_before
            keep1, keep2 = keep1_before, keep2_before
            fidelity = fidelity_before
            break
        end
        accepted_sweeps += 1
        push!(loss_history, 1 - fidelity)
        settings.verbose && print(string(sqrt(max(0.0, fidelity))) * " , ")
        abs(fidelity - fidelity_before) <= settings.loss_tolerance && break
    end
    settings.verbose && println()

    A1_new, A2_new = _square_fu_reassemble_reduced(
        residual1, keep1, keep2, residual2, bond.direction,
    )
    A1_new = _square_fu_normalize(A1_new)
    A2_new = _square_fu_normalize(A2_new)
    old_bond_space = bond.direction === :x ? space(A1_old, 3) : space(A1_old, 2)
    new_bond_space = bond.direction === :x ? space(A1_new, 3) : space(A1_new, 2)
    report = (
        direction=bond.direction,
        site1=Tuple(bond.site1),
        site2=Tuple(bond.site2),
        loss_initial=1 - fidelity_initial,
        loss_final=1 - fidelity,
        fidelity=fidelity,
        iterations=last_iteration,
        accepted_steps=accepted_sweeps,
        gradient_norm=NaN,
        target_norm=target_norm,
        loss_history=loss_history,
        old_bond_space=old_bond_space,
        new_bond_space=new_bond_space,
        singular_space=space(singular_values, 1),
        bond_space_changed=old_bond_space != new_bond_space,
        optimizer=:als,
    )
    if settings.verbose
        println("direct truncation:" * string(report.singular_space))
        println("overlap without optimization:" * string(sqrt(max(0.0, fidelity_initial))))
        println("overlap with environmen after optimization:" *
            string(sqrt(max(0.0, fidelity))))
        flush(stdout)
    end
    return A1_new, A2_new, report
end
